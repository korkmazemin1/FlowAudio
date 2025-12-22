import time
import logging
import traceback
import queue
import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QThread, pyqtSignal
from faster_whisper import WhisperModel

# Import Global Configuration
from config import ASRConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# WORKER THREAD (The GPU Laborer)
# ==========================================
class WhisperWorker(QThread):
    """
    Represents a single GPU worker instance.
    It runs in its own thread, loads its own instance of the model,
    and processes audio tasks from the input queue independently.
    """
    def __init__(self, worker_id, model_size, input_queue, output_queue):
        super().__init__()
        self.worker_id = worker_id
        self.model_size = model_size
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.is_running = True

    def run(self):
        logging.info(f"👷 [Worker-{self.worker_id}] Initializing Model: {self.model_size}...")
        try:
            # Load Model with INT8 Optimization (Crucial for VRAM efficiency)
            # We use 'int8' so MULTIPLE models can fit into the VRAM.
            model = WhisperModel(
                self.model_size, 
                device="cuda", 
                device_index=0, 
                compute_type="int8" 
            )
            logging.info(f"✅ [Worker-{self.worker_id}] Ready for action!")
            
            while self.is_running:
                try:
                    # Get job from queue (Wait up to 1 sec, then check is_running)
                    # Task tuple: (sequence_id, audio_data, context_prompt, capture_timestamp)
                    task = self.input_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Unpack the task including the capture timestamp
                seq_id, audio_data, prompt, capture_time = task
                
                # --- INFERENCE ---
                # This blocks only this worker thread, others continue processing.
                segments, _ = model.transcribe(
                    audio_data,
                    beam_size=5,
                    language=None, # Auto-detect language
                    vad_filter=True,
                    initial_prompt=prompt,
                    condition_on_previous_text=True
                )
                
                text = " ".join([s.text for s in segments]).strip()
                
                # Pass the result AND the original capture time back to the main thread
                self.output_queue.put((seq_id, text, capture_time))
                self.input_queue.task_done()
                
        except Exception as e:
            logging.error(f"❌ [Worker-{self.worker_id}] Crash: {e}")

    def stop(self):
        """Stops the worker loop safely."""
        self.is_running = False

# ==========================================
# ORCHESTRATOR (The Manager)
# ==========================================
class MultiThreadedTranscriber(QThread):
    """
    Manages the Parallel Pipeline:
    1. Captures Audio (Producer) -> Input Queue
    2. Spawns 3 Workers (Consumers) -> GPU Processing
    3. Reorders Results (Sequencer) -> UI Output
    4. Calculates Latency Telemetry
    """
    partial_transcript = pyqtSignal(str)
    log_message = pyqtSignal(str)
    volume_level = pyqtSignal(int)
    
    # New Signal: (capture_time, finish_time, latency, avg_latency)
    latency_data = pyqtSignal(float, float, float, float)
    
    def __init__(self):
        super().__init__()
        # --- HARDCODED CONFIGURATION FOR 3 WORKERS ---
        self.model_size = "medium"  # Balanced for 3 workers
        self.num_workers = 3        
        self.is_running = False
        
        # Communication Queues
        self.input_queue = queue.Queue()  
        self.output_queue = queue.Queue() 
        
        # Workers List
        self.workers = []
        
        # Audio Settings
        self.chunk_duration = 3.0      
        self.silence_threshold = 0.005 
        
        # Sequencing Logic
        self.sequence_counter = 0     
        self.next_expected_id = 0     
        self.result_buffer = {}       
        self.global_context = ""      

        # Telemetry Data
        self.total_latency = 0.0
        self.total_chunks = 0

    def start_workers(self):
        """Spins up the GPU worker threads."""
        self.log_message.emit(f"🚀 Spawning {self.num_workers} Parallel Workers (Model: {self.model_size})...")
        for i in range(self.num_workers):
            worker = WhisperWorker(
                worker_id=i+1, 
                model_size=self.model_size, 
                input_queue=self.input_queue, 
                output_queue=self.output_queue
            )
            worker.start()
            self.workers.append(worker)

    def stop(self):
        """Stops the entire pipeline and all workers."""
        self.is_running = False
        for w in self.workers:
            w.stop()
            w.wait()
        self.workers = []

    def run(self):
        self.start_workers()
        self.is_running = True
        self.log_message.emit(f"🎙️ Pipeline Active ({self.num_workers} Workers). Telemetry Enabled.")
        
        sample_rate = 16000
        chunk_samples = int(sample_rate * self.chunk_duration)
        audio_buffer = []
        
        def audio_callback(indata, frames, time_info, status):
            if status: print(status)
            audio_buffer.append(indata.copy())
            rms = np.sqrt(np.mean(indata**2))
            self.volume_level.emit(int(min(rms * 500, 100)))

        try:
            with sd.InputStream(samplerate=sample_rate, 
                                channels=1, 
                                callback=audio_callback, 
                                device=ASRConfig.MIC_DEVICE_ID):
                
                while self.is_running:
                    # 1. AUDIO CAPTURE & DISPATCH LOGIC
                    current_len = sum(len(x) for x in audio_buffer)
                    if current_len >= chunk_samples:
                        data = np.concatenate(audio_buffer, axis=0).flatten().astype(np.float32)
                        audio_buffer = [] 

                        if np.sqrt(np.mean(data**2)) > self.silence_threshold:
                            prompt = self.global_context[-200:] if self.global_context else None
                            
                            # Capture the exact time we are sending this to the GPU
                            capture_time = time.time()
                            
                            # DISPATCH JOB: Include capture_time in the task
                            self.input_queue.put((self.sequence_counter, data, prompt, capture_time))
                            self.sequence_counter += 1
                    
                    # 2. SEQUENCER LOGIC (Re-ordering Output)
                    try:
                        while not self.output_queue.empty():
                            # Retrieve result + original capture time
                            res_id, res_text, res_capture_time = self.output_queue.get_nowait()
                            
                            if res_text:
                                self.result_buffer[res_id] = (res_text, res_capture_time)

                            # Check for next expected sentence
                            while self.next_expected_id in self.result_buffer:
                                final_text, original_time = self.result_buffer.pop(self.next_expected_id)
                                
                                # Calculate Telemetry
                                finish_time = time.time()
                                latency = finish_time - original_time
                                
                                # Update Average
                                self.total_latency += latency
                                self.total_chunks += 1
                                avg_latency = self.total_latency / self.total_chunks
                                
                                # Emit Telemetry Data
                                self.latency_data.emit(original_time, finish_time, latency, avg_latency)
                                
                                # Update Context & UI
                                self.global_context += " " + final_text
                                self.partial_transcript.emit(final_text)
                                self.next_expected_id += 1
                                
                    except queue.Empty:
                        pass
                    
                    time.sleep(0.01)

        except Exception as e:
            self.log_message.emit(f"❌ Pipeline Error: {e}")
            logging.error(traceback.format_exc())
            
        self.stop()
        self.log_message.emit("🛑 Pipeline Stopped.")