import time
import logging
import traceback
import queue
import numpy as np
import sounddevice as sd
import torch
import gc 
from PyQt6.QtCore import QThread, pyqtSignal
from faster_whisper import WhisperModel

# --- OFFLINE TRANSLATION LIBRARIES ---
# AutoModel kullanarak daha esnek oluyoruz (NLLB desteği için)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import ASRConfig

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("flow_audio_debug.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Engine")

# ==========================================
# WORKER THREAD (Whisper - GPU)
# ==========================================
class WhisperWorker(QThread):
    def __init__(self, worker_id, model_size, input_queue, output_queue):
        super().__init__()
        self.worker_id = worker_id
        self.model_size = model_size
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.is_running = True

    def run(self):
        logger.info(f"👷 [Worker-{self.worker_id}] Initializing Whisper: {self.model_size}...")
        try:
            # Load Model with INT8 Optimization
            model = WhisperModel(
                self.model_size, 
                device="cuda", 
                device_index=0, 
                compute_type="int8" 
            )
            logger.info(f"✅ [Worker-{self.worker_id}] Ready!")
            
            while self.is_running:
                try:
                    task = self.input_queue.get(timeout=1)
                except queue.Empty:
                    continue

                seq_id, audio_data, prompt, capture_time = task
                
                # Inference
                segments, _ = model.transcribe(
                    audio_data,
                    beam_size=5,
                    language="en", 
                    vad_filter=True,
                    initial_prompt=prompt,
                    condition_on_previous_text=True
                )
                
                text = " ".join([s.text for s in segments]).strip()
                
                self.output_queue.put((seq_id, text, capture_time))
                self.input_queue.task_done()
                
        except Exception as e:
            logger.error(f"❌ [Worker-{self.worker_id}] Crash: {e}")

    def stop(self):
        self.is_running = False

# ==========================================
# ORCHESTRATOR (Manager + Translator)
# ==========================================
class MultiThreadedTranscriber(QThread):
    partial_transcript = pyqtSignal(str)
    log_message = pyqtSignal(str)
    volume_level = pyqtSignal(int)
    latency_data = pyqtSignal(float, float, float, float)
    
    def __init__(self):
        super().__init__()
        # --- CONFIGURATION ---
        self.model_size = "medium" 
        self.num_workers = 2        
        self.is_running = False
        
        self.input_queue = queue.Queue()  
        self.output_queue = queue.Queue() 
        
        self.workers = []
        self.chunk_duration = 3.0      
        self.silence_threshold = 0.005 
        
        self.sequence_counter = 0     
        self.next_expected_id = 0     
        self.result_buffer = {}       
        self.global_context = ""      

        self.total_latency = 0.0
        self.total_chunks = 0
        
        # --- NEW MODEL CONFIGURATION (NLLB-600M) ---
        # Bu model 'Safetensors' formatını destekler, eski Torch ile çalışır.
        self.tr_model_name = "facebook/nllb-200-distilled-600M" 
        self.tokenizer = None
        self.tr_model = None

    def load_translator(self):
        try:
            logger.info("🧹 Cleaning VRAM before loading Translator...")
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info(f"🌍 Downloading/Loading Translator: {self.tr_model_name}")
            self.log_message.emit("🌍 Downloading AI Model (Safetensors)...")
            
            # 1. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.tr_model_name)
            
            # 2. Load Model (CRITICAL OPTIMIZATION: float16)
            # torch_dtype=torch.float16 -> Modeli %50 küçültür (1.2GB -> 600MB)
            # use_safetensors=True    -> Eski Torch hatasını atlatır (Güvenli yükleme)
            self.tr_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.tr_model_name, 
                torch_dtype=torch.float16, 
                use_safetensors=True
            ).to("cuda")
            
            self.tr_model.eval()
            
            logger.info("✅ Translator Ready (FP16 Optimized)!")
            self.log_message.emit("✅ Translator Ready (GPU FP16)!")
            
        except Exception as e:
            logger.error(f"❌ Translator Load Error: {e}")
            self.log_message.emit(f"❌ Load Error: {str(e)[:50]}...")

    def start_workers(self):
        # 1. Load Translator
        self.load_translator()
        
        # 2. Start Whisper
        logger.info(f"🚀 Spawning {self.num_workers} Whisper Workers...")
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
        self.is_running = False
        for w in self.workers:
            w.stop()
            w.wait()
        self.workers = []
        
        if self.tr_model:
            del self.tr_model
            del self.tokenizer
            torch.cuda.empty_cache()

    def run(self):
        self.start_workers()
        self.is_running = True
        logger.info("🎙️ SYSTEM LIVE: Listening...")
        
        sample_rate = 16000
        chunk_samples = int(sample_rate * self.chunk_duration)
        audio_buffer = []
        
        def audio_callback(indata, frames, time_info, status):
            if status: logger.warning(status)
            audio_buffer.append(indata.copy())
            rms = np.sqrt(np.mean(indata**2))
            self.volume_level.emit(int(min(rms * 500, 100)))

        try:
            with sd.InputStream(samplerate=sample_rate, 
                                channels=1, 
                                callback=audio_callback, 
                                device=ASRConfig.MIC_DEVICE_ID):
                
                while self.is_running:
                    # 1. CAPTURE
                    current_len = sum(len(x) for x in audio_buffer)
                    if current_len >= chunk_samples:
                        data = np.concatenate(audio_buffer, axis=0).flatten().astype(np.float32)
                        audio_buffer = [] 

                        if np.sqrt(np.mean(data**2)) > self.silence_threshold:
                            prompt = self.global_context[-200:] if self.global_context else None
                            capture_time = time.time()
                            self.input_queue.put((self.sequence_counter, data, prompt, capture_time))
                            self.sequence_counter += 1
                    
                    # 2. PROCESS
                    try:
                        while not self.output_queue.empty():
                            res_id, res_text_en, res_capture_time = self.output_queue.get_nowait()
                            
                            if res_text_en:
                                self.result_buffer[res_id] = (res_text_en, res_capture_time)

                            while self.next_expected_id in self.result_buffer:
                                eng_text, original_time = self.result_buffer.pop(self.next_expected_id)
                                
                                # TRANSLATE (NLLB LOGIC)
                                tr_text = eng_text 
                                if self.tr_model and self.tokenizer and eng_text.strip():
                                    try:
                                        # NLLB requires specifying the target language code
                                        # tur_Latn = Turkish
                                        inputs = self.tokenizer(eng_text, return_tensors="pt", padding=True).to("cuda")
                                        
                                        with torch.no_grad():
                                            translated = self.tr_model.generate(
                                                **inputs, 
                                                forced_bos_token_id=self.tokenizer.lang_code_to_id["tur_Latn"],
                                                max_new_tokens=100
                                            )
                                        tr_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
                                    except Exception as e:
                                        logger.error(f"Translation Fail: {e}")

                                # Stats
                                finish_time = time.time()
                                latency = finish_time - original_time
                                self.total_latency += latency
                                self.total_chunks += 1
                                avg = self.total_latency / self.total_chunks
                                self.latency_data.emit(original_time, finish_time, latency, avg)
                                
                                # Keep English Context for Whisper
                                self.global_context += " " + eng_text
                                
                                # Emit TR
                                self.partial_transcript.emit(tr_text)
                                self.next_expected_id += 1
                                
                    except queue.Empty:
                        pass
                    
                    time.sleep(0.005)

        except Exception as e:
            self.log_message.emit(f"❌ Error: {e}")
            logging.error(traceback.format_exc())
            
        self.stop()