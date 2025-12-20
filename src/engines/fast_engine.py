import time
import logging
import traceback
import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QThread, pyqtSignal
from faster_whisper import WhisperModel

# Import Config
from config import ASRConfig

# Configure local logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FastModelLoader(QThread):
    """
    Loads the Faster-Whisper model using CTranslate2 backend.
    Optimized for RTX 3050 Ti (4GB VRAM) using INT8 quantization.
    """
    finished_loading = pyqtSignal(object, object) 
    log_message = pyqtSignal(str)

    def __init__(self, model_id_str="faster/large-v3"):
        super().__init__()
        # Strip prefix to get clean model name
        self.model_size = model_id_str.replace("faster/", "")

    def run(self):
        self.log_message.emit(f"🚀 [Engine] Initializing: {self.model_size}")
        try:
            start_time = time.time()
            
            # Load Model with INT8 Optimization
            model = WhisperModel(
                self.model_size, 
                device="cuda", 
                device_index=0, 
                compute_type="int8" 
            )
            
            duration = time.time() - start_time
            self.log_message.emit(f"✅ Engine Ready! ({duration:.2f}s)")
            self.finished_loading.emit(model, None)
            
        except Exception as e:
            self.log_message.emit(f"❌ Engine Load Error: {str(e)}")
            self.finished_loading.emit(None, None)

class RealTimeTranscriber(QThread):
    """
    Captures live audio stream, calculates volume levels, and transcribes.
    """
    partial_transcript = pyqtSignal(str)
    log_message = pyqtSignal(str)
    volume_level = pyqtSignal(int) # Signal to update UI VU Meter
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_running = False
        self.chunk_duration = 3.0  # Process every 3 seconds
        # Low threshold to detect even whispers
        self.silence_threshold = 0.005 

    def stop(self):
        self.is_running = False

    def run(self):
        if not self.model: return
        self.is_running = True
        self.log_message.emit(f"🎙️ Live Listening Started (Mic ID: {ASRConfig.MIC_DEVICE_ID})...")
        
        sample_rate = 16000
        chunk_samples = int(sample_rate * self.chunk_duration)
        audio_buffer = []

        # --- AUDIO CALLBACK ---
        def audio_callback(indata, frames, time, status):
            if status: print(f"Audio Status: {status}")
            
            # 1. Copy data for transcription
            audio_buffer.append(indata.copy())
            
            # 2. Calculate Volume for UI (Visual Feedback)
            rms = np.sqrt(np.mean(indata**2))
            
            # Amplify signal for better visibility (0-100 scale)
            vol = int(min(rms * 500, 100)) 
            self.volume_level.emit(vol) 

        try:
            # IMPORTANT: Uses the updated ASRConfig.MIC_DEVICE_ID
            with sd.InputStream(samplerate=sample_rate, 
                                channels=1, 
                                callback=audio_callback, 
                                device=ASRConfig.MIC_DEVICE_ID):
                
                while self.is_running:
                    # Check buffer size
                    current_len = sum(len(x) for x in audio_buffer)
                    
                    if current_len >= chunk_samples:
                        # Process Buffer
                        data = np.concatenate(audio_buffer, axis=0).flatten().astype(np.float32)
                        audio_buffer = [] # Reset

                        # Silence Detection
                        avg_rms = np.sqrt(np.mean(data**2))
                        
                        if avg_rms > self.silence_threshold:
                            # Transcribe
                            segments, _ = self.model.transcribe(
                                data, 
                                beam_size=5, 
                                language="en", 
                                vad_filter=True
                            )
                            text = " ".join([s.text for s in segments]).strip()
                            
                            if text:
                                self.partial_transcript.emit(text)
                    
                    time.sleep(0.1)
                    
        except Exception as e:
            self.log_message.emit(f"❌ Stream Error: {e}")
            logging.error(traceback.format_exc())
            
        self.log_message.emit("🛑 Live Listening Stopped.")