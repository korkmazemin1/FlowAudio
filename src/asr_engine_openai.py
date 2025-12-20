import time
import numpy as np
import torch
import sounddevice as sd
import psutil
from PyQt6.QtCore import QThread, pyqtSignal
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# ==========================================
# SHARED CONFIGURATION
# ==========================================
class ASRConfig:
    """
    Central configuration for hardware and audio settings.
    Shared across both engines.
    """
    SAMPLE_RATE = 16000
    MIC_DEVICE_ID = 2  # IMPORTANT: Your specific Mic ID
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ==========================================
# SHARED SYSTEM MONITOR
# ==========================================
class SystemMonitor:
    @staticmethod
    def get_stats():
        ram = psutil.virtual_memory().percent
        vram_msg = "VRAM: N/A"
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**3
            vram_msg = f"VRAM: {vram:.2f} GB"
        return f"RAM: {ram}% | {vram_msg}"

# ==========================================
# SHARED AUDIO WORKERS (Recorder & Player)
# ==========================================
class AudioRecorder(QThread):
    recording_finished = pyqtSignal(np.ndarray) 
    log_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.audio_buffer = []

    def run(self):
        self.is_recording = True
        self.audio_buffer = [] 
        self.log_message.emit(f"🎤 Opening Mic ID: {ASRConfig.MIC_DEVICE_ID}...")

        try:
            with sd.InputStream(samplerate=ASRConfig.SAMPLE_RATE, channels=1, dtype='float32', device=ASRConfig.MIC_DEVICE_ID) as stream:
                self.log_message.emit("🎤 Recording started... Speak now!")
                while self.is_recording:
                    data, overflowed = stream.read(1024)
                    if overflowed: print("⚠️ Audio Buffer Overflow")
                    self.audio_buffer.append(data)
        except Exception as e:
            self.log_message.emit(f"❌ Microphone Error: {str(e)}")
            self.is_recording = False
            return

        if self.audio_buffer:
            full_audio = np.concatenate(self.audio_buffer, axis=0).flatten()
            self.recording_finished.emit(full_audio)

    def stop(self):
        self.is_recording = False

class AudioPlayer(QThread):
    playback_finished = pyqtSignal()
    log_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.audio_data = None

    def set_audio(self, audio_data):
        self.audio_data = audio_data

    def run(self):
        if self.audio_data is None: return
        try:
            sd.play(self.audio_data, ASRConfig.SAMPLE_RATE)
            sd.wait()
            self.playback_finished.emit()
        except Exception as e:
            self.log_message.emit(f"❌ Playback Error: {e}")
            self.playback_finished.emit()

# ==========================================
# OPENAI SPECIFIC ENGINE
# ==========================================

class OpenAIModelLoader(QThread):
    finished_loading = pyqtSignal(object, object) # Returns (model, processor)
    log_message = pyqtSignal(str)

    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id

    def run(self):
        self.log_message.emit(f"🟦 [OpenAI Engine] Initializing: {self.model_id}")
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, 
                torch_dtype=ASRConfig.TORCH_DTYPE, 
                low_cpu_mem_usage=True, 
                use_safetensors=True
            ).to(ASRConfig.DEVICE)

            processor = AutoProcessor.from_pretrained(self.model_id)
            
            self.log_message.emit(f"✅ OpenAI Model Loaded: {self.model_id}")
            self.finished_loading.emit(model, processor)
        except Exception as e:
            self.log_message.emit(f"❌ OpenAI Load Error: {e}")
            self.finished_loading.emit(None, None)

class OpenAITranscriber(QThread):
    transcription_finished = pyqtSignal(str)
    
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.audio_data = None

    def set_audio(self, audio_array):
        self.audio_data = audio_array

    def run(self):
        if not self.model or self.audio_data is None: return
        try:
            start_time = time.time()
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

            input_features = self.processor(
                self.audio_data, sampling_rate=ASRConfig.SAMPLE_RATE, return_tensors="pt"
            ).input_features.to(ASRConfig.DEVICE, dtype=ASRConfig.TORCH_DTYPE)

            predicted_ids = self.model.generate(input_features, language="en") 
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            result_text = transcription[0].strip()
            
            duration = time.time() - start_time
            peak_mb = 0
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            stats_msg = f"⏱ Time: {duration:.2f}s | 💾 Peak VRAM: {peak_mb:.2f} MB"
            self.transcription_finished.emit(f"{result_text}\n\n[OpenAI Stats]\n{stats_msg}")
            
        except Exception as e:
            self.transcription_finished.emit(f"[OpenAI Error: {str(e)}]")