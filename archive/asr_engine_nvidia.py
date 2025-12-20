import time
import os
import torch
import soundfile as sf
import logging
import traceback
from PyQt6.QtCore import QThread, pyqtSignal

# Import Config from the main engine file
from asr_engine_openai import ASRConfig

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
LOG_FILE = "nvidia_debug.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG, # Debug level to see everything
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

logging.info("=== NVIDIA ENGINE INITIALIZATION STARTED ===")

# ==========================================
# SAFE IMPORT BLOCK
# ==========================================
NEMO_AVAILABLE = False
IMPORT_ERROR_MSG = ""

try:
    logging.info("Attempting to import NeMo models...")
    # Canary uses EncDecMultiTaskModel
    from nemo.collections.asr.models import EncDecMultiTaskModel
    logging.info("SUCCESS: NeMo imported successfully.")
    NEMO_AVAILABLE = True
except Exception as e:
    full_error = traceback.format_exc()
    logging.error("FAILURE: NeMo Import Failed.")
    logging.error(f"Traceback:\n{full_error}")
    NEMO_AVAILABLE = False
    IMPORT_ERROR_MSG = str(e)

# ==========================================
# WORKER THREADS
# ==========================================

class NvidiaModelLoader(QThread):
    finished_loading = pyqtSignal(object, object)
    log_message = pyqtSignal(str)

    def __init__(self, model_id):
        super().__init__()
        self.model_id = "nvidia/canary-1b" 

    def run(self):
        self.log_message.emit(f"🟩 [NVIDIA NeMo] Initializing Canary-1B...")
        
        if not NEMO_AVAILABLE:
            self.log_message.emit("❌ CRITICAL: NeMo Toolkit failed to load.")
            self.log_message.emit(f"🔍 Error: {IMPORT_ERROR_MSG}")
            self.finished_loading.emit(None, None)
            return

        try:
            self.log_message.emit("⚙️ Loading Canary Model (This implies downloading ~4GB if not cached)...")
            
            # Load Model
            start_load = time.time()
            canary_model = EncDecMultiTaskModel.from_pretrained(self.model_id)
            end_load = time.time()
            logging.info(f"Model loaded in {end_load - start_load:.2f} seconds.")
            
            # Optimize Decoding
            logging.info("Optimizing decoding strategy (Beam Size=1)...")
            decode_cfg = canary_model.cfg.decoding
            decode_cfg.beam.beam_size = 1
            canary_model.change_decoding_strategy(decode_cfg)

            # Move to GPU
            if torch.cuda.is_available():
                logging.info(f"Moving model to GPU: {torch.cuda.get_device_name(0)}")
                canary_model = canary_model.cuda()
            
            canary_model.eval()
            
            self.log_message.emit(f"✅ NVIDIA Canary-1B Ready!")
            self.finished_loading.emit(canary_model, None) 
            
        except Exception as e:
            logging.error(f"Loader Error: {e}")
            self.log_message.emit(f"❌ Load Error. See '{LOG_FILE}'")
            self.finished_loading.emit(None, None)

class NvidiaTranscriber(QThread):
    transcription_finished = pyqtSignal(str)
    
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor 
        self.audio_data = None
        self.temp_filename = "temp_nvidia_input.wav"

    def set_audio(self, audio_array):
        self.audio_data = audio_array

    def run(self):
        if not self.model or self.audio_data is None: return

        logging.info("--- Starting Transcription ---")
        try:
            start_time = time.time()
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

            # 1. Save RAM audio to file
            logging.debug(f"Saved temp audio: {self.temp_filename}")
            sf.write(self.temp_filename, self.audio_data, ASRConfig.SAMPLE_RATE)

            # 2. Transcribe
            logging.info("Calling model.transcribe()...")
            
            # --- CRITICAL FIX IS HERE ---
            # Old versions used 'paths2audio_files', new versions use 'audio'
            # We explicitly use 'audio' to fix the TypeError.
            predicted_text_list = self.model.transcribe(
                audio=[self.temp_filename], # <--- NAME CHANGED HERE
                batch_size=1
            )
            
            # 3. Extract Text
            if isinstance(predicted_text_list, list) and len(predicted_text_list) > 0:
                first_result = predicted_text_list[0]
                # Canary usually returns plain text string in list, but just in case:
                if hasattr(first_result, 'text'):
                    result_text = first_result.text
                else:
                    result_text = str(first_result)
            else:
                result_text = "[No Output]"
                logging.warning(f"Unexpected output: {predicted_text_list}")

            # 4. Cleanup & Stats
            if os.path.exists(self.temp_filename):
                os.remove(self.temp_filename)

            duration = time.time() - start_time
            peak_mb = 0
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            logging.info(f"Transcription finished. Result: {result_text[:30]}...")
            
            stats_msg = f"⏱ Time: {duration:.2f}s | 💾 Peak VRAM: {peak_mb:.2f} MB"
            self.transcription_finished.emit(f"{result_text}\n\n[NVIDIA Canary Stats]\n{stats_msg}")
            
        except Exception as e:
            full_trace = traceback.format_exc()
            logging.error(f"Inference Internal Error: {e}")
            logging.error(f"Traceback: {full_trace}")
            self.transcription_finished.emit(f"[Transcription Failed. See Log File]")