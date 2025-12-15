import sys
import time
import numpy as np
import torch
import sounddevice as sd
import psutil
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QTextEdit, QLabel, QPushButton, QHBoxLayout)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "openai/whisper-tiny"
SAMPLE_RATE = 16000
# IMPORTANT: Keep using ID 2 as it worked for you
MIC_DEVICE_ID = 2 
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ==========================================
# WORKER THREADS
# ==========================================

class ModelLoader(QThread):
    """
    Background thread to load the Whisper model without freezing the GUI.
    """
    finished_loading = pyqtSignal(object, object)
    log_message = pyqtSignal(str)

    def run(self):
        self.log_message.emit(f"🚀 Loading Model: {MODEL_ID} on {DEVICE.upper()}...")
        try:
            # Load Model
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                MODEL_ID, 
                torch_dtype=TORCH_DTYPE, 
                low_cpu_mem_usage=True, 
                use_safetensors=True
            ).to(DEVICE)

            # Load Processor
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            
            self.log_message.emit("✅ Model Loaded Successfully!")
            self.finished_loading.emit(model, processor)
        except Exception as e:
            self.log_message.emit(f"❌ Error loading model: {e}")

class AudioRecorder(QThread):
    """
    Background thread to capture audio from the specific microphone (ID 2).
    Stores audio in memory (RAM) only. No file saving.
    """
    # Emits: (audio_array) - No filename anymore
    recording_finished = pyqtSignal(np.ndarray) 
    log_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.audio_buffer = []

    def run(self):
        self.is_recording = True
        self.audio_buffer = [] 
        
        self.log_message.emit(f"🎤 Opening Mic ID: {MIC_DEVICE_ID}...")

        try:
            # Open the stream with the specific device ID
            with sd.InputStream(samplerate=SAMPLE_RATE, 
                                channels=1, 
                                dtype='float32', 
                                device=MIC_DEVICE_ID) as stream:
                
                self.log_message.emit("🎤 Recording started... Speak now!")
                
                while self.is_recording:
                    # Read audio chunks
                    data, overflowed = stream.read(1024)
                    if overflowed:
                        print("⚠️ Audio Buffer Overflow")
                    self.audio_buffer.append(data)
        
        except Exception as e:
            self.log_message.emit(f"❌ Microphone Error: {str(e)}")
            self.is_recording = False
            return

        # Process audio when stopped
        if self.audio_buffer:
            # Flatten to 1D array
            full_audio = np.concatenate(self.audio_buffer, axis=0).flatten()
            
            # REMOVED: File saving logic (sf.write)
            
            # Emit the audio data directly from RAM
            self.recording_finished.emit(full_audio)

    def stop(self):
        self.is_recording = False

class AudioPlayer(QThread):
    """
    Background thread to play back the recorded audio from memory.
    """
    playback_finished = pyqtSignal()
    log_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.audio_data = None

    def set_audio(self, audio_data):
        self.audio_data = audio_data

    def run(self):
        if self.audio_data is None:
            return
        
        try:
            # Play audio using sounddevice
            sd.play(self.audio_data, SAMPLE_RATE)
            sd.wait() # Block thread until audio finishes
            self.playback_finished.emit()
        except Exception as e:
            self.log_message.emit(f"❌ Playback Error: {e}")
            self.playback_finished.emit()

class Transcriber(QThread):
    """
    Background thread to run the Whisper inference on GPU/CPU.
    """
    transcription_finished = pyqtSignal(str)
    
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.audio_data = None

    def set_audio(self, audio_array):
        self.audio_data = audio_array

    def run(self):
        if self.model is None or self.audio_data is None:
            return

        try:
            # 1. Preprocess features (CPU)
            input_features = self.processor(
                self.audio_data, 
                sampling_rate=SAMPLE_RATE, 
                return_tensors="pt"
            ).input_features

            # 2. Move to GPU
            input_features = input_features.to(DEVICE, dtype=TORCH_DTYPE)

            # 3. Generate tokens (Inference)
            # You can change language="tr" here for Turkish
            predicted_ids = self.model.generate(input_features, language="en") 

            # 4. Decode to text
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            result_text = transcription[0].strip()
            
            self.transcription_finished.emit(result_text)
            
        except Exception as e:
            self.transcription_finished.emit(f"[Error: {str(e)}]")

# ==========================================
# MAIN GUI WINDOW
# ==========================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.model = None
        self.processor = None
        self.current_audio = None 
        
        # Window Config
        self.setWindowTitle(f"FlowAudio - Mic ID: {MIC_DEVICE_ID}")
        self.setGeometry(100, 100, 650, 500)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        # Stylesheet (Dark Theme)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QTextEdit { 
                background-color: #2d2d2d; color: #ffffff; font-size: 16px; 
                border: 1px solid #3e3e3e; padding: 10px; border-radius: 5px;
            }
            QLabel { color: #aaaaaa; font-size: 12px; }
            QPushButton {
                background-color: #333; color: white; border-radius: 5px;
                padding: 10px; font-weight: bold; border: 1px solid #444;
            }
            QPushButton:disabled { background-color: #222; color: #555; }
            QPushButton:hover { background-color: #444; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 1. Output Text Area
        self.text_area = QTextEdit()
        self.text_area.setPlaceholderText("Logs and Transcriptions will appear here...")
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

        # 2. Status Bar
        self.status_label = QLabel("Initializing...")
        layout.addWidget(self.status_label)

        # 3. Control Buttons Layout
        btn_layout = QHBoxLayout()

        # Button: Record
        self.btn_record = QPushButton("⏺ Record")
        self.btn_record.setStyleSheet("color: #ff5555;") # Red text
        self.btn_record.setEnabled(False) 
        self.btn_record.clicked.connect(self.toggle_recording)
        btn_layout.addWidget(self.btn_record)

        # Button: Play (Listen)
        self.btn_play = QPushButton("▶ Playback")
        self.btn_play.setStyleSheet("color: #55ff55;") # Green text
        self.btn_play.setEnabled(False) 
        self.btn_play.clicked.connect(self.play_audio)
        btn_layout.addWidget(self.btn_play)

        # Button: Transcribe (ASR)
        self.btn_asr = QPushButton("⚡ Transcribe")
        self.btn_asr.setStyleSheet("color: #55ffff;") # Cyan text
        self.btn_asr.setEnabled(False) 
        self.btn_asr.clicked.connect(self.start_transcription)
        btn_layout.addWidget(self.btn_asr)

        layout.addLayout(btn_layout)

        # 4. Workers Initialization
        self.is_recording = False
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(1000)

        # Load Model
        self.loader = ModelLoader()
        self.loader.log_message.connect(self.update_status)
        self.loader.finished_loading.connect(self.on_model_loaded)
        self.loader.start()

        # Recorder & Player setup
        self.recorder = AudioRecorder()
        self.recorder.recording_finished.connect(self.on_recording_finished)
        self.recorder.log_message.connect(self.update_status)
        
        self.player = AudioPlayer()
        self.player.playback_finished.connect(self.on_playback_finished)
        self.player.log_message.connect(self.update_status)

        self.transcriber = None

    def on_model_loaded(self, model, processor):
        self.model = model
        self.processor = processor
        self.transcriber = Transcriber(model, processor)
        self.transcriber.transcription_finished.connect(self.on_transcription_result)
        
        self.btn_record.setEnabled(True)
        self.update_status("System Ready.")

    def toggle_recording(self):
        if not self.is_recording:
            # START RECORDING
            self.is_recording = True
            self.btn_record.setText("⏹ Stop")
            self.btn_record.setStyleSheet("background-color: #cc0000; color: white;")
            
            self.btn_play.setEnabled(False)
            self.btn_asr.setEnabled(False)
            self.text_area.clear()
            self.update_status("Starting Recorder...")
            
            self.recorder.start()
        else:
            # STOP RECORDING
            self.is_recording = False
            self.btn_record.setText("⏺ Record")
            self.btn_record.setStyleSheet("color: #ff5555;")
            self.update_status("Finalizing audio...")
            self.recorder.stop()

    def on_recording_finished(self, audio_array):
        """Called when audio is captured (Stored in RAM)."""
        self.current_audio = audio_array
        
        duration = len(audio_array) / SAMPLE_RATE
        
        # Log to Text Area
        self.text_area.append(f"💾 Captured to RAM. Duration: {duration:.2f}s")
        
        self.update_status(f"Captured {duration:.1f}s. Ready.")
        
        # Enable next steps
        self.btn_play.setEnabled(True)
        self.btn_asr.setEnabled(True)

    def play_audio(self):
        """Plays the currently stored audio from RAM."""
        if self.current_audio is not None:
            self.update_status("🔊 Playing from memory...")
            self.btn_play.setEnabled(False) 
            self.btn_record.setEnabled(False)
            self.btn_asr.setEnabled(False)
            
            self.player.set_audio(self.current_audio)
            self.player.start()

    def on_playback_finished(self):
        self.update_status("Playback finished.")
        self.btn_play.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.btn_asr.setEnabled(True)

    def start_transcription(self):
        if self.current_audio is not None:
            self.update_status("🧠 Transcribing...")
            self.btn_asr.setEnabled(False)
            self.btn_record.setEnabled(False)
            self.btn_play.setEnabled(False)
            
            self.transcriber.set_audio(self.current_audio)
            self.transcriber.start()

    def on_transcription_result(self, text):
        self.text_area.append("\n📝 Transcription:")
        self.text_area.append("-------------------")
        self.text_area.append(text)
        self.text_area.append("-------------------")
        self.update_status("✅ Transcription Complete.")
        
        self.btn_asr.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.btn_play.setEnabled(True)

    def update_status(self, msg):
        current_stats = ""
        if "|" in self.status_label.text():
            current_stats = self.status_label.text().split("|")[-1]
            self.status_label.setText(f"{msg} | {current_stats}")
        else:
            self.status_label.setText(msg)

    def update_stats(self):
        ram = psutil.virtual_memory().percent
        msg = f"RAM: {ram}%"
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**3
            msg += f" | VRAM: {vram:.2f} GB"
        
        current_msg = self.status_label.text().split("|")[0].strip()
        self.status_label.setText(f"{current_msg} | {msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())