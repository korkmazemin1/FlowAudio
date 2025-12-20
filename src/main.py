import sys
import gc # Garbage Collector for memory cleanup
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QTextEdit, QLabel, QPushButton, QHBoxLayout, QFrame, 
                             QGraphicsDropShadowEffect, QComboBox, QProgressBar)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor

# IMPORT SHARED UTILS
from asr_engine_openai import AudioRecorder, AudioPlayer, ASRConfig, SystemMonitor

# ==========================================
# CUSTOM WIDGET: LOADING OVERLAY
# ==========================================
class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.hide()
        self.setStyleSheet("background-color: rgba(0, 0, 0, 200);")
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        container = QFrame()
        container.setStyleSheet("""
            QFrame { background-color: #1E1E1E; border: 2px solid #00FFCC; border-radius: 15px; padding: 20px; min-width: 300px; }
        """)
        container_layout = QVBoxLayout(container)
        
        self.lbl_text = QLabel("Initializing Engine...\nPlease Wait.")
        self.lbl_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_text.setStyleSheet("color: white; font-size: 18px; font-weight: bold; background: transparent;")
        container_layout.addWidget(self.lbl_text)
        
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar { border: none; background-color: #333; height: 10px; border-radius: 5px; margin-top: 15px; }
            QProgressBar::chunk { background-color: #00FFCC; border-radius: 5px; }
        """)
        container_layout.addWidget(self.progress)
        layout.addWidget(container)

    def show_loading(self, msg):
        self.lbl_text.setText(f"🚀 {msg}")
        self.resize(self.parent().size())
        self.show()
        self.raise_()

    def hide_loading(self):
        self.hide()

# ==========================================
# MAIN WINDOW CLASS
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # --- State ---
        self.model = None
        self.processor = None
        self.current_audio = None 
        self.is_recording = False
        self.active_engine_type = None 
        
        # --- UI Setup ---
        self.setWindowTitle(f"FlowAudio Multi-Engine - Mic ID: {ASRConfig.MIC_DEVICE_ID}")
        self.resize(1000, 750)
        self.showMaximized() 
        self.apply_styles()
        self.init_ui()

        self.loading_overlay = LoadingOverlay(self)

        # --- Timer ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats_display)
        self.timer.start(1000)

        # --- Shared Workers ---
        self.recorder = AudioRecorder()
        self.recorder.recording_finished.connect(self.on_recording_finished)
        self.recorder.log_message.connect(self.update_status)
        
        self.player = AudioPlayer()
        self.player.playback_finished.connect(self.on_playback_finished)
        self.player.log_message.connect(self.update_status)

        self.transcriber = None
        self.loader = None

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget { color: #E0E0E0; font-family: 'Segoe UI', Roboto, Helvetica, sans-serif; }
            QComboBox { background-color: #2D2D2D; color: #FFFFFF; border: 1px solid #444; padding: 8px; border-radius: 5px; font-size: 14px; }
            QTextEdit { background-color: #1E1E1E; color: #00FFCC; font-size: 16px; border: 1px solid #333; border-radius: 10px; padding: 15px; }
            QPushButton { background-color: #2D2D2D; color: white; border-radius: 8px; padding: 12px 20px; font-size: 14px; font-weight: bold; border: 1px solid #444; }
            QPushButton:hover { background-color: #3D3D3D; border: 1px solid #666; }
            QPushButton:disabled { background-color: #1a1a1a; color: #444; border: 1px solid #222; }
            QPushButton#BtnLoad { background-color: #0055AA; border-bottom: 3px solid #0088FF; }
            QPushButton#BtnRecord { border-bottom: 3px solid #FF4444; }
            QPushButton#BtnASR { background-color: #004444; border-bottom: 3px solid #00FFFF; color: #E0FFFF; }
        """)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(15)

        # Header
        title = QLabel("FLOW AUDIO INTELLIGENCE (MULTI-ENGINE)")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFFFFF;")
        main_layout.addWidget(title)

        # Model Selection
        model_frame = QFrame()
        model_frame.setStyleSheet("background-color: #1E1E1E; border-radius: 10px;")
        model_layout = QHBoxLayout(model_frame)
        
        lbl_model = QLabel("Select Engine/Model:")
        self.combo_models = QComboBox()
        self.combo_models.addItems([
            "nvidia/canary-1b", # Canary first!
            "openai/whisper-large-v3",
            "openai/whisper-tiny"
        ])
        self.combo_models.setMinimumWidth(300)
        
        self.btn_load_model = QPushButton("⬇ Initialize Engine")
        self.btn_load_model.setObjectName("BtnLoad")
        self.btn_load_model.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_model.clicked.connect(self.start_engine_loading)

        model_layout.addWidget(lbl_model)
        model_layout.addWidget(self.combo_models)
        model_layout.addWidget(self.btn_load_model)
        main_layout.addWidget(model_frame)

        # Output
        self.text_area = QTextEdit()
        self.text_area.setPlaceholderText("Select a model and click Initialize Engine to start.")
        self.text_area.setReadOnly(True)
        main_layout.addWidget(self.text_area)

        # Status
        self.status_label = QLabel("System Idle.")
        self.status_label.setStyleSheet("color: #888; font-size: 14px; background: #181818; padding: 8px; border-radius: 5px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        # Controls
        btn_frame = QFrame()
        btn_frame.setStyleSheet("background-color: #181818; border-radius: 15px;")
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(20, 20, 20, 20)
        
        self.btn_record = QPushButton("⏺ START RECORDING")
        self.btn_record.setObjectName("BtnRecord")
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self.toggle_recording)
        
        self.btn_play = QPushButton("▶ PLAYBACK")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.play_audio)
        
        self.btn_asr = QPushButton("⚡ TRANSCRIBE")
        self.btn_asr.setObjectName("BtnASR")
        self.btn_asr.setEnabled(False)
        self.btn_asr.clicked.connect(self.start_transcription)

        btn_layout.addWidget(self.btn_record)
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_asr)
        main_layout.addWidget(btn_frame)

    def resizeEvent(self, event):
        if hasattr(self, 'loading_overlay') and self.loading_overlay:
            self.loading_overlay.resize(self.size())
        super().resizeEvent(event)

    # --- LOGIC: MEMORY MANAGEMENT (GPU CLEANER) ---
    
    def unload_current_engine(self):
        """
        Forces PyTorch to release VRAM.
        Essential when switching between huge models like Canary and Whisper.
        """
        if self.model or self.processor:
            self.update_status("🧹 Cleaning up GPU memory...")
            
            # 1. Delete Python references
            if self.model: del self.model
            if self.processor: del self.processor
            if self.transcriber: del self.transcriber
            
            self.model = None
            self.processor = None
            self.transcriber = None
            
            # 2. Force Python Garbage Collection
            gc.collect()
            
            # 3. Force PyTorch to release cached VRAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect() # Extra cleanup
            
            self.update_status("🧹 GPU Memory Cleaned.")

    # --- LOGIC: DYNAMIC ENGINE SWITCHING ---

    def start_engine_loading(self):
        selected_model = self.combo_models.currentText()
        
        # STEP 1: CLEANUP OLD MODEL
        self.loading_overlay.show_loading(f"Unloading old engine...")
        QApplication.processEvents() # Update UI
        self.unload_current_engine()

        # STEP 2: LOAD NEW MODEL
        self.update_status(f"Requesting Engine for: {selected_model}")
        self.loading_overlay.show_loading(f"Loading {selected_model}...")
        
        self.btn_load_model.setEnabled(False)
        self.combo_models.setEnabled(False)
        self.set_buttons_enabled(False)

        # Dynamic Import & Instantiation
        if "nvidia" in selected_model.lower():
            # Use NVIDIA NeMo Engine
            from archive.asr_engine_nvidia import NvidiaModelLoader
            self.active_engine_type = "nvidia"
            self.loader = NvidiaModelLoader(selected_model)
        else:
            # Use OpenAI Engine (Default for others)
            from asr_engine_openai import OpenAIModelLoader
            self.active_engine_type = "openai"
            self.loader = OpenAIModelLoader(selected_model)

        self.loader.log_message.connect(self.update_status)
        self.loader.finished_loading.connect(self.on_model_loaded)
        self.loader.start()

    def on_model_loaded(self, model, processor):
        self.loading_overlay.hide_loading()
        self.btn_load_model.setEnabled(True)
        self.combo_models.setEnabled(True)

        if model is None:
            self.text_area.append("❌ Fatal Error: Engine failed to initialize. Check logs.")
            return

        self.model = model
        self.processor = processor
        
        # Initialize the correct Transcriber based on active engine
        if self.active_engine_type == "nvidia":
            from archive.asr_engine_nvidia import NvidiaTranscriber
            self.transcriber = NvidiaTranscriber(model, processor)
        else:
            from asr_engine_openai import OpenAITranscriber
            self.transcriber = OpenAITranscriber(model, processor)

        self.transcriber.transcription_finished.connect(self.on_transcription_result)
        
        self.btn_record.setEnabled(True)
        self.text_area.append(f"✅ Engine Ready: {self.active_engine_type.upper()} Mode Active.")
        self.update_status("Ready.")

    # --- STANDARD OPERATIONS ---

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.btn_record.setText("⏹ STOP RECORDING")
            self.btn_record.setStyleSheet("background-color: #880000; color: white; border: 2px solid #FF0000;")
            self.set_buttons_enabled(False)
            self.text_area.clear()
            self.recorder.start()
        else:
            self.is_recording = False
            self.btn_record.setText("⏺ START RECORDING")
            self.btn_record.setStyleSheet("") 
            self.recorder.stop()

    def on_recording_finished(self, audio_array):
        self.current_audio = audio_array
        duration = len(audio_array) / ASRConfig.SAMPLE_RATE
        self.text_area.append(f"💾 Audio Captured. Duration: {duration:.2f}s")
        self.set_buttons_enabled(True)

    def play_audio(self):
        if self.current_audio is not None:
            self.set_buttons_enabled(False)
            self.player.set_audio(self.current_audio)
            self.player.start()

    def on_playback_finished(self):
        self.set_buttons_enabled(True)

    def start_transcription(self):
        if self.current_audio is not None:
            self.set_buttons_enabled(False)
            self.transcriber.set_audio(self.current_audio)
            self.transcriber.start()

    def on_transcription_result(self, text):
        self.text_area.append("\n📝 RESULT:")
        self.text_area.append("---------")
        self.text_area.append(text)
        self.text_area.append("---------")
        self.set_buttons_enabled(True)

    def set_buttons_enabled(self, enabled):
        self.btn_play.setEnabled(enabled)
        self.btn_asr.setEnabled(enabled)
        self.btn_record.setEnabled(enabled if not self.is_recording else True)

    def update_status(self, msg):
        if "🚀" in msg or "✅" in msg or "❌" in msg or "🧹" in msg:
            self.text_area.append(f"<span style='color:#aaa;'>{msg}</span>")
        
        current = self.status_label.text().split("|")[-1].strip() if "|" in self.status_label.text() else ""
        self.status_label.setText(f"{msg} | {current}")

    def update_stats_display(self):
        stats = SystemMonitor.get_stats()
        current_msg = self.status_label.text().split("|")[0].strip()
        self.status_label.setText(f"{current_msg} | {stats}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())