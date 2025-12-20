import sys
import gc
import torch
import sounddevice as sd # Required for listing microphones
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QTextEdit, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QProgressBar, QGroupBox)
from PyQt6.QtCore import Qt, QTimer

# --- MODULAR IMPORTS ---
from config import ASRConfig
from utils.monitor import SystemMonitor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.loader = None
        self.live_transcriber = None
        
        self.init_ui()
        self.populate_microphones() # Scan mics on startup
        
        # Hardware Stats Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(1000)

    def init_ui(self):
        self.setWindowTitle("FlowAudio Modular AI System")
        self.resize(1000, 800)
        self.setStyleSheet("background-color: #121212; color: #E0E0E0; font-family: 'Segoe UI';")
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # ==========================================
        # SECTION 1: HARDWARE & SETTINGS
        # ==========================================
        hw_group = QGroupBox("Device Settings")
        hw_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        hw_layout = QHBoxLayout(hw_group)

        # -- Engine Selection --
        self.combo_engine = QComboBox()
        self.combo_engine.addItems(["faster/large-v3", "faster/medium"])
        self.combo_engine.setStyleSheet("background: #333; color: white; padding: 5px; border: 1px solid #555;")
        
        # -- Microphone Selection --
        self.combo_mic = QComboBox()
        self.combo_mic.setStyleSheet("background: #333; color: white; padding: 5px; border: 1px solid #555; min-width: 250px;")
        self.combo_mic.currentIndexChanged.connect(self.on_mic_changed) # Event Listener

        # -- Load Button --
        btn_load = QPushButton("⬇ LOAD ENGINE")
        btn_load.setStyleSheet("background: #0055AA; color: white; padding: 8px 15px; font-weight: bold; border-radius: 5px;")
        btn_load.clicked.connect(self.load_engine)

        hw_layout.addWidget(QLabel("🧠 AI Model:"))
        hw_layout.addWidget(self.combo_engine)
        hw_layout.addWidget(QLabel("🎤 Microphone:"))
        hw_layout.addWidget(self.combo_mic)
        hw_layout.addWidget(btn_load)
        
        layout.addWidget(hw_group)

        # ==========================================
        # SECTION 2: OUTPUT AREA
        # ==========================================
        self.text_area = QTextEdit()
        self.text_area.setPlaceholderText("Select your Microphone, Load the Engine, and Start Live Session...")
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet("background: #1E1E1E; color: #00FFCC; font-size: 16px; border: 1px solid #333; border-radius: 5px;")
        layout.addWidget(self.text_area)

        # ==========================================
        # SECTION 3: LIVE CONTROL & VU METER
        # ==========================================
        control_group = QGroupBox("Live Control")
        control_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        control_layout = QVBoxLayout(control_group)
        
        # -- VU Meter Bar --
        self.vu_meter = QProgressBar()
        self.vu_meter.setRange(0, 100)
        self.vu_meter.setValue(0)
        self.vu_meter.setTextVisible(False)
        self.vu_meter.setFixedHeight(12)
        self.vu_meter.setStyleSheet("""
            QProgressBar { border: 1px solid #444; border-radius: 6px; background-color: #222; }
            QProgressBar::chunk { background-color: #00FF00; border-radius: 6px; }
        """)
        
        vu_label_layout = QHBoxLayout()
        vu_label_layout.addWidget(QLabel("Volume Level:"))
        vu_label_layout.addWidget(self.vu_meter)
        control_layout.addLayout(vu_label_layout)
        
        # -- Start/Stop Button --
        self.btn_live = QPushButton("🎙️ START LIVE SESSION")
        self.btn_live.setEnabled(False)
        self.btn_live.setCheckable(True)
        self.btn_live.setStyleSheet("""
            QPushButton { background-color: #222; color: #888; padding: 15px; font-size: 16px; font-weight: bold; border-radius: 8px; border: 1px solid #444; }
            QPushButton:enabled { color: white; border-color: #00FFCC; }
            QPushButton:checked { background-color: #880000; border-color: #FF0000; color: white; }
        """)
        self.btn_live.clicked.connect(self.toggle_live)
        control_layout.addWidget(self.btn_live)
        
        layout.addWidget(control_group)

        # ==========================================
        # SECTION 4: STATUS BAR
        # ==========================================
        self.status_bar = QLabel("System Idle.")
        self.status_bar.setStyleSheet("color: #888; font-size: 12px;")
        self.status_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_bar)

    # --- MICROPHONE LOGIC ---
    def populate_microphones(self):
        """Scans system audio devices and populates the combo box."""
        self.combo_mic.clear()
        devices = sd.query_devices()
        
        input_devices = []
        for i, dev in enumerate(devices):
            # Filter only input devices (max_input_channels > 0)
            if dev['max_input_channels'] > 0:
                name = f"{i}: {dev['name']}"
                self.combo_mic.addItem(name, userData=i) # Store ID as UserData
                input_devices.append(i)

        # Try to select the default one from Config
        index = self.combo_mic.findData(ASRConfig.MIC_DEVICE_ID)
        if index >= 0:
            self.combo_mic.setCurrentIndex(index)
        
        self.log_to_ui(f"🔍 Found {len(input_devices)} microphones.")

    def on_mic_changed(self, index):
        """Updates the Global Config when user selects a mic."""
        if index < 0: return
        
        mic_id = self.combo_mic.currentData()
        mic_name = self.combo_mic.currentText()
        
        # Update Global Config
        ASRConfig.MIC_DEVICE_ID = mic_id
        
        self.log_to_ui(f"🎤 Microphone Changed to: {mic_name} (ID: {mic_id})")
        
        # If live session is running, warn user to restart
        if self.btn_live.isChecked():
            self.update_status("⚠️ Restart Live Session to apply Mic change!")

    # --- ENGINE LOGIC ---
    def load_engine(self):
        model_name = self.combo_engine.currentText()
        self.log_to_ui(f"🛠️ Loading Engine: {model_name}...")
        
        if self.model: 
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

        from engines.fast_engine import FastModelLoader
        
        self.loader = FastModelLoader(model_name)
        self.loader.log_message.connect(self.update_status)
        self.loader.finished_loading.connect(self.on_engine_ready)
        self.loader.start()

    def on_engine_ready(self, model, _):
        if model:
            self.model = model
            self.btn_live.setEnabled(True)
            self.update_status("✅ Engine Loaded Successfully.")
        else:
            self.update_status("❌ Engine Failed to Load.")

    def toggle_live(self, active):
        if active:
            self.btn_live.setText("⏹ STOP LIVE SESSION")
            self.text_area.clear()
            self.log_to_ui(f"🎙️ Starting Stream on Mic ID: {ASRConfig.MIC_DEVICE_ID}")
            
            from engines.fast_engine import RealTimeTranscriber
            self.live_transcriber = RealTimeTranscriber(self.model)
            
            self.live_transcriber.partial_transcript.connect(self.append_text)
            self.live_transcriber.log_message.connect(self.update_status)
            self.live_transcriber.volume_level.connect(self.update_vu_meter)
            
            self.live_transcriber.start()
        else:
            self.btn_live.setText("🎙️ START LIVE SESSION")
            self.vu_meter.setValue(0)
            if self.live_transcriber:
                self.live_transcriber.stop()
                self.live_transcriber.wait()

    def update_vu_meter(self, level):
        self.vu_meter.setValue(level)

    def append_text(self, text):
        self.text_area.append(f"🗣️ {text}")
        sb = self.text_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def log_to_ui(self, msg):
        self.text_area.append(f"<span style='color: gray;'>{msg}</span>")

    def update_status(self, msg):
        clean_msg = msg.split("|")[0]
        # Keep stats if available
        current_text = self.status_bar.text()
        if "|" in current_text:
            stats = current_text.split("|")[-1].strip()
            self.status_bar.setText(f"{clean_msg} | {stats}")
        else:
            self.status_bar.setText(clean_msg)

    def update_stats(self):
        stats = SystemMonitor.get_stats()
        current_msg = self.status_bar.text().split("|")[0].strip()
        self.status_bar.setText(f"{current_msg} | {stats}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())