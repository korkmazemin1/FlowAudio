import sys
import re  
import time
import logging # Added logging
from datetime import datetime
import sounddevice as sd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QTextEdit, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QProgressBar, QGroupBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QTextCursor

from config import ASRConfig
from utils.monitor import SystemMonitor

# --- CENTRAL LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("flow_audio_debug.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UI")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pipeline = None 
        
        logger.info("🖥️ Application Starting...")
        
        self.init_ui()
        self.populate_microphones()
        
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(1000)

    def init_ui(self):
        self.setWindowTitle("FlowAudio - LOGGING ENABLED (Medium)")
        self.resize(1000, 850)
        self.setStyleSheet("background-color: #121212; color: #E0E0E0; font-family: 'Segoe UI';")
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)

        # 1. Controls
        hw_group = QGroupBox("Kontrol Merkezi")
        hw_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; font-weight: bold; }")
        hw_layout = QHBoxLayout(hw_group)

        self.combo_mic = QComboBox()
        self.combo_mic.setStyleSheet("background: #333; color: white; padding: 10px; min-width: 300px; font-size: 14px;")
        self.combo_mic.currentIndexChanged.connect(self.on_mic_changed)

        self.btn_live = QPushButton("⚡ BAŞLAT (LOGGING ON)")
        self.btn_live.setCheckable(True)
        self.btn_live.setMinimumHeight(45)
        self.btn_live.setStyleSheet("""
            QPushButton { background: #222; color: #aaa; font-weight: bold; font-size: 14px; border: 1px solid #444; border-radius: 5px; }
            QPushButton:checked { background: #006600; color: white; border-color: #00FF00; }
            QPushButton:hover { border-color: #666; }
        """)
        self.btn_live.clicked.connect(self.toggle_live)

        hw_layout.addWidget(QLabel("🎤 Kaynak:"))
        hw_layout.addWidget(self.combo_mic)
        hw_layout.addWidget(self.btn_live)
        layout.addWidget(hw_group)

        # 2. Output
        self.text_area = QTextEdit()
        self.text_area.setPlaceholderText("Sistem Hazır.\n\nLOG DOSYASI: flow_audio_debug.log\nModel: Medium\nÇeviri: GPU Offline")
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet("""
            QTextEdit {
                background: #1E1E1E; color: #00FFCC; font-size: 24px; 
                border: 1px solid #333; line-height: 1.6; padding: 20px; font-weight: 500;
            }
        """)
        layout.addWidget(self.text_area)

        # 3. Telemetry Footer
        footer_layout = QHBoxLayout()
        
        self.vu_meter = QProgressBar()
        self.vu_meter.setRange(0, 100)
        self.vu_meter.setTextVisible(False)
        self.vu_meter.setFixedWidth(100)
        self.vu_meter.setStyleSheet("QProgressBar { border: 1px solid #444; background: #222; height: 8px;} QProgressBar::chunk { background: #00FF00; }")
        
        self.lbl_latency = QLabel("Gecikme: 0.0s")
        self.lbl_latency.setStyleSheet("color: #FF5555; font-weight: bold; margin-left: 10px;")
        
        self.lbl_times = QLabel("In: --:--:-- | Out: --:--:--")
        self.lbl_times.setStyleSheet("color: #888; margin-left: 10px; font-size: 11px;")

        self.status_bar = QLabel("Sistem Boşta.")
        self.status_bar.setStyleSheet("color: #888; margin-left: 20px;")
        
        footer_layout.addWidget(QLabel("Vol:"))
        footer_layout.addWidget(self.vu_meter)
        footer_layout.addWidget(self.lbl_latency)
        footer_layout.addWidget(self.lbl_times)
        footer_layout.addStretch()
        footer_layout.addWidget(self.status_bar)
        layout.addLayout(footer_layout)

    # --- TELEMETRY ---
    def update_telemetry(self, capture_ts, finish_ts, latency, avg_latency):
        t_in = datetime.fromtimestamp(capture_ts).strftime('%H:%M:%S.%f')[:-3]
        t_out = datetime.fromtimestamp(finish_ts).strftime('%H:%M:%S.%f')[:-3]
        
        self.lbl_latency.setText(f"Gecikme: {latency:.2f}s (Ort: {avg_latency:.2f}s)")
        self.lbl_times.setText(f"In: {t_in} | Out: {t_out}")
        
        if latency > 2.0: 
            self.lbl_latency.setStyleSheet("color: red; font-weight: bold; margin-left: 10px;")
        else:
            self.lbl_latency.setStyleSheet("color: #00FF00; font-weight: bold; margin-left: 10px;")

    # --- TEXT CLEANER ---
    def append_text_instantly(self, text):
        if not text: return
        
        clean_text = text.replace("...", " ").strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)

        current_text = self.text_area.toPlainText().strip()
        if current_text:
            last_word = current_text.split()[-1]
            first_word_new = clean_text.split()[0]
            
            last_word_pure = re.sub(r'[^\w\s]', '', last_word).lower()
            first_word_pure = re.sub(r'[^\w\s]', '', first_word_new).lower()

            if last_word_pure == first_word_pure:
                parts = clean_text.split()[1:]
                clean_text = " ".join(parts)

        if not clean_text: return

        cursor = self.text_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        if self.text_area.toPlainText() and not self.text_area.toPlainText().endswith(" ") and not clean_text.startswith(" "):
            cursor.insertText(" ")
            
        cursor.insertText(clean_text)
        self.text_area.setTextCursor(cursor)
        sb = self.text_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    # --- SYSTEM LOGIC ---
    def populate_microphones(self):
        self.combo_mic.clear()
        devices = sd.query_devices()
        logger.info(f"Found {len(devices)} Audio Devices.")
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                self.combo_mic.addItem(f"{i}: {dev['name']}", userData=i)
        
        idx = self.combo_mic.findData(ASRConfig.MIC_DEVICE_ID)
        if idx >= 0: self.combo_mic.setCurrentIndex(idx)

    def on_mic_changed(self, index):
        if index >= 0:
            new_mic = self.combo_mic.currentText()
            ASRConfig.MIC_DEVICE_ID = self.combo_mic.currentData()
            logger.info(f"🎤 Microphone Changed to: {new_mic}")

    def toggle_live(self, active):
        if active:
            logger.info("▶️ User pressed START")
            if self.pipeline: 
                self.pipeline.stop()
                self.pipeline.wait()
            
            self.btn_live.setText("⏹ DURDUR")
            self.text_area.clear()
            
            from engines.fast_engine import MultiThreadedTranscriber
            
            self.pipeline = MultiThreadedTranscriber() 
            
            self.pipeline.partial_transcript.connect(self.append_text_instantly)
            self.pipeline.latency_data.connect(self.update_telemetry)
            self.pipeline.volume_level.connect(self.vu_meter.setValue)
            self.pipeline.log_message.connect(self.update_status)
            
            self.pipeline.start()
        else:
            logger.info("⏹️ User pressed STOP")
            self.btn_live.setText("⚡ BAŞLAT (LOGGING ON)")
            if self.pipeline:
                self.pipeline.stop()
                self.pipeline.wait()
                self.update_status("Sistem Durdu.")

    def update_status(self, msg):
        self.status_bar.setText(msg.split("|")[0])

    def update_stats(self):
        stats = SystemMonitor.get_stats()
        curr = self.status_bar.text().split("|")[0]
        self.status_bar.setText(f"{curr} | {stats}")
    
    def closeEvent(self, event):
        logger.info("❌ Application Closing...")
        if self.pipeline:
            self.pipeline.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())