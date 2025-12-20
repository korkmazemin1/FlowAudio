import sounddevice as sd
from PyQt6.QtCore import QThread, pyqtSignal
from config import ASRConfig

class AudioPlayer(QThread):
    """
    Handles audio playback in a background thread to prevent UI freezing.
    """
    playback_finished = pyqtSignal()
    log_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.audio_data = None

    def set_audio(self, audio_array):
        self.audio_data = audio_array

    def run(self):
        if self.audio_data is None: return
        
        self.log_message.emit("▶ Playing Audio...")
        try:
            sd.play(self.audio_data, ASRConfig.SAMPLE_RATE)
            sd.wait() # Block thread until playback finishes
            self.log_message.emit("⏹ Playback Finished.")
        except Exception as e:
            self.log_message.emit(f"❌ Playback Error: {e}")
        
        self.playback_finished.emit()