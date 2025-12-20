import sounddevice as sd
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from config import ASRConfig

class AudioRecorder(QThread):
    """
    Handles audio recording from the microphone in a background thread.
    Returns raw numpy arrays.
    """
    recording_finished = pyqtSignal(object) # Emits numpy array
    log_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.frames = []

    def run(self):
        self.is_recording = True
        self.frames = []
        self.log_message.emit("🎙️ Recording Started...")
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Audio Input Status: {status}")
            self.frames.append(indata.copy())

        try:
            with sd.InputStream(samplerate=ASRConfig.SAMPLE_RATE, 
                                channels=ASRConfig.CHANNELS, 
                                callback=callback,
                                device=ASRConfig.MIC_DEVICE_ID):
                while self.is_recording:
                    sd.sleep(100) # Keep thread alive
        except Exception as e:
            self.log_message.emit(f"❌ Microphone Error: {e}")

    def stop(self):
        self.is_recording = False
        self.wait() # Wait for thread to exit safely
        
        if self.frames:
            # Flatten list of chunks into a single 1D array
            audio_data = np.concatenate(self.frames, axis=0).flatten()
            duration = len(audio_data) / ASRConfig.SAMPLE_RATE
            self.log_message.emit(f"⏹ Recording Stopped ({duration:.2f}s)")
            self.recording_finished.emit(audio_data)
        else:
            self.log_message.emit("⚠️ No audio recorded.")