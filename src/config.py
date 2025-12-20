import torch

class ASRConfig:
    """
    Global Configuration for FlowAudio.
    Centralizes all constants and hardware settings.
    """
    # --- Audio Settings ---
    SAMPLE_RATE = 16000
    CHANNELS = 1
    
    # --- Hardware Settings ---
    # Default Microphone ID (Will be updated by UI selection)
    MIC_DEVICE_ID = 1  
    
    # --- Compute Settings ---
    # Auto-detect CUDA (NVIDIA GPU) or fallback to CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use FP16 for GPU to save memory, FP32 for CPU
    TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    @staticmethod
    def get_device_name():
        """Returns the name of the active computing device."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"