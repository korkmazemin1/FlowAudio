import psutil

# pynvml is optional, used for NVIDIA GPU stats
try:
    import pynvml
    HAS_NVIDIA = True
except ImportError:
    HAS_NVIDIA = False

class SystemMonitor:
    """
    Utility class to fetch real-time system statistics (CPU, RAM, GPU).
    """
    @staticmethod
    def get_stats():
        # 1. CPU & RAM Usage
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        
        # 2. GPU Usage (if NVIDIA is present)
        gpu_util = 0
        vram_alloc = 0
        
        if HAS_NVIDIA:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # GPU Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                
                # VRAM Memory Usage
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_alloc = mem_info.used / (1024**3) # Convert Bytes to GB
            except Exception:
                pass # Fail silently if GPU is busy or driver issue

        return f"CPU: {cpu}% | RAM: {ram}% | GPU: {gpu_util}% | VRAM: {vram_alloc:.2f} GB"