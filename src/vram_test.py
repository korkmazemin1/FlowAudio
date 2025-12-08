import torch
import time
import psutil
import os
import gc
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init(autoreset=True)

def print_info(message):
    print(f"{Fore.CYAN}[INFO] {Style.RESET_ALL}{message}")

def print_success(message):
    print(f"{Fore.GREEN}[SUCCESS] {Style.RESET_ALL}{message}")

def print_warning(message):
    print(f"{Fore.YELLOW}[WARNING] {Style.RESET_ALL}{message}")

def check_system_status():
    """
    Analyzes and prints the current System RAM and GPU VRAM status.
    Returns True if GPU is available, False otherwise.
    """
    print("\n" + "="*50)
    print(f"{Fore.MAGENTA}SYSTEM AND GPU ANALYSIS (RTX 3050 Ti Setup){Style.RESET_ALL}")
    print("="*50)
    
    # RAM Check
    ram = psutil.virtual_memory()
    print(f"System RAM: {ram.total / (1024**3):.2f} GB (Used: {ram.percent}%)")

    # GPU Check
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPUs Found: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            # VRAM Info
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            print(f"  - Total VRAM: {total_mem / (1024**3):.2f} GB")
            print(f"  - Free VRAM: {free_mem / (1024**3):.2f} GB")
            print(f"  - CUDA Version: {torch.version.cuda}")
        return True
    else:
        print_warning("No CUDA-supported GPU found! Operations will run on CPU (Slow).")
        return False

def download_and_test_whisper(model_id):
    """
    Downloads ONLY Whisper models, loads into VRAM, and measures performance.
    """
    print(f"\n---> {Fore.YELLOW}Testing Whisper Model: {model_id}{Style.RESET_ALL}")
    
    start_time = time.time()
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use float16 for GPU to save memory (Crucial for 4GB VRAM)
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Measure initial VRAM
        start_vram = 0
        if device == "cuda":
            torch.cuda.empty_cache()
            start_vram = torch.cuda.memory_allocated()

        print_info("Downloading and loading model weights...")

        # --- WHISPER SPECIFIC LOADING ---
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        
        model.to(device)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate used VRAM
        used_vram = 0
        if device == "cuda":
            end_vram = torch.cuda.memory_allocated()
            used_vram = (end_vram - start_vram) / (1024**2) # Convert to MB

        print_success(f"Model Loaded Successfully!")
        print(f"  - Time Elapsed: {elapsed_time:.2f} seconds")
        if device == "cuda":
            print(f"  - VRAM Usage: {used_vram:.2f} MB")
        
        # Cleanup to free memory for the next test
        del model
        del processor
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
        return {"status": "Success", "time": elapsed_time, "vram": used_vram}

    except Exception as e:
        print_warning(f"Error occurred: {e}")
        # If OOM happens, we catch it here
        if "out of memory" in str(e).lower():
             print_warning("This model is too large for your 4GB VRAM.")
        return {"status": "Failed", "time": 0, "vram": 0}

def main():
    has_gpu = check_system_status()
    
    # OPTIMIZED LIST FOR 4GB VRAM (RTX 3050 Ti)
    models_to_test = [
        # 1. Tiny: Super fast, low accuracy
        {"id": "openai/whisper-tiny", "note": "Lightning fast"},
        
        # 2. Base: Good standard
        {"id": "openai/whisper-base", "note": "Balanced Speed"},
        
        # 3. Small: The Sweet Spot for 4GB VRAM
        {"id": "openai/whisper-small", "note": "High Quality / Recommended"},
        
        # 4. Medium: The Stress Test (May work or fail depending on background apps)
        {"id": "openai/whisper-medium", "note": "High Quality (VRAM Heavy)"},
    ]

    print(f"\n{Fore.BLUE}STARTING WHISPER BENCHMARK PROCESS...{Style.RESET_ALL}")
    print("Testing models specifically selected for your hardware.\n")

    results = []

    for m in models_to_test:
        res = download_and_test_whisper(m["id"])
        results.append({
            "model": m["id"],
            "note": m["note"],
            **res
        })
    
    # Summary Report
    print("\n" + "="*80)
    print(f"{Fore.MAGENTA}WHISPER BENCHMARK SUMMARY (4GB VRAM){Style.RESET_ALL}")
    print("="*80)
    print(f"{'Model Name':<25} | {'Status':<10} | {'VRAM (MB)':<10} | {'Note'}")
    print("-" * 80)
    for r in results:
        status_color = Fore.GREEN if r['status'] == "Success" else Fore.RED
        print(f"{r['model']:<25} | {status_color}{r['status']:<10}{Style.RESET_ALL} | {r['vram']:<10.2f} | {r['note']}")
    
    print("\nNext Step: We will build the 'Live Audio Transcriber' using the best performing model (likely 'Small').")

if __name__ == "__main__":
    main()