import torch
import time
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

def print_gpu_utilization(stage_name):
    """
    Prints the current GPU memory usage stats.
    """
    if not torch.cuda.is_available():
        print(f"[{stage_name}] CUDA is not available. Using CPU.")
        return

    # Force garbage collection to get accurate reading
    torch.cuda.empty_cache()
    
    # Get memory stats
    allocated = torch.cuda.memory_allocated() / 1024**3  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # Convert bytes to GB
    
    print(f"--- [GPU LOG: {stage_name}] ---")
    print(f"Allocated Memory : {allocated:.3f} GB (Tensors used by model/data)")
    print(f"Reserved Memory  : {reserved:.3f} GB (Total cached by PyTorch)")
    print("-" * 40)

def main():
    print("🚀 Starting Whisper Model GPU Test...")
    
    # 1. Setup Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-tiny" # Using 'tiny' for quick testing. Change to 'small' or 'medium' if needed.

    print(f"✅ Device: {device.upper()}")
    print_gpu_utilization("Initial State")

    # ---------------------------------------------------------
    # STEP 1: Load Model & Processor
    # ---------------------------------------------------------
    start_load = time.time()
    
    print(f"\n📥 Loading model: {model_id}...")
    
    # Load model to GPU directly
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    
    end_load = time.time()
    load_duration = end_load - start_load
    
    print(f"✅ Model loaded in {load_duration:.2f} seconds.")
    print_gpu_utilization("After Model Load")

    # ---------------------------------------------------------
    # STEP 2: Generate Dummy Audio Input
    # ---------------------------------------------------------
    # Whisper expects 16000Hz audio. Let's create 5 seconds of random noise.
    # Shape: (batch_size, sequence_length) -> (1, 16000 * 5)
    print("\n🔊 Generating dummy audio data (5 seconds of noise)...")
    dummy_audio = torch.randn(16000 * 5) # Move to CPU first
    
    # Preprocess the audio
    input_features = processor(
        dummy_audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features
    
    input_features = input_features.to(device, dtype=torch_dtype)

    # ---------------------------------------------------------
    # STEP 3: Run Inference
    # ---------------------------------------------------------
    print("\n⚡ Running Inference (Transcribing noise)...")
    
    # Reset peak memory stats to track specifically the inference phase
    torch.cuda.reset_peak_memory_stats()
    start_inf = time.time()
    
    # Generate tokens
    predicted_ids = model.generate(input_features, language="en")
    
    # Decode tokens to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    end_inf = time.time()
    inf_duration = end_inf - start_inf
    
    # ---------------------------------------------------------
    # STEP 4: Final Report
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("📊 PERFORMANCE REPORT")
    print("="*40)
    print(f"Model           : {model_id}")
    print(f"Load Time       : {load_duration:.4f} sec")
    print(f"Inference Time  : {inf_duration:.4f} sec")
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU Memory : {peak_mem:.3f} GB")
    
    print("="*40)
    print(f"Output Text (Gibberish due to noise): '{transcription[0]}'")

if __name__ == "__main__":
    main()