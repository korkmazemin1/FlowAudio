import torch
import sys

def check_cuda_status():
    print("="*40)
    print("🚀 PyTorch CUDA Test Script")
    print("="*40)

    # 1. Version Information
    print(f"Python Version  : {sys.version.split()[0]}")
    print(f"PyTorch Version : {torch.__version__}")
    
    # 2. Check CUDA Availability
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print(f"\n✅ CUDA Status    : ACTIVE (AVAILABLE)")
        print(f"CUDA Version    : {torch.version.cuda}")
        print(f"GPU Count       : {torch.cuda.device_count()}")
        
        current_device = torch.cuda.current_device()
        print(f"Current GPU     : {torch.cuda.get_device_name(current_device)}")
        
        # 3. Performance / Operation Test
        print("\n🔄 Running GPU tensor operation test...")
        try:
            # Create random tensors on GPU
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            
            # Matrix multiplication (stresses the GPU)
            z = torch.matmul(x, y)
            
            print("✅ Operation Successful! Tensor device:", z.device)
            print("🎉 Setup is working perfectly!")
            
        except Exception as e:
            print(f"❌ GPU detected but an error occurred during operation:\n{e}")
            
    else:
        print("\n❌ CUDA Status    : DISABLED (NOT AVAILABLE)")
        print("⚠️ System is currently using CPU only.")
        print("Possible causes: NVIDIA driver not installed or PyTorch CPU version installed.")

    print("="*40)

if __name__ == "__main__":
    check_cuda_status()