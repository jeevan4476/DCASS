import torch
import sys

def check_gpu():
    print("="*60)
    print("DCASS HARDWARE SANITY CHECK")
    print("="*60)
    
    # 1. Software Versions
    print(f"Python Version:   {sys.version.split()[0]}")
    print(f"PyTorch Version:  {torch.__version__}")
    
    # 2. Check CUDA Availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available:   {'YES' if cuda_available else 'NO'}")
    
    if cuda_available:
        # 3. GPU Details
        print(f"CUDA Version:     {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"GPU Count:        {device_count}")
        
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"Active GPU:       {gpu_name}")
        
        # 4. Functional Test (The "Real" Test)
        print("\nRunning Tensor Test...", end=" ")
        try:
            # Create a random tensor and move it to GPU
            x = torch.rand(1000, 1000).to("cuda")
            y = torch.rand(1000, 1000).to("cuda")
            
            # Perform matrix multiplication (heavy op)
            z = torch.matmul(x, y)
            
            # Move back to CPU to ensure full round-trip works
            z_cpu = z.to("cpu")
            
            print("PASSED ✅")
            print("  -> Tensor allocation, computation, and transfer are working.")
            
        except Exception as e:
            print("FAILED ❌")
            print(f"\nError Details: {e}")
            
    else:
        print("\n⚠️  WARNING: System is running in CPU-ONLY mode.")
        print("If you have an NVIDIA GPU, you may have installed the wrong PyTorch version.")
        print("Try reinstalling with: pip install torch --index-url https://download.pytorch.org/whl/cu118")

    print("="*60)

if __name__ == "__main__":
    check_gpu()