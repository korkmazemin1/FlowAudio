import os
import shutil
from pathlib import Path

def clean_nemo_models():
    """
    Scans standard cache directories for NVIDIA NeMo models (specifically Canary)
    and offers to delete them to free up disk space.
    """
    print("🧹 Starting NVIDIA NeMo Cache Cleaner...")
    
    # Get the current user's home directory
    user_home = Path.home()
    
    # Define potential paths where NeMo/Torch might store downloaded models
    # NeMo usually stores weights in .cache/torch/NeMo
    possible_paths = [
        user_home / ".cache" / "torch" / "NeMo",
        user_home / ".cache" / "huggingface" / "hub"
    ]
    
    found_any = False

    # Iterate through possible cache locations
    for cache_path in possible_paths:
        if cache_path.exists():
            print(f"\n📂 Scanning directory: {cache_path}")
            
            # Check subdirectories
            for item in cache_path.iterdir():
                # Look for folders related to NeMo, Canary, or Nvidia
                if item.is_dir() and ("NeMo" in item.name or "canary" in item.name.lower() or "nvidia" in item.name.lower()):
                    
                    # Calculate directory size in GB
                    size_bytes = sum(f.stat().st_size for f in item.glob('**/*') if f.is_file())
                    size_gb = size_bytes / (1024**3)
                    
                    print(f"   👉 Found: {item.name} (Size: ~{size_gb:.2f} GB)")
                    
                    # Request user confirmation before deletion
                    confirm = input(f"   ⚠️ Do you want to DELETE '{item.name}'? (y/n): ").lower()
                    
                    if confirm == 'y':
                        try:
                            print(f"   🗑️ Deleting: {item.name}...")
                            shutil.rmtree(item)
                            print("   ✅ Deleted successfully.")
                            found_any = True
                        except Exception as e:
                            print(f"   ❌ Error occurred: {e}")
                    else:
                        print("   ⏭️ Skipped.")
        else:
            # Path does not exist, likely no cache there
            pass

    # Final summary
    if not found_any:
        print("\n✨ No NVIDIA NeMo cache files were deleted.")
        print("Your disk seems clean or you chose to skip.")
    else:
        print("\n✨ Cleanup process completed. Disk space reclaimed.")

if __name__ == "__main__":
    clean_nemo_models()
    input("\nPress Enter to exit...")