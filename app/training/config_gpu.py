import os
import sys
import glob
from pathlib import Path

def configure_gpu():
    print("🔧 SEDANG MENGKONFIGURASI GPU & XLA...")
    
    # 1. Cari path environment secara cross-platform
    import sysconfig
    site_packages = sysconfig.get_path('purelib')
    
    # 2. Cari file 'libdevice.10.bc'
    possible_paths = [
        os.path.join(site_packages, "nvidia", "cuda_nvcc", "nvvm", "libdevice", "libdevice.10.bc"),
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cuda_nvcc", "nvvm", "libdevice", "libdevice.10.bc"), # Windows fallback
        "/usr/lib/cuda/nvvm/libdevice/libdevice.10.bc", # Arch Linux default
        "/opt/cuda/nvvm/libdevice/libdevice.10.bc"
    ]
    
    found_path = None
    for p in possible_paths:
        if os.path.exists(p):
            found_path = p
            break
            
    if not found_path:
        print("❌ CRITICAL: 'libdevice.10.bc' TIDAK DITEMUKAN!")
        print("   Solusi: pip install nvidia-cuda-nvcc-cu12")
        return False

    print(f"✅ Libdevice ditemukan di: {found_path}")

    # 3. Set Environment Variable 'XLA_FLAGS'
    cuda_dir = str(Path(found_path).parents[2])
    os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={cuda_dir}"
    print(f"✅ XLA_FLAGS diset ke: {cuda_dir}")

    # 4. Inject library paths (LD_LIBRARY_PATH for Linux, PATH for Windows)
    nvidia_base = os.path.join(site_packages, "nvidia")
    libs_paths = glob.glob(os.path.join(nvidia_base, "*", "lib"))
    
    if os.name == 'nt':  # Windows
        # On Windows, we add to PATH
        current_path = os.environ.get("PATH", "")
        new_path = ";".join(libs_paths) + ";" + current_path
        os.environ["PATH"] = new_path
        print("✅ PATH updated with NVIDIA libraries.")
    else:  # Linux
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = ":".join(libs_paths) + ":" + current_ld
        os.environ["LD_LIBRARY_PATH"] = new_ld
        print("✅ LD_LIBRARY_PATH updated with NVIDIA libraries.")
    
    return True

if __name__ == "__main__":
    configure_gpu()