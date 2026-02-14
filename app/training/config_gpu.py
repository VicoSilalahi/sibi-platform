import os
import sys
import glob
from pathlib import Path

def configure_gpu():
    print("🔧 SEDANG MENGKONFIGURASI GPU & XLA...")
    
    # 1. Cari path environment
    env_path = sys.prefix
    site_packages = glob.glob(os.path.join(env_path, "lib", "python*", "site-packages"))[0]
    
    # 2. Cari file 'libdevice.10.bc' (Kunci utamanya!)
    # Biasanya ada di: site-packages/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
    # Atau di sistem: /usr/lib/cuda/nvvm/libdevice/libdevice.10.bc
    
    possible_paths = [
        os.path.join(site_packages, "nvidia", "cuda_nvcc", "nvvm", "libdevice", "libdevice.10.bc"),
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
    # Kita harus menunjuk ke folder INDUK dari folder 'nvvm'
    # Jika file di: .../nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
    # Maka Cuda Dir adalah: .../nvidia/cuda_nvcc/
    
    cuda_dir = str(Path(found_path).parents[2]) # Naik 2 level dari folder libdevice
    
    # Set flag XLA agar TF tahu di mana cuda berada
    os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={cuda_dir}"
    print(f"✅ XLA_FLAGS diset ke: {cuda_dir}")

    # 4. Inject LD_LIBRARY_PATH (Agar library runtime terbaca)
    # Ini script yang kita pakai sebelumnya, kita gabung di sini biar praktis.
    nvidia_base = os.path.join(site_packages, "nvidia")
    libs_paths = glob.glob(os.path.join(nvidia_base, "*", "lib"))
    
    # Gabungkan dengan path yang sudah ada
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld = ":".join(libs_paths) + ":" + current_ld
    os.environ["LD_LIBRARY_PATH"] = new_ld
    
    return True

if __name__ == "__main__":
    configure_gpu()