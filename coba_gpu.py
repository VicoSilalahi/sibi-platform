import os
import sys
import glob

# 1. Cari lokasi library nvidia yang terinstall via pip
#    (Biasanya ada di dalam site-packages/nvidia/...)
env_path = sys.prefix
site_packages = glob.glob(os.path.join(env_path, "lib", "python*", "site-packages"))[0]
nvidia_path = os.path.join(site_packages, "nvidia")

# 2. Kumpulkan semua folder library penting (cudnn, cublas, dll)
libs = [
    os.path.join(nvidia_path, "cudnn", "lib"),
    os.path.join(nvidia_path, "cublas", "lib"),
    os.path.join(nvidia_path, "cuda_runtime", "lib"),
    os.path.join(nvidia_path, "cufft", "lib"),
    os.path.join(nvidia_path, "curand", "lib"),
    os.path.join(nvidia_path, "cusolver", "lib"),
    os.path.join(nvidia_path, "cusparse", "lib"),
    os.path.join(nvidia_path, "nccl", "lib"),
    # Tambahkan path driver sistem (hanya sebagai cadangan terakhir)
    "/usr/lib" 
]

# 3. Rakit path baru dan paksa masuk ke LD_LIBRARY_PATH
#    Kita taruh path nvidia DI DEPAN agar dibaca duluan
new_ld_path = ":".join(libs)
os.environ["LD_LIBRARY_PATH"] = new_ld_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

print(f"✅ Path NVIDIA berhasil disuntikkan: {len(libs)} folder terdeteksi.")

# --- BARU IMPORT TENSORFLOW SETELAH INI ---
import tensorflow as tf

# Cek GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n🎉 SUKSES BESAR! GPU: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("\n❌ Masih gagal. Cek output terminal untuk nama file yang hilang.")