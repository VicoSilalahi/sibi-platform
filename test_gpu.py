import os
import glob
import sys

env_path = sys.prefix
site_packages = glob.glob(os.path.join(env_path, "lib", "python*", "site-packages"))[0]
nvidia_base = os.path.join(site_packages, "nvidia")

# Daftar file WAJIB untuk TensorFlow 2.15
required_libs = {
    "cudnn": "libcudnn.so.8",        # SUDAH OK
    "cublas": "libcublas.so.12",     # Cek ini
    "cublas": "libcublasLt.so.12",   # Cek ini juga
    "cufft": "libcufft.so.10",       # Cek ini
    "curand": "libcurand.so.10",     # Cek ini
    "cusolver": "libcusolver.so.11", # !!! Sering bermasalah !!!
    "cusparse": "libcusparse.so.12"  # Cek ini
}

print(f"🔍 Memeriksa kelengkapan library di: {nvidia_base}")
print("-" * 50)

all_good = True
for folder, filename in required_libs.items():
    # Cari di subfolder (misal: nvidia/cublas/lib/libcublas.so.12)
    search_path = os.path.join(nvidia_base, "*", "lib", filename + "*")
    found = glob.glob(search_path)
    
    if found:
        print(f"✅ {filename} \t-> Ditemukan!")
    else:
        print(f"❌ {filename} \t-> HILANG/SALAH VERSI")
        all_good = False

print("-" * 50)
if not all_good:
    print("SOLUSI: Anda perlu membuat symlink manual atau install ulang paket spesifik.")
else:
    print("Semua file lengkap. Masalahnya ada di LD_LIBRARY_PATH.")