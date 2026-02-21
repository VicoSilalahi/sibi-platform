import os
import glob
import sys
from pathlib import Path

def create_symlink(folder_name, target_link_name):
    # 1. Cari lokasi folder library di dalam environment
    env_path = sys.prefix
    site_packages = Path(glob.glob(os.path.join(env_path, "lib", "python*", "site-packages"))[0])
    lib_dir = site_packages / "nvidia" / folder_name / "lib"

    print(f"📂 Memeriksa folder: {folder_name}...")
    
    if not lib_dir.exists():
        print(f"   ❌ Folder {lib_dir} tidak ditemukan! (Apakah sudah pip install nvidia-{folder_name}-cu12?)")
        return

    # 2. Cari file asli (.so) yang ada di situ
    # Kita cari file apa saja yang berakhiran angka (versi asli)
    candidates = list(lib_dir.glob("*.so.*"))
    
    if not candidates:
        print(f"   ⚠️  Folder kosong/tidak ada file .so")
        return

    # Ambil file dengan nama terpanjang sebagai sumber (biasanya itu file aslinya)
    # Contoh: libcufft.so.10.9.0.58
    source_file = max(candidates, key=lambda p: len(str(p)))
    
    # Tentukan nama link yang mau dibuat
    link_path = lib_dir / target_link_name

    # 3. Buat Link
    if link_path.exists():
        print(f"   👌 Link {target_link_name} sudah ada.")
    else:
        try:
            os.symlink(source_file.name, link_path)
            print(f"   ✅ SUKSES: Membuat link {target_link_name} -> {source_file.name}")
        except Exception as e:
            print(f"   ❌ Gagal membuat link: {e}")

# --- EKSEKUSI PERBAIKAN ---
print("🛠️ MEMULAI PERBAIKAN LIBRARY...\n")

# 1. Perbaiki libcufft (Biasanya TF cari so.10 atau so.11)
# Kita buatkan keduanya biar aman
create_symlink("cufft", "libcufft.so.10") 
create_symlink("cufft", "libcufft.so.11")

# 2. Perbaiki libcurand (TF cari so.10)
create_symlink("curand", "libcurand.so.10")

# 3. Perbaiki libcusolver (TF cari so.11)
create_symlink("cusolver", "libcusolver.so.11")

# 4. Perbaiki libcusparse (TF cari so.12)
create_symlink("cusparse", "libcusparse.so.12")

print("\n🚀 Selesai! Silakan coba jalankan 'coba_gpu.py' lagi.")