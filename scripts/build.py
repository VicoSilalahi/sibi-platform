import os
import sys
import subprocess
import shutil

def run_command(command):
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        # sys.exit(result.returncode)
    return result.returncode

def build_windows(mode="standalone"):
    print(f"--- Building for Windows ({mode}) ---")
    
    # Base command for Nuitka
    # --standalone: Create a folder with all dependencies
    # --onefile: Create a single .exe
    # --windows-disable-console: Hide console window
    
    mode_flag = "--onefile" if mode == "onefile" else "--standalone"
    
    command = (
        f"nuitka {mode_flag} --windows-disable-console "
        "--enable-plugin=tk-inter "
        "--include-data-dir=piper=piper "
        "--include-data-dir=datasets=datasets "
        "--include-data-dir=app/models/saved_models=app/models/saved_models "
        "--include-data-files=actions.json=actions.json "
        f"--output-dir=dist/windows_{mode} "
        "app/gui.py"
    )
    
    run_command(command)

def build_linux(mode="standalone"):
    print(f"--- Building for Linux ({mode}) ---")
    
    mode_flag = "--onefile" if mode == "onefile" else "--standalone"
    
    command = (
        f"nuitka {mode_flag} "
        "--enable-plugin=tk-inter "
        "--include-data-dir=piper=piper "
        "--include-data-dir=datasets=datasets "
        "--include-data-dir=app/models/saved_models=app/models/saved_models "
        "--include-data-files=actions.json=actions.json "
        f"--output-dir=dist/linux_{mode} "
        "app/gui.py"
    )
    
    run_command(command)

def main():
    # Ensure dist directory exists
    os.makedirs("dist", exist_ok=True)
    
    target = sys.platform
    mode = "standalone"
    
    if len(sys.argv) > 1:
        target = sys.argv[1]
    
    if "--onefile" in sys.argv:
        mode = "onefile"
    
    if target == "win32" or target == "windows":
        build_windows(mode)
    elif target in ["linux", "linux2"]:
        build_linux(mode)
    else:
        print(f"Unsupported platform for build: {target}")

if __name__ == "__main__":
    main()
