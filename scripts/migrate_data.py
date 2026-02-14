import os
import socket
import time
import shutil
import argparse
from app.config import DATA_PATH, ACTIONS, INGESTED_SUFFIX

def migrate_folder(root_dir, extension, mode):
    """Consolidate and process files based on the selected mode.
    
    Modes:
    - 'rename': Rename non-compliant files to hostname_timestamp scheme.
    - 'done': Append _done suffix to all usable files.
    - 'reset': Remove _done suffix from all files.
    """
    if not os.path.exists(root_dir):
        return

    hostname = socket.gethostname().replace(" ", "_")
    
    for action in os.listdir(root_dir):
        action_path = os.path.join(root_dir, action)
        if not os.path.isdir(action_path) or action not in ACTIONS:
            continue
            
        print(f"Processing {root_dir}/{action} in mode: {mode}")
        
        # 1. Always consolidate from 'processed' folder first
        processed_dir = os.path.join(action_path, "processed")
        if os.path.exists(processed_dir):
            files = [f for f in os.listdir(processed_dir) if f.endswith(extension)]
            for f in files:
                src = os.path.join(processed_dir, f)
                dest = os.path.join(action_path, f)
                if os.path.exists(dest):
                    name, ext = os.path.splitext(f)
                    dest = os.path.join(action_path, f"{name}_{int(time.time())}{ext}")
                shutil.move(src, dest)
                print(f"  Consolidated: {f}")
            try:
                os.rmdir(processed_dir)
            except OSError:
                pass

        # 2. Perform mode-specific operations
        files = [f for f in os.listdir(action_path) if f.endswith(extension)]
        
        for i, f in enumerate(files):
            old_path = os.path.join(action_path, f)
            name, ext = os.path.splitext(f)
            
            if mode == "rename":
                # Rename if it doesn't match hostname_timestamp scheme
                # We check for hostname in the filename
                if not f.startswith(f"{hostname}_"):
                    new_name = f"{hostname}_{int(time.time())}_{i}{ext}"
                    new_path = os.path.join(action_path, new_name)
                    os.rename(old_path, new_path)
                    print(f"  Renamed: {f} -> {new_name}")

            elif mode == "done":
                # Append _done if not present
                if INGESTED_SUFFIX not in f:
                    new_name = f"{name}{INGESTED_SUFFIX}{ext}"
                    new_path = os.path.join(action_path, new_name)
                    # Check for collisions
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"  Marked Done: {f} -> {new_name}")

            elif mode == "reset":
                # Remove _done if present
                if INGESTED_SUFFIX in f:
                    new_name = f.replace(INGESTED_SUFFIX, "")
                    new_path = os.path.join(action_path, new_name)
                    # Use timestamp if collision
                    if os.path.exists(new_path):
                        new_name = f.replace(INGESTED_SUFFIX, f"_{int(time.time())}")
                        new_path = os.path.join(action_path, new_name)
                    os.rename(old_path, new_path)
                    print(f"  Reset: {f} -> {new_name}")

def main():
    parser = argparse.ArgumentParser(description="SIBI Data Migration Tool")
    parser.add_argument("mode", choices=["rename", "done", "reset"], 
                        help="rename: Fix names | done: Usable -> _done | reset: _done -> Usable")
    args = parser.parse_args()

    print(f"--- SIBI Data Migration: {args.mode.upper()} ---")
    
    migrate_folder("raw_videos", ".avi", args.mode)
    migrate_folder("raw_glove", ".csv", args.mode)
    
    print("\nOperation complete!")

if __name__ == "__main__":
    main()
