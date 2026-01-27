import os
import socket
import time
import shutil
import argparse
from app.config import DATA_PATH, ACTIONS

def migrate_folder(root_dir, extension, undo=False, mark_done=False):
    """Rename, consolidate, or reset files in a raw data directory."""
    if not os.path.exists(root_dir):
        return

    hostname = socket.gethostname().replace(" ", "_")
    from app.config import INGESTED_SUFFIX
    
    for action in os.listdir(root_dir):
        action_path = os.path.join(root_dir, action)
        if not os.path.isdir(action_path) or action not in ACTIONS:
            continue
            
        print(f"Processing {root_dir}/{action}...")
        
        # --- UNDO MODE: Remove _done suffix ---
        if undo:
            files = [f for f in os.listdir(action_path) if f.endswith(extension)]
            for f in files:
                if INGESTED_SUFFIX in f:
                    old_path = os.path.join(action_path, f)
                    new_name = f.replace(INGESTED_SUFFIX, "")
                    new_path = os.path.join(action_path, new_name)
                    
                    if os.path.exists(new_path):
                        new_name = f.replace(INGESTED_SUFFIX, f"_{int(time.time())}")
                        new_path = os.path.join(action_path, new_name)
                    
                    os.rename(old_path, new_path)
                    print(f"  Reset: {f} -> {new_name}")
            continue

        # --- MARK DONE MODE: Add _done suffix to all ---
        if mark_done:
            files = [f for f in os.listdir(action_path) if f.endswith(extension)]
            for i, f in enumerate(files):
                if INGESTED_SUFFIX not in f:
                    old_path = os.path.join(action_path, f)
                    name, ext = os.path.splitext(f)
                    
                    # Also ensure hostname/timestamp if it was a raw file
                    if hostname not in name:
                        name = f"{hostname}_{int(time.time()) + i}"
                    
                    new_filename = f"{name}{INGESTED_SUFFIX}{ext}"
                    new_path = os.path.join(action_path, new_filename)
                    
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"  Marked Done: {f} -> {new_filename}")
            continue

        # --- NORMAL MODE: Consolidate and Rename ---
        # 1. Handle processed subfolder (These will get _done suffix)
        processed_dir = os.path.join(action_path, "processed")
        if os.path.exists(processed_dir):
            files = [f for f in os.listdir(processed_dir) if f.endswith(extension)]
            for i, f in enumerate(files):
                src = os.path.join(processed_dir, f)
                new_name = f"{hostname}_{int(time.time()) + i}{INGESTED_SUFFIX}{extension}"
                dest = os.path.join(action_path, new_name)
                shutil.move(src, dest)
                print(f"  Consolidated & Renamed: {f} -> {new_name}")
            
            try:
                os.rmdir(processed_dir)
            except OSError:
                pass

        # 2. Rename remaining raw files to new scheme
        files = [f for f in os.listdir(action_path) if f.endswith(extension)]
        for i, f in enumerate(files):
            if hostname in f or INGESTED_SUFFIX in f:
                continue
                
            old_path = os.path.join(action_path, f)
            new_name = f"{hostname}_{int(time.time()) + i}{extension}"
            new_path = os.path.join(action_path, new_name)
            
            os.rename(old_path, new_path)
            print(f"  Renamed: {f} -> {new_name}")

def main():
    parser = argparse.ArgumentParser(description="SIBI Data Migration Tool")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--undo", action="store_true", help="Remove _done suffix from files")
    group.add_argument("--mark-done", action="store_true", help="Add _done suffix to all files manually")
    args = parser.parse_args()

    print("--- SIBI Data Migration Tool ---")
    if args.undo:
        print("Resetting ingestion status (removing _done suffixes)...")
    elif args.mark_done:
        print("Manually marking all files as ingested (_done)...")
    else:
        print("Consolidating folders and updating naming scheme...")
    
    migrate_folder("raw_videos", ".avi", undo=args.undo, mark_done=args.mark_done)
    migrate_folder("raw_glove", ".csv", undo=args.undo, mark_done=args.mark_done)
    
    print("\nOperation complete!")

if __name__ == "__main__":
    main()
