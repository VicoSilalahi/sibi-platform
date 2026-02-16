import os
import shutil
import argparse

def merge_directories(source_dirs, target_dir, copy_mode=True):
    """
    Merges multiple SIBI dataset directories into one.
    Structure: source_dir/action_name/file.avi -> target_dir/action_name/prefix_file.avi
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")

    total_files = 0
    
    for src in source_dirs:
        if not os.path.isdir(src):
            print(f"Skipping {src}: Not a directory")
            continue
            
        print(f"\nProcessing source: {src}")
        # Use the source directory name as a prefix, but strip 'raw_videos_'/'raw_glove_'
        # for folder name consistency between hardware types
        folder_name = os.path.basename(os.path.normpath(src))
        prefix = folder_name.replace('raw_videos_', '').replace('raw_glove_', '')
        
        for action in os.listdir(src):
            action_path = os.path.join(src, action)
            if not os.path.isdir(action_path):
                continue
                
            target_action_path = os.path.join(target_dir, action)
            os.makedirs(target_action_path, exist_ok=True)
            
            files = [f for f in os.listdir(action_path) if f.endswith(('.avi', '.mp4', '.csv'))]
            for f in files:
                src_file = os.path.join(action_path, f)
                # New filename: prefix_original-name.avi
                new_filename = f"{prefix}_{f}"
                dest_file = os.path.join(target_action_path, new_filename)
                
                if copy_mode:
                    shutil.copy2(src_file, dest_file)
                else:
                    shutil.move(src_file, dest_file)
                
                total_files += 1
                
            print(f"  - Action '{action}': {len(files)} files {'copied' if copy_mode else 'moved'}")

    print("\n" + "="*30)
    print(f"Merge Complete! Total files: {total_files}")
    print(f"Target: {target_dir}")
    print("="*30)

def main():
    parser = argparse.ArgumentParser(description="Merge multiple SIBI dataset directories into one unified database.")
    parser.add_argument("--sources", nargs="+", help="Space-separated list of source directories to merge")
    parser.add_argument("--parent", type=str, help="Parent directory containing multiple recording sessions to merge")
    parser.add_argument("--dest", type=str, default="unified_database", help="Target directory for the merged database")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them (deletes originals)")

    args = parser.parse_args()

    sources = []
    if args.sources:
        sources.extend(args.sources)
    
    if args.parent:
        if os.path.isdir(args.parent):
            # Add all subdirectories of the parent
            subdirs = [os.path.join(args.parent, d) for d in os.listdir(args.parent) 
                      if os.path.isdir(os.path.join(args.parent, d))]
            sources.extend(subdirs)
        else:
            print(f"Error: Parent directory {args.parent} not found.")
            return

    if not sources:
        print("No source directories specified. Use --sources or --parent.")
        return

    merge_directories(sources, args.dest, copy_mode=not args.move)

if __name__ == "__main__":
    main()
