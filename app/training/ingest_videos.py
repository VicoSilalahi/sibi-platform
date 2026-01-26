import os
import argparse
from app.data.camera.video_loader import VideoLoader
from app.data.writer import save_sequence
from app.config import ACTIONS, DATA_PATH

def ingest_videos(input_dir='raw_videos'):
    """Ingest raw videos from directory into the landmark dataset."""
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' not found.")
        return

    # Using VideoLoader to process each action subdirectory
    for action in os.listdir(input_dir):
        if action not in ACTIONS:
            # Skip only if it's not a known action and not the 'processed' global folder (if it exists)
            if action != "processed":
                print(f"Skipping unknown action: {action}")
            continue
            
        action_dir = os.path.join(input_dir, action)
        if not os.path.isdir(action_dir):
            continue
            
        print(f"Processing videos for action: '{action}'")
        loader = VideoLoader(action_dir)
        from app.config import INGESTED_SUFFIX
        def save_callback(filename, landmarks):
            video_path = os.path.join(action_dir, filename)
            if landmarks:
                save_sequence(action, landmarks, modality='camera')
                
                # Mark as ingested by renaming with suffix
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}{INGESTED_SUFFIX}{ext}"
                dest_path = os.path.join(action_dir, new_filename)
                
                os.rename(video_path, dest_path)
                print(f"Ingested and marked: {new_filename}")
            else:
                print(f"Failed to extract landmarks from {filename}")

        loader.process_directory(save_callback)
        print(f"Finished processing '{action}'")

def main():
    parser = argparse.ArgumentParser(description="Ingest Raw Videos into SIBI Dataset")
    parser.add_argument("--indir", type=str, default="raw_videos", help="Directory containing action subfolders with videos")
    
    args = parser.parse_args()
    ingest_videos(args.indir)

if __name__ == "__main__":
    main()
