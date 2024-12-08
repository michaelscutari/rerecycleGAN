# extract_frames.py
import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, frame_size=(432, 240), frame_rate=1):
    """
    Extract frames from a video file.
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_size: Tuple of (width, height)
        frame_rate: Extract every nth frame
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read")
            break
            
        frame_count += 1
        if frame_count % frame_rate != 0:
            continue
            
        # Resize frame
        frame = cv2.resize(frame, frame_size)
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        saved_count += 1
        
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    video_path = "../videos/trimmed/trimmed_live_action.mp4"
    output_dir = "extracted_frames"
    extract_frames(video_path, output_dir, frame_rate=60)