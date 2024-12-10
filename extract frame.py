import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a video at a given frame rate.
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save the frames.
        frame_rate (int): Number of frames to extract per second.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    
    frame_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_name, frame)
        frame_count += 1
        success, frame = cap.read()
    
    cap.release()
    print(f"Frames extracted to {output_dir}")

# Example usage:
extract_frames("path_to_video.mp4", "output_frames_directory", frame_rate=1)
