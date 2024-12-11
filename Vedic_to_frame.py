import cv2
import os
import argparse
def extract_frames_from_videos(input_dir, output_dir, fps):
    """
    Extract frames from all videos in a directory at the specified frame rate (fps).
    Args:
        input_dir (str): Directory containing video files.
        output_dir (str): Directory to save extracted frames.
        fps (int): Frames per second to extract.
    """
    os.makedirs(output_dir, exist_ok=True)
    for video_name in os.listdir(input_dir):
        if video_name.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(input_dir, video_name)
            video_output_dir = os.path.join(output_dir, os.path.splitext(video_name)[0])
            os.makedirs(video_output_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Unable to open video file {video_path}")
                continue
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(video_fps / fps)
            frame_count = 0
            saved_frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    frame_filename = os.path.join(video_output_dir, f"frame_{saved_frame_count:04d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    saved_frame_count += 1
                frame_count += 1
            cap.release()
            print(f"Extracted {saved_frame_count} frames from {video_name} and saved to {video_output_dir}")
if _name_ == "_main_":
    parser = argparse.ArgumentParser(description="Extract frames from videos in a directory")
    parser.add_argument("--input", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--output", type=str, required=True, help="Directory to save extracted frames")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second to extract")
    args = parser.parse_args()
    extract_frames_from_videos(args.input, args.output, args.fps)
