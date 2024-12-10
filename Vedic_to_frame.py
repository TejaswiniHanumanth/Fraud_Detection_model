import cv2
video_path = "path_to_video.mp4"
vidcap = cv2.VideoCapture(video_path)
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite(f"frame_{count}.jpg", image)
    success, image = vidcap.read()
    count += 1
