import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from load_config import load_config

config = load_config()

class TransactionDataset(Dataset):
    def __init__(self, video_folder, labels, transform=None, frame_skip=None):
        self.video_folder = video_folder
        self.labels = labels
        self.transform = transform
        self.frame_skip = frame_skip or config['hyperparameters']['frame_skip']
        self.data = self._extract_frames()

    def _extract_frames(self):
        data = []
        for video_name, label in self.labels.items():
            video_path = os.path.join(self.video_folder, video_name)
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.frame_skip == 0:
                    data.append((frame, label))
                frame_idx += 1

            cap.release()
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame, label = self.data[idx]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.transform:
            frame = self.transform(frame)

        return frame, label
