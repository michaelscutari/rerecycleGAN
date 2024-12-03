import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, video_path, transform=None, frame_size=(432, 240), frame_rate=1):
        self.video_path = video_path
        self.transform = transform
        self.frame_size = frame_size  # Set to (432, 240)
        self.frame_rate = frame_rate
        self.frames = self._load_frames()

    def _load_frames(self):
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % self.frame_rate != 0:
                continue
            # Resize the frame to the desired frame size
            frame = cv2.resize(frame, self.frame_size)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        return frames

    def __len__(self):
        # Subtract 2 because we need previous, current, and next frames
        return len(self.frames) - 2

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        prev_frame = self.frames[idx]
        current_frame = self.frames[idx + 1]
        next_frame = self.frames[idx + 2]
        return prev_frame, current_frame, next_frame