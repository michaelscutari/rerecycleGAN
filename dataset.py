import os
from PIL import Image
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import hflip

class RecycleGANDataset(Dataset):
    def __init__(self, video_path_A, video_path_B, transform=None, frame_size=(432, 240), frame_rate=1):
        self.video_path_A = video_path_A
        self.video_path_B = video_path_B
        self.transform = transform
        self.frame_size = frame_size  # Set to (432, 240)
        self.frame_rate = frame_rate
        self.frames_A = self._load_frames(self.video_path_A)
        self.frames_B = self._load_frames(self.video_path_B)

        # Ensure both videos have enough frames
        min_length = min(len(self.frames_A), len(self.frames_B))
        if min_length < 3:
            raise ValueError("Both videos must have at least 3 frames after processing.")
        # Truncate to the shortest length
        self.frames_A = self.frames_A[:min_length]
        self.frames_B = self.frames_B[:min_length]

    def _load_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
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
            # Convert BGR (OpenCV format) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        return frames

    def __len__(self):
        # Subtract 2 because we need previous, current, and next frames
        return len(self.frames_A) - 2

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        # For video A
        prev_frame_A = self.frames_A[idx]
        current_frame_A = self.frames_A[idx + 1]
        next_frame_A = self.frames_A[idx + 2]
        # For video B
        prev_frame_B = self.frames_B[idx]
        current_frame_B = self.frames_B[idx + 1]
        next_frame_B = self.frames_B[idx + 2]

        return (
            current_frame_A,
            current_frame_B,
            prev_frame_A,
            prev_frame_B,
            next_frame_A,
            next_frame_B,)