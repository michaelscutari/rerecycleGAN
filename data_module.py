import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from dataset import RecycleGANDataset

class RecycleGANDataModule(pl.LightningDataModule):
    def __init__(self, video_path_A, video_path_B, batch_size=4, transform=None):
        super().__init__()
        self.video_path_A = video_path_A
        self.video_path_B = video_path_B
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        self.dataset = RecycleGANDataset(
            self.video_path_A,
            self.video_path_B,
            transform=self.transform,
            frame_size=(432, 240),
            frame_rate=1
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset, 
            batch_size=self.batch_size,
            shuffle=False,  # Changed because DistributedSampler handles shuffling
            num_workers=4,  # Add num_workers for parallel data loading
            pin_memory=True,  # Better GPU transfer
            sampler=torch.utils.data.distributed.DistributedSampler(self.dataset)
        )