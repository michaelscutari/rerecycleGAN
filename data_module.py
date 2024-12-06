import pytorch_lightning as pl
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
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)