# train.py

import os
import torch
import pytorch_lightning as pl
from torchvision import transforms
from model import RecycleGAN
from data_module import RecycleGANDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def train():
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((240, 432)),  # Ensure the frames are resized correctly
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # DataModule
    data_module = RecycleGANDataModule(
        video_path_A="videos/domainA_video.mp4",
        video_path_B="videos/domainB_video.mp4",
        batch_size=4,
        transform=transform
    )

    # Model
    model = RecycleGAN(
        l_adv=1.0,
        l_cycle=10.0,
        l_iden=5.0,
        l_temp=1.0,
        learning_rate_d=0.0002,
        learning_rate_g=0.0002
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='recyclegan-epoch{epoch}',
        every_n_epochs=5,
        save_top_k=-1,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(lfogging_interval='epoch')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16,  # mixed precision
        enable_progress_bar=True,
        gradient_clip_val=1.0  # gradient clipping
    )

    # Train
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    train()