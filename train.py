# train.py

import os
import torch
import pytorch_lightning as pl
from torchvision import transforms
from model import RecycleGAN
from data_module import RecycleGANDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')

wandb_logger = WandbLogger(project="rerecycleGAN", log_model=True)

def train():
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((240, 432)),  # Ensure the frames are resized correctly
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # DataModule
    data_module = RecycleGANDataModule(
        video_path_A="videos/trimmed/trimmed_animated.mp4",
        video_path_B="videos/trimmed/trimmed_live_action.mp4",
        batch_size=4,
        transform=transform
    )

    # Model
    model = RecycleGAN(
        l_adv=1.0,
        l_cycle=10.0,
        l_iden=5.0,
        l_temp=1.0,
        learning_rate_d=0.0001,
        learning_rate_g=0.0002,
        learning_rate_p=0.0002,
        lr_warmup_epochs=10
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='recyclegan-epoch{epoch}',
        every_n_epochs=5,
        save_top_k=-1,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        accelerator='gpu',
        devices=-1,
        strategy='ddp_find_unused_parameters_true', # some parameters not used during updates (e.g. in the generator, discriminator is not used) 
        callbacks=[checkpoint_callback, lr_monitor],
        precision='16-mixed',  # mixed precision
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    train()