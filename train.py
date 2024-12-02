# train.py

import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from model import RecycleGAN
# import dataset here
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def train():
    # insert transforms and dataset here

    # dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # model
    model = RecycleGAN(
        l_adv=1.0,
        l_cycle=10.0,
        l_iden=5.0,
        l_temp=1.0,
        learning_rate_d=0.0002,
        learning_rate_g=0.0002
    )

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='recyclegan-epoch{epoch}',
        every_n_epochs=5,        
        save_top_k=-1,            
        save_last=True           
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # trainer
    trainer = pl.Trainer(
        max_epochs=100,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16,  # mixed precision
        progress_bar_refresh_rate=20,
        gradient_clip_val=1.0  # gradient clipping
    )

    # train
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    train()