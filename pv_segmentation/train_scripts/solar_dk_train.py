import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import torchvision.transforms.v2 as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models.base import BaseModel
from dataloaders.solar_dk_dataset import SolarDKDataset
from dataloaders.nl_dataset import NLSegmentationDataset
from pytorch_lightning.loggers import WandbLogger
import wandb
from models.architectures.base_cnn import BaseCNN
from models.architectures.unet import UNetModel
from models.architectures.deep_lab import DeepLabModel
from models.architectures.fcn import FCN
from losses import LossCombined, LossCE, LossDice, LossJaccard, LossFocal
import numpy as np


def main(best_model="last"):
    wandb_logger = WandbLogger(
        project="Funetuning on SolarDK",
        entity="5ARIP",
        config={
            "learning_rate": 1e-5,
            "architecture": "DeepLabV3+",
            "dataset": "SolarDK",
        },
    )

    # SOLAR DK DATASET ---------------------
    solar_dk_train_folder = "data/solardk_dataset_neurips_v2/gentofte_trainval/train"
    solar_dk_validation_folder = "data/solardk_dataset_neurips_v2/gentofte_trainval/val"
    solar_dk_test_folder = "data/solardk_dataset_neurips_v2/herlev_test/test"

    # LOAD THE DATASET
    train_dataset = SolarDKDataset(image_dir=solar_dk_train_folder)
    validation_dataset = SolarDKDataset(image_dir=solar_dk_validation_folder)
    test_dataset = SolarDKDataset(image_dir=solar_dk_test_folder)

    ## CREATE THE DATALOADERS
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    # Define path to checkpoint previous model
    checkpoint_path = f"Training on NL Dataset/caat99ql/checkpoints/25-0.83.ckpt"

    # Load model from checkpoint
    base_model = BaseModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # for param in base_model.model.model.encoder.parameters():
    #     param.requires_grad = False

    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, base_model.model.parameters()), lr=1e-7)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-8)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )
    # scheduler = ExponentialLR(optimizer, gamma=0.95, verbose=True)

    base_model.optimizer = optimizer
    base_model.scheduler = scheduler

    base_model.loss_fn = LossCombined()

    check_point_callback = ModelCheckpoint(
        monitor="val_jaccard",
        mode="max",
        save_top_k=1,
        save_last=False,
        every_n_epochs=1,
        filename="{epoch:02d}-{val_jaccard:.2f}",
        auto_insert_metric_name=False,
        verbose=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_jaccard", patience=10, mode="max"
    )

    solar_dk_trainer = pl.Trainer(
        num_nodes=1,
        strategy="auto",
        accelerator="gpu",
        devices=1,
        max_epochs=100,
        min_epochs=5,
        enable_checkpointing=True,
        logger=wandb_logger,
        callbacks=[check_point_callback, early_stopping_callback],
        log_every_n_steps=10,
    )

    solar_dk_trainer.fit(
        model=base_model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )

    solar_dk_trainer.test(base_model, test_loader)

    wandb.finish()


if __name__ == "__main__":
    import argparse

    # Get the arguments
    parser = argparse.ArgumentParser()
    # Add best model number
    parser.add_argument("--best_model", type=str, default="best_model")
    main(parser.parse_args().best_model)
