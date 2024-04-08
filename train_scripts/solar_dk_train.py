# From the dataset, create a train and test set
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from segmentation_models_pytorch.losses import (
    JaccardLoss,
)

import torchvision.transforms.v2 as transforms

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from models.base import BaseModel

from dataloaders.solar_dk_dataset import SolarDKDataset


class LossJaccard(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = JaccardLoss(mode="multiclass")

    def forward(self, y_hat, y):
        y = y.argmax(dim=1)
        return self.loss(y_hat, y)


def main(best_model="last"):
    # SOLAR DK DATASET ---------------------
    solar_dk_train_folder = "data/solardk_dataset_neurips_v2/gentofte_trainval/train"
    solar_dk_validation_folder = "data/solardk_dataset_neurips_v2/gentofte_trainval/val"
    solar_dk_test_folder = "data/solardk_dataset_neurips_v2/herlev_test/test"

    ## LOAD THE DATASET
    # Define the transforms
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
        ]
    )

    train_dataset = SolarDKDataset(
        solar_dk_train_folder, total_samples=1000, transform=transform
    )

    validation_dataset = SolarDKDataset(solar_dk_validation_folder)

    test_dataset = SolarDKDataset(solar_dk_test_folder)

    ## CREATE THE DATALOADERS
    solar_dk_train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    solar_dk_validation_loader = DataLoader(
        validation_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    solar_dk_test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    # DEFINE THE MODEL
    base_model = BaseModel.load_from_checkpoint(
        f"lightning_logs/version_{best_model}/checkpoints/last.ckpt",
    )

    model = base_model.model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=5)

    loss_fn = LossJaccard()
    base_model = BaseModel(model, loss_fn, optimizer, scheduler=scheduler)

    solar_dk_trainer = pl.Trainer(
        num_nodes=1,
        strategy="ddp_spawn",
        accelerator="gpu",
        devices=1,
        max_epochs=100,
        min_epochs=30,
        enable_checkpointing=True,
        callbacks=[
            ModelCheckpoint(
                save_top_k=1,
                save_last="link",
                every_n_train_steps=100,
                monitor="jaccard_index",
                mode="max",
                auto_insert_metric_name=True,
            ),
        ],
    )

    solar_dk_trainer.fit(
        base_model,
        solar_dk_train_loader,
        solar_dk_validation_loader,
    )

    solar_dk_trainer.test(base_model, solar_dk_test_loader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # Add best model number
    parser.add_argument("--best_model", type=str, default="last")

    main(parser.parse_args().best_model)
