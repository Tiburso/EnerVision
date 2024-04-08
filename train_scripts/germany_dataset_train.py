# From the dataset, create a train and test set
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from losses import CombinedLoss
from train_scripts.solar_dk_train import LossJaccard

import torchvision.transforms.v2 as transforms

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from models.base import BaseModel

from dataloaders.germany_dataset import GermanyDataset

from sklearn.model_selection import train_test_split

torch.manual_seed(0)
torch.set_num_threads(4)


def main(best_model: str = "last"):
    # GERMANY DATASET ---------------------
    germany_folder = "data/germany_dataset"

    ## LOAD THE DATASET
    # Define the transforms
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0)),
        ]
    )

    dataset = GermanyDataset(germany_folder, transform=transform)

    # Create a train, validation and test set
    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=0
    )

    train_indices, validation_indices = train_test_split(
        train_indices, test_size=0.2, random_state=0
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(dataset, validation_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    ## CREATE THE DATALOADERS
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    # DEFINE THE MODEL
    base_model = BaseModel.load_from_checkpoint(
        "lightning_logs/version_240425/checkpoints/last.ckpt",
    )

    model = base_model.model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=5)
    loss_fn = CombinedLoss()

    base_model = BaseModel(model, loss_fn, optimizer, scheduler=scheduler)

    solar_dk_trainer = pl.Trainer(
        num_nodes=1,
        strategy="ddp_spawn",
        accelerator="gpu",
        devices=1,
        max_epochs=150,
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
        train_loader,
        validation_loader,
    )

    solar_dk_trainer.test(base_model, test_loader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--best_model", type=str, default="last")

    main(parser.parse_args().best_model)
