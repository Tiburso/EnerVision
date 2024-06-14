import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models.base import BaseModel
from dataloaders.nl_dataset import NLSegmentationDataset
from pv_segmentation.dataloaders.france_dataset import FranceDataset
from pytorch_lightning.loggers import WandbLogger
import wandb
from models.architectures.deep_lab import DeepLabModel
from losses import LossCombined, LossCE, LossDice, LossJaccard, LossFocal
from sklearn.model_selection import train_test_split
import torch.utils.data as Data


def main(best_model="last"):
    # SET UP WEIGHTS & BIASES ENVIRONMENT
    wandb_logger = WandbLogger(
        project = "Test model",
        entity = "5ARIP",
        config={
        "dataset": "NL Segmentation + France",
        }
    )

    # LOAD NL DATASET -------------------------------------------------
    nl_test_folder = "data/NL-Solar-Panel-Seg-1/test"
    nl_test_dataset = NLSegmentationDataset(image_dir=nl_test_folder)

    # LOAD FRANCE DATASET --------------------------------------------
    france_folder = "data/bdappv"
    france_dataset = FranceDataset(france_folder)

    # Create a train, validation and test set
    train_indices, test_indices = train_test_split(
        range(len(france_dataset)), test_size=0.2, random_state=0
    )

    germany_test_dataset = Data.Subset(france_dataset, test_indices)

    test_dataset = Data.ConcatDataset([germany_test_dataset, nl_test_dataset])

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    # LOAD MODEL
    checkpoint_path = "pv_segmentation/16-0.84.ckpt"
    base_model = BaseModel.load_from_checkpoint(checkpoint_path)

    # CHECKPOINT
    check_point_callback = ModelCheckpoint(
        monitor="val_jaccard",
        mode="max",
        save_top_k=1,
        save_last=False,
        every_n_epochs=1,
        filename="{epoch:02d}-{val_jaccard:.2f}",
        auto_insert_metric_name=False,
        verbose=True
    )

    # EARLY STOPPING
    early_stopping_callback = EarlyStopping(
        monitor="val_jaccard",
        patience=10,
        mode="max"
    )

    # CREATE TRAINER
    trainer = pl.Trainer(
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

    # TEST MODEL
    trainer.test(base_model, test_loader)

    wandb.finish()


if __name__ == "__main__":
    # Get the arguments
    parser = argparse.ArgumentParser()
    # Add best model number
    parser.add_argument("--best_model", type=str, default="best_model")
    main(parser.parse_args().best_model)
