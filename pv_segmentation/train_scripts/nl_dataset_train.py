import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models.base import BaseModel
from dataloaders.nl_dataset import NLSegmentationDataset
from dataloaders.germany_dataset import GermanyDataset
from pytorch_lightning.loggers import WandbLogger
import wandb
from models.architectures.deep_lab import DeepLabModel
from losses import LossCombined, LossCE, LossDice, LossJaccard, LossFocal
from sklearn.model_selection import train_test_split
import torch.utils.data as Data


def main(best_model="last"):
    # SET UP WEIGHTS & BIASES ENVIRONMENT
    wandb_logger = WandbLogger(
        project="Training model",
        entity="5ARIP",
        config={
            "learning_rate": 1e-5,
            "architecture": "DeepLabV3+",
            "dataset": "NL Segmentation + Germany",
        },
    )

    # LOAD NL DATASET -------------------------------------------------
    nl_train_folder = "data/NL-Solar-Panel-Seg-1/train"
    nl_validation_folder = "data/NL-Solar-Panel-Seg-1/valid"
    nl_test_folder = "data/NL-Solar-Panel-Seg-1/test"

    nl_train_dataset = NLSegmentationDataset(image_dir=nl_train_folder)
    nl_validation_dataset = NLSegmentationDataset(image_dir=nl_validation_folder)
    nl_test_dataset = NLSegmentationDataset(image_dir=nl_test_folder)

    # LOAD GERMANY DATASET --------------------------------------------
    germany_folder = "data/bdappv"
    germany_dataset = GermanyDataset(germany_folder)

    # Create a train, validation and test set
    train_indices, test_indices = train_test_split(
        range(len(germany_dataset)), test_size=0.2, random_state=0
    )
    train_indices, validation_indices = train_test_split(
        train_indices, test_size=0.2, random_state=0
    )
    germany_train_dataset = Data.Subset(germany_dataset, train_indices)
    germany_validation_dataset = Data.Subset(germany_dataset, validation_indices)
    germany_test_dataset = Data.Subset(germany_dataset, test_indices)

    # CONCAT NL AND GERMANY DATASET -----------------------------------
    train_dataset = Data.ConcatDataset([germany_train_dataset, nl_train_dataset])
    validation_dataset = Data.ConcatDataset(
        [germany_validation_dataset, nl_validation_dataset]
    )
    test_dataset = Data.ConcatDataset([germany_test_dataset, nl_test_dataset])

    # CREATE THE DATALOADERS
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

    # DEFINE THE MODEL, OPTIMIZER, SCHEDULER and LOSS FUNCTION
    model = DeepLabModel(num_classes=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.95, verbose=True)
    loss_fn = LossCombined()

    # LOAD BASE MODEL
    base_model = BaseModel(model, loss_fn, optimizer, scheduler=scheduler)

    # CHECKPOINT
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

    # EARLY STOPPING
    early_stopping_callback = EarlyStopping(
        monitor="val_jaccard", patience=10, mode="max"
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

    # START TRAINING
    trainer.fit(
        model=base_model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
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
