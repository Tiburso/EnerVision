# From the dataset, create a train and test set
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from losses import CombinedLoss

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from models.base import BaseModel
from models.architectures import (
    DeepLabModel,
)

from dataloaders.nl_dataset import CocoSegmentationDataset

torch.manual_seed(0)
torch.set_num_threads(4)


def main():
    # NL DATASET ---------------------
    nl_train_folder = "data/NL-Solar-Panel-Seg-1/train"
    nl_validation_folder = "data/NL-Solar-Panel-Seg-1/valid"
    nl_test_folder = "data/NL-Solar-Panel-Seg-1/test"

    ## LOAD THE DATASET
    nl_train_dataset = CocoSegmentationDataset(nl_train_folder)
    nl_validation_dataset = CocoSegmentationDataset(nl_validation_folder)
    nl_test_dataset = CocoSegmentationDataset(nl_test_folder)

    ## CREATE THE DATALOADERS
    nl_train_loader = DataLoader(
        nl_train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    nl_validation_loader = DataLoader(
        nl_validation_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    nl_test_loader = DataLoader(
        nl_test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    # DEFINE THE MODEL
    model = DeepLabModel(num_classes=2, backbone="resnet152")

    loss_fn = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=2)

    base_model = BaseModel(model, loss_fn, optimizer, scheduler=scheduler)

    nl_trainer = pl.Trainer(
        num_nodes=1,
        strategy="ddp_spawn",
        accelerator="gpu",
        devices=1,
        max_epochs=10,
        min_epochs=3,
        enable_checkpointing=True,
        callbacks=[
            ModelCheckpoint(
                save_top_k=1,
                save_last="link",
                every_n_train_steps=1000,
                monitor="jaccard_index",
                mode="max",
                auto_insert_metric_name=True,
            ),
        ],
    )

    # First iteration of training
    nl_trainer.fit(
        base_model,
        nl_train_loader,
        nl_validation_loader,
    )

    nl_trainer.test(base_model, nl_test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
