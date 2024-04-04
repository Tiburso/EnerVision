# From the dataset, create a train and test set
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from losses import AsymmetricUnifiedFocalLoss, CombinedLoss

import torchvision.transforms.v2 as transforms

from torch.optim.lr_scheduler import PolynomialLR, ReduceLROnPlateau

# Add EarlyStopping
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models.base import BaseModel
from models.architectures import (
    DeepLabModel,
)

from dataloaders.solar_dk_dataset import SolarDKDataset
from dataloaders.nl_dataset import CocoSegmentationDataset


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
    train_dataset, batch_size=4, shuffle=True, num_workers=4
)
solar_dk_validation_loader = DataLoader(
    validation_dataset, batch_size=4, shuffle=False, num_workers=4
)
solar_dk_test_loader = DataLoader(
    test_dataset, batch_size=4, shuffle=False, num_workers=4
)

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
    nl_train_dataset, batch_size=4, shuffle=True, num_workers=4
)
nl_validation_loader = DataLoader(
    nl_validation_dataset, batch_size=4, shuffle=False, num_workers=4
)
nl_test_loader = DataLoader(nl_test_dataset, batch_size=4, shuffle=False, num_workers=4)

# DEFINE THE MODEL
model = DeepLabModel(num_classes=2, backbone="resnet152")

loss_fn = CombinedLoss()
# loss_fn = AsymmetricUnifiedFocalLoss(weight=0.3, delta=0.6, gamma=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# scheduler = PolynomialLR(optimizer, power=0.9, total_iters=3000)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10)
# scheduler = None

base_model = BaseModel(model, loss_fn, optimizer, scheduler=scheduler)

# base_model = BaseModel.load_from_checkpoint(
#     "lightning_logs/version_211837/checkpoints/last.ckpt"
# )

nl_trainer = pl.Trainer(
    num_nodes=1,
    strategy="ddp",
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

# Second iteration of training

loss_fn = AsymmetricUnifiedFocalLoss(weight=0.3, delta=0.2, gamma=2)
base_model.loss_fn = loss_fn

solar_dk_trainer = pl.Trainer(
    num_nodes=1,
    strategy="ddp",
    accelerator="gpu",
    devices=1,
    max_epochs=150,
    min_epochs=50,
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
    ckpt_path="best",
)

solar_dk_trainer.test(base_model, solar_dk_test_loader, ckpt_path="best")
