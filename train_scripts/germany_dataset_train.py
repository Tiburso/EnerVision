# From the dataset, create a train and test set
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from losses import CombinedLoss
import torchvision.transforms.v2 as transforms

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from models.base import BaseModel

from dataloaders.germany_dataset import GermanyDataset

torch.manual_seed(0)
torch.set_num_threads(4)

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
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [int(0.7 * len(dataset)), int(0.2 * len(dataset)), int(0.1 * len(dataset))]
)


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
    strategy="ddp",
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
    solar_dk_train_loader,
    solar_dk_validation_loader,
)

solar_dk_trainer.test(base_model, solar_dk_test_loader)
