# From the dataset, create a train and test set
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from losses import AsymmetricUnifiedFocalLoss

from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
from torchmetrics.functional.classification import dice

from torch.optim.lr_scheduler import PolynomialLR

# Add EarlyStopping
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models.base import BaseModel
from models.architectures import (
    DeepLabModel,
)

from dataloaders.solar_dk_dataset import SolarDKDataset
import torchvision.transforms.v2 as transforms


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(mode="binary")
        self.jaccard_loss = JaccardLoss(mode="binary")

    def forward(self, y_pred, y_true):
        cross_entropy_loss = self.cross_entropy(y_pred, y_true)
        dice_loss = self.dice_loss(y_pred, y_true)
        jaccard_loss = self.jaccard_loss(y_pred, y_true)

        return cross_entropy_loss + 2 * dice_loss + 3 * jaccard_loss


train_folder = "data/solardk_dataset_neurips_v2/gentofte_trainval/train"
validation_folder = "data/solardk_dataset_neurips_v2/gentofte_trainval/val"
test_folder = "data/solardk_dataset_neurips_v2/herlev_test/test"

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
    ]
)

# Apply the transformations to the dataset
train_dataset = SolarDKDataset(train_folder, transform=train_transform)
validation_dataset = SolarDKDataset(validation_folder)
test_dataset = SolarDKDataset(test_folder)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
validation_loader = DataLoader(
    validation_dataset, batch_size=8, shuffle=False, num_workers=4
)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

model = DeepLabModel(num_classes=1, backbone="resnet152")

loss_fn = CombinedLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
scheduler = PolynomialLR(optimizer, power=0.9, total_iters=100)

base_model = BaseModel(model, loss_fn, optimizer, scheduler=scheduler)

trainer = pl.Trainer(
    max_epochs=150,
    min_epochs=5,
    enable_checkpointing=True,
    callbacks=[
        EarlyStopping(monitor="val_jaccard", mode="max", patience=5),
        ModelCheckpoint(
            save_top_k=1,
            save_last="link",
            monitor="val_jaccard",
            mode="max",
            auto_insert_metric_name=True,
        ),
    ],
)

trainer.fit(
    base_model,
    train_loader,
    validation_loader,
)

trainer.fit(
    base_model,
    train_loader,
    validation_loader,
)

trainer.test(base_model, test_loader)
