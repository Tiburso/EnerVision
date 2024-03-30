# From the dataset, create a train and test set
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from torchmetrics.functional import jaccard_index
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.base import BaseModel
from models.architectures import (
    DeepLabModel,
    FCNResNetModel,
    MaskRCNNModel,
    Yolov8Model,
)

from dataloaders.solar_dk_dataset import SolarDKDataset
import torchvision.transforms.v2 as transforms


"""
Combined loss class combines both the binary cross entropy loss and the dice loss
"""


class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        bce_loss = self.bce_loss(y_pred, y_true)
        jaccard_loss = jaccard_index(y_pred, y_true, task="binary")

        # Normalize the jaccard loss
        jaccard_loss = 1 - jaccard_loss

        return self.alpha * bce_loss + (1 - self.alpha) * jaccard_loss


train_folder = "data/solardk_dataset_neurips_v2/gentofte_trainval/train"
validation_folder = "data/solardk_dataset_neurips_v2/gentofte_trainval/val"
test_folder = "data/solardk_dataset_neurips_v2/herlev_test/test"

train_transform = transforms.Compose(
    [
        transforms.ToDtype(torch.uint8, scale=True),
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToDtype(torch.uint8, scale=True),
        transforms.Resize((512, 512)),
    ]
)

# Apply the transformations to the dataset
train_dataset = SolarDKDataset(train_folder, transform=train_transform)
validation_dataset = SolarDKDataset(validation_folder, transform=test_transform)
test_dataset = SolarDKDataset(test_folder, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
validation_loader = DataLoader(
    validation_dataset, batch_size=8, shuffle=False, num_workers=4
)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = DeepLabModel(num_classes=1, backbone="resnet50")
# model = FCNResNetModel(num_classes=1, backbone="resnet50")
# model = MaskRCNNModel(num_classes=1)
# model = Yolov8Model(num_classes=1)

loss_fn = CombinedLoss(alpha=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

base_model = BaseModel(model, loss_fn, optimizer, scheduler)
trainer = pl.Trainer(
    strategy="ddp_find_unused_parameters_true",
    max_epochs=150,
    min_epochs=10,
    enable_checkpointing=True,
)

trainer.fit(
    base_model,
    train_loader,
    validation_loader,
)

trainer.test(base_model, test_loader)
