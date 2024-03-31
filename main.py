# From the dataset, create a train and test set
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from losses import AsymmetricUnifiedFocalLoss
from torchmetrics.functional.classification import dice

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add EarlyStopping
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models.base import BaseModel
from models.architectures import (
    DeepLabModel,
    FCNResNetModel,
    MaskRCNNModel,
    Yolov8Model,
)

from dataloaders.solar_dk_dataset import SolarDKDataset
import torchvision.transforms.v2 as transforms


class CombinedBCEDiceLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedBCEDiceLoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, y_hat, y):
        bce = self.bce(y_hat, y)
        y_hat = torch.sigmoid(y_hat)
        dice_loss = 1 - dice(y_hat, y.int())
        return self.alpha * bce + (1 - self.alpha) * dice_loss


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

model = DeepLabModel(num_classes=2, backbone="resnet152")

treshold = 0.5
loss_fn = AsymmetricUnifiedFocalLoss(weight=0.4, delta=0.9, gamma=0.3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

base_model = BaseModel(model, loss_fn, optimizer, treshold=treshold)
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

trainer.test(base_model, test_loader)
