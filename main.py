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
from dataloaders.nl_dataset import CocoSegmentationDataset


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(mode="binary")
        self.jaccard_loss = JaccardLoss(mode="binary")

    def forward(self, y_pred, y_true):
        cross_entropy_loss = self.cross_entropy(y_pred, y_true)
        dice_loss = self.dice_loss(y_pred, y_true)
        jaccard_loss = self.jaccard_loss(y_pred, y_true)

        return cross_entropy_loss + 2 * dice_loss + 3 * jaccard_loss


## SOLAR DK DATASET
solar_dk_train_folder = "data/solardk_dataset_neurips_v2/gentofte_trainval/train"
solar_dk_validation_folder = "data/solardk_dataset_neurips_v2/gentofte_trainval/val"
solar_dk_test_folder = "data/solardk_dataset_neurips_v2/herlev_test/test"

## NL DATASET
nl_train_folder = "data/NL-Solar-Panel-Seg-1/train"
nl_validation_folder = "data/NL-Solar-Panel-Seg-1/valid"
nl_test_folder = "data/NL-Solar-Panel-Seg-1/test"

## LOAD THE DATASET
train_dataset = CocoSegmentationDataset(nl_train_folder) + SolarDKDataset(
    solar_dk_train_folder
)

validation_dataset = CocoSegmentationDataset(nl_validation_folder) + SolarDKDataset(
    solar_dk_validation_folder
)

test_dataset = CocoSegmentationDataset(nl_test_folder) + SolarDKDataset(
    solar_dk_test_folder
)


## CREATE THE DATALOADERS
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
validation_loader = DataLoader(
    validation_dataset, batch_size=4, shuffle=False, num_workers=4
)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)


# DEFINE THE MODEL
model = DeepLabModel(num_classes=2, backbone="resnet152")

# loss_fn = CombinedLoss()
loss_fn = AsymmetricUnifiedFocalLoss(weight=0.3, delta=0.6, gamma=2)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# scheduler = PolynomialLR(optimizer, power=0.9, total_iters=100)
scheduler = None

base_model = BaseModel(model, loss_fn, optimizer, scheduler=scheduler)

trainer = pl.Trainer(
    num_nodes=1,
    strategy="ddp",
    accelerator="gpu",
    devices=1,
    max_epochs=15,
    min_epochs=1,
    enable_checkpointing=True,
    callbacks=[
        EarlyStopping(monitor="jaccard_index", mode="max", patience=2),
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

trainer.fit(
    base_model,
    train_loader,
    validation_loader,
)

trainer.test(base_model, test_loader)
