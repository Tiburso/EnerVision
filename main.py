# From the dataset, create a train and test set
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from torchvision.ops.focal_loss import sigmoid_focal_loss

from models.base import BaseModel
from models.architectures import (
    DeepLabModel,
    FCNResNetModel,
    MaskRCNNModel,
    Yolov8Model,
)

from dataloaders.solar_dk_dataset import SolarDKDataset
import torchvision.transforms.v2 as transforms


class SigmoidFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        return sigmoid_focal_loss(input, target, self.gamma, self.alpha)


train_folder = "data/solardk_dataset_neurips_v2/gentofte_trainval/train"
validation_folder = "data/solardk_dataset_neurips_v2/gentofte_trainval/val"
test_folder = "data/solardk_dataset_neurips_v2/herlev_test/test"

transform = transforms.Compose(
    [
        transforms.ToDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = SolarDKDataset(train_folder, transform=transform)
validation_dataset = SolarDKDataset(validation_folder, transform=transform)
test_dataset = SolarDKDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
validation_loader = DataLoader(
    validation_dataset, batch_size=16, shuffle=False, num_workers=4
)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = DeepLabModel(num_classes=1, backbone="resnet101")
# model = FCNResNetModel(num_classes=1, backbone="resnet50")
# model = MaskRCNNModel(num_classes=1)
# model = Yolov8Model(num_classes=1)

loss_fn = SigmoidFocalLoss(alpha=0.90)
optimizer = torch.optim.AdamW(model.parameters())

base_model = BaseModel(model, loss_fn, optimizer)
trainer = pl.Trainer(
    strategy="ddp_find_unused_parameters_true",
    max_epochs=10,
    min_epochs=3,
    enable_checkpointing=True,
)

trainer.fit(
    base_model,
    train_loader,
    validation_loader,
)

trainer.test(base_model, test_loader)
