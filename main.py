# From the dataset, create a train and test set
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from models.base import BaseModel
from models.architectures.deep_lab import DeepLabModel
from models.architectures.mask_rcnn import MaskRCNNModel
from models.architectures.yolov8 import Yolov8Model

from dataloaders.solar_dk_dataset import SolarDKDataset
import torchvision.transforms.v2 as transforms

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

model = DeepLabModel(num_classes=2)
# model = MaskRCNNModel(num_classes=1)
# model = Yolov8Model(num_classes=1)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
