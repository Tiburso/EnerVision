# From the dataset, create a train and test set
import os
import torch
from sklearn.model_selection import train_test_split
from dataloaders.germany_dataset import GermanyDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from models.base import BaseModel
from models.architectures.deep_lab import DeepLabModel
from models.architectures.mask_rcnn import MaskRCNNModel

dataset = os.listdir("germany_dataset/labels")

train, test = train_test_split(dataset, test_size=0.2)

train_dataset = GermanyDataset(train)
test_dataset = GermanyDataset(test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# model = DeepLabModel(input_size=832, num_classes=2)
model = MaskRCNNModel(num_classes=2)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

base_model = BaseModel(model, loss_fn, optimizer)
trainer = pl.Trainer(max_epochs=10, min_epochs=5, fast_dev_run=True)

trainer.fit(
    base_model,
    train_loader,
)
