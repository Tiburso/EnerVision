import torch
import torchvision.models as models
import pytorch_lightning as pl
from torch import nn, optim

class ResNetModel(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        # Load a pre-trained ResNet
        model = models.resnet18(pretrained=True)
        # Modify the final layer to match the number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)