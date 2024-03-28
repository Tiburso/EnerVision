import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch import nn


class DeepLabModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load a pre-trained DeepLab model
        self.model = deeplabv3_resnet50(
            pretrained=True, weights=DeepLabV3_ResNet50_Weights.DEFAULT
        )

        # Change the classifier to output the number of classes
        self.model.classifier = DeepLabHead(2048, num_classes)

    def forward(self, x):
        x = self.model(x)["out"]
        return x

    def target(self, y):
        return y
