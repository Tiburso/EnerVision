import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
)

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch import nn


class DeepLabModel(nn.Module):
    def __init__(self, num_classes, backbone="resnet50"):
        super().__init__()
        # Load a pre-trained DeepLab model
        if backbone == "resnet50":
            self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        else:
            self.model = deeplabv3_resnet101(
                weights=DeepLabV3_ResNet101_Weights.DEFAULT
            )

        # Change the classifier to output the number of classes
        self.model.classifier = DeepLabHead(2048, num_classes)

    def forward(self, x):
        x = self.model(x)["out"]
        return x

    def target(self, y):
        return y
