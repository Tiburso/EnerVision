import torch
from torch import nn
from torchvision.models.segmentation import (
    fcn_resnet101,
    fcn_resnet50,
    FCN_ResNet101_Weights,
    FCN_ResNet50_Weights,
)

from torchvision.models.segmentation.fcn import FCNHead


class FCNResNetModel(nn.Module):
    def __init__(self, num_classes, backbone="resnet50"):
        super().__init__()
        # Load a pre-trained FCN model
        if backbone == "resnet50":
            self.model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
        else:
            self.model = fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT)

        # Change the classifier to output the number of classes
        self.model.classifier = FCNHead(2048, num_classes)

    def forward(self, x):
        x = self.model(x)["out"]
        return x

    def target(self, y):
        return y
