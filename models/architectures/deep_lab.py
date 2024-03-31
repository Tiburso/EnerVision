from segmentation_models_pytorch import DeepLabV3Plus

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch import nn


class DeepLabModel(nn.Module):
    def __init__(self, num_classes, backbone="resnet50"):
        super().__init__()
        self.model = DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def target(self, y):
        return y
