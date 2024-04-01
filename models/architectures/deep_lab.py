from segmentation_models_pytorch import DeepLabV3Plus
from torch import nn


class DeepLabModel(nn.Module):
    def __init__(self, num_classes, backbone="resnet50"):
        super().__init__()

        self.model = DeepLabV3Plus(
            encoder_name=backbone,
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def target(self, y):
        return y
