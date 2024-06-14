from segmentation_models_pytorch import DeepLabV3Plus
from torch import nn


class DeepLabModel(nn.Module):
    """Class to create a DeepLabv3+ model

    Args:
        num_classes (int): Number of outputs of model
        backbone (str, optional): Backbone to be used in model. Defaults to "resnet101".
    """
    def __init__(self, num_classes, backbone="resnet101"):
        super().__init__()

        self.model = DeepLabV3Plus(
            encoder_name=backbone,
            classes=num_classes,
            in_channels=3,
            # encoder_weights="imagenet"
        )

    def forward(self, x):
        x = self.model(x)
        return x
