import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torch import nn


class DeepLabModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # Load a pre-trained DeepLab model
        model = deeplabv3_resnet50(pretrained=True)

        # Modify the first layer to accept the number of channels in the input
        model.backbone.conv1 = nn.Conv2d(
            input_size[0],
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

        # Modify the final layer to match the number of classes
        model.classifier[4] = nn.Conv2d(
            256, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]
