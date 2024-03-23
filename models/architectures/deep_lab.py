import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from torch import nn


class DeepLabModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # Load a pre-trained DeepLab model
        self.model = deeplabv3_resnet50(
            pretrained=True, weights=DeepLabV3_ResNet50_Weights.DEFAULT
        )

        # Change the classifier to output the number of classes
        self.model.classifier[4] = nn.Conv2d(
            256, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        self.model.aux_classifier[4] = nn.Conv2d(
            256, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )

        self.warmup(input_size)

    def forward(self, x):
        x = self.model(x)["out"]
        # x = self.classifier(x)
        return x

    def warmup(self, input_size):
        self.eval()
        self(torch.randn(1, 3, input_size, input_size))

    def target(self, y):
        return y["masks"]
