from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import pytorch_lightning as pl
import torch.nn as nn
import torch

class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        return self.model(x)['out'] # Output is [batch_size, num_classes, height, width]