import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch import nn


class MaskRCNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load a pre-trained Mask RCNN model
        model = maskrcnn_resnet50_fpn(pretrained=True)
        # Modify the final layer to match the number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = MaskRCNNPredictor(in_features, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)
