import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn


class MaskRCNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load a pre-trained Mask RCNN model
        model = maskrcnn_resnet50_fpn(pretrained=True)

        # Modify the final layer to match the number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels=in_features_mask,
            dim_reduced=dim_reduced,
            num_classes=num_classes,
        )

        self.model = model

    def forward(self, x):
        out = self.model(x)[0]

        print(out)

        return out
