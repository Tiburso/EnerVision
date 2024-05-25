import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import (
    DiceLoss,
    JaccardLoss,
    FocalLoss
)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = LossCE()
        self.jaccard_loss = LossJaccard()
        self.ce_loss = LossCE()

    def forward(self, y_hat, y):
        return (self.dice_loss + self.jaccard_loss + self.ce_loss) / 3.0


class LossCE(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, y_hat, y):
        y = y.squeeze(1).long()
        return self.loss(y_hat, y) 


class LossDice(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = DiceLoss(mode="multiclass", classes=2, from_logits=True)

    def forward(self, y_hat, y):
        y = y.long()
        return self.loss(y_hat, y)


class LossJaccard(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = JaccardLoss(mode="multiclass", classes=2, from_logits=True)

    def forward(self, y_hat, y):
        y = y.long()
        return self.loss(y_hat, y)
    

class LossFocal(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = FocalLoss(mode="multiclass", alpha=0.25)

    def forward(self, y_hat, y):
        y = y.squeeze(1).long()
        return self.loss(y_hat, y)
    
