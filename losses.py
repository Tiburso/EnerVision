import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import (
    DiceLoss,
    JaccardLoss,
    FocalLoss,
)


class LossCombined(nn.Module):
    """Class for the loss function that combines the jacard and binary cross entropy loss."""
    def __init__(self):
        super().__init__()
        self.jaccard_loss = LossJaccard()
        self.ce_loss = LossCE()

    def forward(self, y_hat, y):
        jaccard_loss = self.jaccard_loss(y_hat, y)
        ce_loss = self.ce_loss(y_hat, y)
        return (jaccard_loss + ce_loss) / 2.0


class LossCE(nn.Module):
    """Class for the binary cross entropy loss function."""
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, y_hat, y):
        y = y.float()
        return self.loss(y_hat, y) 


class LossDice(nn.Module):
    """Class for the dice loss function."""
    def __init__(self):
        super().__init__()
        self.loss = DiceLoss(mode="binary", from_logits=True)

    def forward(self, y_hat, y):
        y = y.long()
        return self.loss(y_hat, y)


class LossJaccard(nn.Module):
    """Class for the jacard loss function."""
    def __init__(self):
        super().__init__()
        self.loss = JaccardLoss(mode="binary", from_logits=True)

    def forward(self, y_hat, y):
        y = y.long()

        y_prob = torch.sigmoid(y_hat)
        y_pred = (y_prob > 0.5).int()
        y_ground = y.int()

        if torch.all(y_pred == 0) and torch.all(y_ground == 0):
            return torch.tensor(0.0)
        else:
            return self.loss(y_hat, y)
    

class LossFocal(nn.Module):
    """Class for the focal loss function."""
    def __init__(self):
        super().__init__()
        self.loss = FocalLoss(mode="binary", alpha=0.25)

    def forward(self, y_hat, y):
        y = y.long()
        return self.loss(y_hat, y)
    
