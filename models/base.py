import pytorch_lightning as pl
from torch import nn
import torch
from torchmetrics.functional.classification import (
    dice,
    binary_precision,
    binary_recall,
    binary_f1_score,
    binary_jaccard_index,
    multiclass_precision,
    multiclass_recall,
    multiclass_f1_score,
    multiclass_jaccard_index,
)


class BaseModel(pl.LightningModule):
    def __init__(
        self, model, loss_fn, optimizer, scheduler=None, treshold=0.5, metrics=None
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.treshold = treshold
        self.metrics = metrics

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, y_hat, y):
        y = self.model.target(y)

        return self.loss_fn(y_hat, y)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)

        loss = self.calculate_loss(y_hat, y)

        self.log("train_loss", loss)

        return loss

    def validation(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)

        loss = self.calculate_loss(y_hat, y)

        # Turn the predictions positive
        y_hat = self.model.target(y_hat)

        # metrics = {
        #     "val_loss": loss,
        #     "val_precision": binary_precision(y_hat, y, threshold=self.treshold),
        #     "val_recall": binary_recall(y_hat, y, threshold=self.treshold),
        #     "val_f1": binary_f1_score(y_hat, y, threshold=self.treshold),
        #     "val_dice": dice(y_hat, y.int(), threshold=self.treshold),
        # }

        metrics = {
            "val_loss": loss,
            "val_precision": multiclass_precision(y_hat, y, num_classes=2),
            "val_recall": multiclass_recall(y_hat, y, num_classes=2),
            "val_f1": multiclass_f1_score(y_hat, y, num_classes=2),
            "val_jaccard": multiclass_jaccard_index(y_hat, y, num_classes=2),
        }

        self.log_dict(metrics, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.validation(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.validation(batch, batch_idx)

    def configure_optimizers(self):
        if self.scheduler:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.scheduler,
                "monitor": "val_jaccard",
            }

        return self.optimizer
