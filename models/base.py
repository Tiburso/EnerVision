import pytorch_lightning as pl
from torch import nn
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, JaccardIndex


class BaseModel(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer, scheduler=None, metrics=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics

        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")
        self.jaccard = JaccardIndex(task="binary")

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

        y_hat = torch.sigmoid(y_hat)
        y_hat = (y_hat > 0.5).float()

        metrics = {
            "val_loss": loss,
            "val_accuracy": self.accuracy(y_hat, y),
            "val_precision": self.precision(y_hat, y),
            "val_recall": self.recall(y_hat, y),
            "val_f1": self.f1(y_hat, y),
            "val_jaccard": self.jaccard(y_hat, y),
        }

        self.log_dict(metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.validation(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.validation(batch, batch_idx)

    def configure_optimizers(self):
        return self.optimizer
