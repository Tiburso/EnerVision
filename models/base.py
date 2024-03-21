import pytorch_lightning as pl
from torch import nn
import torch


class BaseModel(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer, scheduler=None, metrics=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)

        print(y_hat.shape)
        print(y.shape)

        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss)

        return loss

    def validation(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss_fn(y_hat, y)

        self.log("val_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.validation(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.validation(batch, batch_idx)

    def configure_optimizers(self):
        return self.optimizer
