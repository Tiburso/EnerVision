import pytorch_lightning as pl
from torchmetrics.functional.classification import (
    accuracy,
    precision,
    recall,
    f1_score,
    jaccard_index,
    dice,
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

        self.save_hyperparameters(logger=False)

    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y = self.model.target(y)

        y_hat = self.forward(X)

        loss = self.calculate_loss(y_hat, y)
        self.log("train_loss", loss, sync_dist=True)

        if batch_idx % 1000 == 0:
            # Reconvert into a single channel
            y = y.argmax(dim=1)

            metrics = {
                "train_jaccard": jaccard_index(
                    y_hat, y, task="multiclass", num_classes=2
                ),
                "train_dice": dice(y_hat, y.int()),
            }

            self.log_dict(metrics, sync_dist=True)

        return loss

    def validation(self, batch, batch_idx):
        X, y = batch
        y = self.model.target(y)

        y_hat = self.forward(X)

        loss = self.calculate_loss(y_hat, y)

        # Reconvert into a single channel
        y = y.argmax(dim=1)

        metrics = {
            "testing_loss": loss,
            "jaccard_index": jaccard_index(y_hat, y, task="multiclass", num_classes=2),
            "accuracy": accuracy(y_hat, y, task="multiclass", num_classes=2),
            "precision": precision(y_hat, y, task="multiclass", num_classes=2),
            "recall": recall(y_hat, y, task="multiclass", num_classes=2),
            "f1_score": f1_score(y_hat, y, task="multiclass", num_classes=2),
            "dice": dice(y_hat, y.int()),
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
                "monitor": "jaccard_index",
            }

        return self.optimizer
