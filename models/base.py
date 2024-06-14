import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryJaccardIndex, Dice, BinaryPrecision, BinaryRecall, PrecisionRecallCurve, BinarySpecificity, BinaryAccuracy


class BaseModel(pl.LightningModule):
    """
    Pytorch Lightning Base model class

    Args:
        model (torch.nn.Module): Model to be trained on.
        loss_fn (torch.nn.Module): Loss function to be used.
        optimizer (torch.optim.Optimizer: Optimizer to be used.
        scheduler (torch.optim.lr_scheduler, optional): _Learning rate scheduler to be used. Defaults to None.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.
        metrics (list, optional): Metrics to be logged. Defaults to None.
    """
    def __init__(
        self, model, loss_fn, optimizer, scheduler=None, threshold=0.5, metrics=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.treshold = threshold
        self.binary_jaccard_index = BinaryJaccardIndex()
        self.dice = Dice(multiclass=False)
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.specificity = BinarySpecificity()
        self.accuracy = BinaryAccuracy()

        self.save_hyperparameters(logger=False)
        self.save_hyperparameters(ignore=['model'])
        self.save_hyperparameters(ignore=['loss_fn'])

    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)
    
    def calculate_metrics(self, y_hat, y, prefix):
        # Calculate probabilities
        probs = torch.sigmoid(y_hat)
        # Classify pixel to 0 or 1
        y_hat = (probs > self.treshold).int()
        y = y.int()

        has_predicted = torch.any(y_hat == 1)
        has_actual = torch.any(y == 1)

        metrics = {}
        # Only calculate metrics that need positives when there are positives predicted or in ground truth
        if has_predicted or has_actual:
            metrics.update({
                f"{prefix}_dice": self.dice(y_hat, y),
                f"{prefix}_jaccard": self.binary_jaccard_index(y_hat, y),
                f"{prefix}_precision": self.precision(y_hat, y),
                f"{prefix}_recall": self.recall(y_hat, y),
            })
        metrics.update({
            f"{prefix}_specificity": self.specificity(y_hat, y),
            f"{prefix}_accuracy": self.accuracy(y_hat, y)
        })

        return metrics

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.calculate_loss(y_hat, y)
        self.log("train_loss", loss, sync_dist=True)

        if batch_idx % 250 == 0:
            metrics = self.calculate_metrics(y_hat, y, "train")
            self.log_dict(metrics, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.calculate_loss(y_hat, y)
        self.log("val_loss", loss, sync_dist=True)

        metrics = self.calculate_metrics(y_hat, y, "val")
        self.log_dict(metrics, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        if self.scheduler:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.scheduler,
                "monitor": "val_jaccard",
            }

        return self.optimizer