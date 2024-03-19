import pytorch_lightning as pl
from torch import nn
import torch


class BaseModel(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer, scheduler, metrics):
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


class ImageRecognitionModel(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(hidden_size * 2 * 8 * 8, hidden_size * 64),
            nn.ReLU(),
            nn.Linear(hidden_size * 64, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# model = ImageRecognitionModel(hidden_size=32, num_classes=10)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

# metrics = {"accuracy": nn.CrossEntropyLoss()}

# model = BaseModel(model, loss_fn, optimizer, scheduler, metrics)

# trainer = pl.Trainer(max_epochs=10)

# # Definer the loaders here
# trainer.fit(model, train_loader, val_loader)
