import pytorch_lightning as pl
from torch import nn
import torch


class BaseCNN(nn.Module):
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
