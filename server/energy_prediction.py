import torch
from torch import nn
import pytorch_lightning as pl
import pandas as pd
import numpy as np


class EnergyPredictionModel(nn.Module):
    def __init__(self, dynamic_feature_size, static_feature_size, hidden_size):
        super(EnergyPredictionModel, self).__init__()
        self.dynamic_feature_size = dynamic_feature_size
        self.static_feature_size = static_feature_size
        self.hidden_size = hidden_size

        # LSTM module for dynamic features
        self.dynamic_rnn = nn.LSTM(
            input_size=dynamic_feature_size, hidden_size=hidden_size, batch_first=True
        )

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size + static_feature_size, hidden_size)
        self.relu = nn.ReLU()

        # Second Fully connected layer
        self.fc2 = nn.Linear(hidden_size, 3)

    def forward(self, dynamic_features, static_features):
        # Pass dynaamic features through LSTM
        _, (h_n, _) = self.dynamic_rnn(dynamic_features)
        h_n = h_n.squeeze(0)

        # Concatenate dynamic features with static features
        concatenated_features = torch.cat((h_n, static_features), dim=1)

        # Pass concatenated features through first fully connected layer
        x = self.relu(self.fc1(concatenated_features))

        # Pass through the second fully connected layer
        output = self.fc2(x)

        return output


class EnergyPredictionPL(pl.LightningModule):
    def __init__(
        self, dynamic_feature_size, static_feature_size, hidden_size, learning_rate
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = EnergyPredictionModel(
            dynamic_feature_size, static_feature_size, hidden_size
        )
        self.loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction="mean")
        self.train_losses = []
        self.validation_losses = []
        self.test_losses = []
        self.learning_rate = learning_rate

    def forward(self, x_dynamic, x_static):
        return self.model(x_dynamic, x_static)

    def training_step(self, batch, batch_idx):
        x_dynamic, x_static, y_true = batch
        y_pred = self(x_dynamic, x_static)

        loss = self.loss_fn(y_pred.float(), y_true.float())
        self.log("train_loss", loss)
        self.train_losses.append(loss.item())
        return loss


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the features so they are usable in the model

    Args:
        df (pd.DataFrame): The dataframe containing the features

    Returns:
        pd.DataFrame: The normalized dataframe
    """

    # Normalize static numeric features
    for feature in ["tilt", "azimuth"]:
        mean = df[feature].mean()
        std = df[feature].std()
        if std > 0:
            df[feature] = (df[feature] - mean) / std
        else:
            df[feature] = 0

    # Normalizing dynamic features
    dynamic_cols = [
        "temperature_sequence",
        "wind_speed_sequence",
        "dni_sequence",
        "dhi_sequence",
        "global_irradiance_sequence",
    ]

    for col in dynamic_cols:
        col_data = np.concatenate(df[col].values)
        mean = col_data.mean()
        std = col_data.std()
        if std > 0:
            df[col] = df[col].apply(lambda x: (np.array(x) - mean) / std)
        else:
            df[col] = df[col].apply(lambda x: np.zeros_like(x))

    return df
