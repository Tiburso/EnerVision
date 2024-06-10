import torch
from torch import nn
import pytorch_lightning as pl
import pandas as pd
import numpy as np


class EnergyPredictionModel(nn.Module):
    def _init_(
        self,
        dynamic_feature_size,
        static_feature_size,
        hidden_size,
        fc_size,
        dropout_rate=0.1,
        dataset_values=None,
    ):
        super(EnergyPredictionModel, self)._init_()
        self.dynamic_rnn1 = nn.LSTM(
            input_size=dynamic_feature_size, hidden_size=hidden_size, batch_first=True
        )
        # self.dynamic_rnn2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

        # Increased depth in fully connected layers
        self.fc1 = nn.Linear(hidden_size + static_feature_size, 2 * fc_size)
        self.fc2 = nn.Linear(2 * fc_size, fc_size)
        self.fc3 = nn.Linear(fc_size, 24)

        self.gelu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        # Load dataset values
        self.dynamic_cols = dynamic_feature_size

        # values for inference
        self.mean = dataset_values["mean"]
        self.std = dataset_values["std"]
        self.min = dataset_values["output_mins"]
        self.max = dataset_values["output_maxs"]

    def forward(self, dynamic_features, static_features):
        # Two LSTM layers
        output, (h_n, _) = self.dynamic_rnn1(dynamic_features)
        # output, (h_n, _) = self.dynamic_rnn2(output)
        h_n = h_n.squeeze(0)

        concatenated_features = torch.cat((h_n, static_features), dim=1)

        x = self.fc1(concatenated_features)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        output = self.sigmoid(x)

        return output

    def predict(self, x_dynamic, x_static):
        static_dim = x_static.shape[1] - 1
        print(static_dim)
        # Print input shapes for debugging
        print("input", x_dynamic.shape, x_static.shape)
        print("dynamic cols value", self.dynamic_cols)

        # Ensure x_static and x_dynamic are torch tensors and are of correct type
        x_static = torch.tensor(x_static, dtype=torch.float32)
        x_dynamic = torch.tensor(x_dynamic, dtype=torch.float32)

        # Print shapes after conversion to tensors
        print("shape x_static, x_dynamic", x_static.shape, x_dynamic.shape)

        # Normalize static features
        static_mean = torch.tensor(self.mean[:static_dim] + [0], dtype=torch.float32)
        static_std = torch.tensor(self.std[:static_dim] + [1], dtype=torch.float32)
        x_static_norm = (x_static - static_mean) / static_std

        # Print normalized static features
        print("x_static_norm", x_static_norm.shape, x_static_norm)

        # Normalize dynamic features
        dynamic_mean = torch.tensor(
            self.mean[static_dim:], dtype=torch.float32
        ).reshape(1, -1)
        dynamic_std = torch.tensor(self.std[static_dim:], dtype=torch.float32).reshape(
            1, -1
        )
        x_dynamic_norm = (x_dynamic - dynamic_mean) / dynamic_std

        # Print normalized dynamic features
        print("x_dynamic_norm", x_dynamic_norm.shape, x_dynamic_norm)

        # Switch model to evaluation mode
        self.eval()

        # Perform inference
        with torch.no_grad():
            normalized_output = self.forward(x_dynamic_norm, x_static_norm)

        # Denormalize the output if needed
        denormalized_output = normalized_output * (self.max - self.min) + self.min
        return denormalized_output

    def set_dataset_values(self, dataset_values):
        self.mean = dataset_values["mean"]
        self.std = dataset_values["std"]
        self.min = dataset_values["output_mins"]
        self.max = dataset_values["output_maxs"]


class TrainEnergyPrediction(pl.LightningModule):
    def _init_(
        self,
        dynamic_feature_size,
        static_feature_size,
        hidden_size,
        fc_size,
        learning_rate,
        loss_type="mse",
        dataset_values=None,
    ):
        super()._init_()
        self.save_hyperparameters()
        self.model = EnergyPredictionModel(
            dynamic_feature_size,
            static_feature_size,
            hidden_size,
            fc_size,
            dropout_rate=0.1,
            dataset_values=dataset_values,
        )

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="sum")
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction="sum")
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss(reduction="sum", delta=1.0)
        else:
            raise ValueError("Unsupported loss type. Choose from 'mse', 'l1', 'nll'.")

        self.train_losses = []
        self.validation_losses = []
        self.test_losses = []
        self.train_auc = []
        self.validation_auc = []
        self.train_r2 = []
        self.val_r2 = []
        self.learning_rate = learning_rate

    def forward(self, x_dynamic, x_static):
        return self.model(x_dynamic, x_static)

    def training_step(self, batch, batch_idx):
        x_dynamic, x_static, y_true = batch

        y_pred = self(x_dynamic, x_static)

        train_loss = self.loss_fn(y_pred, y_true)
        self.train_losses.append(train_loss.item())
        sum_loss = nn.L1Loss(reduction="sum")(y_pred, y_true)

        # Calculate AUC for predictions and ground truth
        auc_pred = self.calculate_auc(y_pred)
        auc_gt = self.calculate_auc(y_true)
        auc_ratio = auc_pred / auc_gt if auc_gt != 0 else 0

        self.train_auc.append(auc_ratio.item())

        # Calculate R^2 score
        r2 = self.r2_score(y_true, y_pred)
        self.train_r2.append(r2.item())

        # Log metrics
        self.log("train_auc", auc_ratio, on_step=True, on_epoch=True, logger=True)
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_l1_sum", sum_loss, on_step=False, on_epoch=True, logger=True)
        self.log("train_r2_score", r2, on_step=True, on_epoch=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        x_dynamic, x_static, y_true = batch
        y_pred = self(x_dynamic, x_static)
        val_loss = self.loss_fn(y_pred, y_true)
        self.validation_losses.append(val_loss.item())
        val_eval_metric = nn.L1Loss(reduction="mean")(y_pred, y_true)

        # Calculate AUC for predictions and ground truth
        auc_pred = self.calculate_auc(y_pred)
        auc_gt = self.calculate_auc(y_true)
        auc_ratio = auc_pred / auc_gt if auc_gt != 0 else 0

        self.validation_auc.append(auc_ratio.item())

        # Calculate R^2 score
        r2 = self.r2_score(y_true, y_pred)
        self.val_r2.append(r2.item())

        # Log metrics
        self.log("validation_auc", auc_ratio, on_step=True, on_epoch=True, logger=True)
        self.log("val_train_loss", val_loss, on_step=False, on_epoch=True, logger=True)
        self.log(
            "val_evaluation_metric",
            val_eval_metric,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log("val_r2_score", r2, on_step=True, on_epoch=True)

        return val_eval_metric

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x_dynamic, x_static, y_true = batch
        y_pred = self(x_dynamic, x_static)
        test_loss = self.loss_fn(y_pred, y_true)
        sum_loss = nn.L1Loss(reduction="sum")(y_pred, y_true)

        # Calculate AUC for predictions and ground truth
        auc_pred = self.calculate_auc(y_pred)
        auc_gt = self.calculate_auc(y_true)
        auc_ratio = auc_pred / auc_gt if auc_gt != 0 else 0

        # Calculate R^2 score
        r2 = self.r2_score(y_true, y_pred)

        # Log metrics
        self.log(f"Total test_loss", test_loss)
        self.log(f"Total L1 loss", sum_loss)
        self.log(f"Total auc_ratio", auc_ratio)
        self.log(
            "test_r2_score",
            r2,
        )
        self.test_losses.append(test_loss.item())

        return {
            "test_loss": test_loss,
            "sum_l1_loss": sum_loss,
            "auc_ratio": auc_ratio,
            "test_r2_score": r2,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def calculate_auc(self, y_values):
        y_values = y_values.detach().numpy()
        auc = np.trapz(y_values, axis=1)
        return np.mean(auc)

    def r2_score(self, y_true, y_pred):
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2
