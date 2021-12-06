import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineLinearRegressor(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, output_dim: int = 1):
        """
        Just initialize a simple linear regressor
        """
        super(BaselineLinearRegressor, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        out = self.l2(x)
        return out


class BaselineLinRegressorWithDropout(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128,
                 output_dim: int = 1, dropout_p: float = 0.25):
        """
        Just initialize a simple linear regressor with dropout
        """
        super(BaselineLinRegressorWithDropout, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = F.relu(self.dropout(self.l1(x)))
        out = self.l2(x)
        return out


class BaselineEnsembleMethodRegression(nn.Module):
    def __init__(self, device, num_models: int = 10) -> None:
        """
        :param num_models: number of models to use in the ensemble
        """
        super(BaselineEnsembleMethodRegression, self).__init__()
        self.num_models = num_models
        self.device = device
        self.models = [BaselineLinearRegressor().to(device)
                       for i in range(self.num_models)]

    def forward(self, x):
        out = torch.cat([model.forward(x) for model in self.models], dim=1)
        out = torch.mean(out, dim=1)
        return out


class BaselineMnist(nn.Module):
    def __init__(self, input_dim: int = 784, output_dim: int = 10,
                 hidden_dim: int = 1200) -> None:
        """
        :param input_dim: input dimension of the model (flattened image)
        :param ouput_dim: output dimension of the model (num classes)
        :param hidden_dim: Hidden dimension of the model
        """
        super().__init__()
        self.l1 = torch.nn.Linear(in_features=input_dim,
                                  out_features=hidden_dim)
        self.l2 = torch.nn.Linear(in_features=hidden_dim,
                                  out_features=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        :rtype: object
        :param x: Input vector
        :return: Output of the model
        """
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        out = F.softmax(x, dim=1)
        return out
