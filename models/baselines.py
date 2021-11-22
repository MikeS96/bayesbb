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
