import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# File with distributions
import dist as dst


class BNNLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 mixture_params: Tuple = (0, 6, 1 / 4), normal_params: float = -3,
                 mixture_prior: bool = True) -> None:
        """
        Initialize Bayesian Linear Layer

        :param in_features: Input features of the Linear Layer
        :param out_features: Output features of the Linear Layer
        :type normal_params: Params for Normal prior
        :type mixture_params: Params for mixture of Gaussian's prior
        :type mixture_prior: Bool to use mixture as prior or normal
        """
        super().__init__()

        # Initialization
        self.in_features = in_features
        self.out_features = out_features

        # Initialize variational parameters for w
        self.w = dst.GaussianPosterior(in_features, out_features, bias=False)
        # Initialize variational parameters for b
        self.b = dst.GaussianPosterior(in_features, out_features, bias=True)
        # Initialize priors
        if mixture_prior:
            self.w_prior = dst.MixturePrior(mixture_params)
            self.b_prior = dst.MixturePrior(mixture_params)
        else:
            self.w_prior = dst.NormalPrior(normal_params)
            self.b_prior = dst.NormalPrior(normal_params)
        self.log_prior = 0
        self.log_posterior = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Bayesian Linear layer
        :param x: Input of the model
        :return: Output tensor given the samples parameters
        """
        # Sample weights and bias
        w = self.w.sample()
        b = self.b.sample()

        # ******************* #
        # Computing log prior #
        # ******************* #
        # Evaluate prior - weights and bias
        w_logp_prior = self.w_prior.log_prob(w)
        b_logp_prior = self.b_prior.log_prob(b)
        self.log_prior = w_logp_prior + b_logp_prior

        # *********************** #
        # Computing log posterior #
        # *********************** #
        w_logp_post = self.w.log_prob(w)
        b_logp_post = self.b.log_prob(b)
        self.log_posterior = w_logp_post + b_logp_post

        # Forward pass
        out = F.linear(x, w, b)
        return out


class BayesianRegressor(nn.Module):
    def __init__(self, hidden_dim: int = 64, var_gauss: int = 0.5,
                 mixture_params: Tuple = (0, 6, 1 / 4), normal_params: float = -3,
                 mixture_prior: bool = True) -> None:
        """

        :param hidden_dim: Hidden dimension of the model
        :param var_gauss:
        :param mixture_params:
        :param normal_params:
        :param mixture_prior:
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.var_gauss  = var_gauss
        self.l1 = BNNLinear(in_features=1, out_features=self.hidden_dim, mixture_params=mixture_params,
                            normal_params=normal_params, mixture_prior=mixture_prior)
        self.l2 = BNNLinear(in_features=self.hidden_dim, out_features=self.hidden_dim, mixture_params=mixture_params,
                            normal_params=normal_params, mixture_prior=mixture_prior)
        self.l3 = BNNLinear(in_features=self.hidden_dim, out_features=1, mixture_params=mixture_params,
                            normal_params=normal_params, mixture_prior=mixture_prior)
