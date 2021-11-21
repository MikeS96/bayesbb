import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Tuple

# File with distributions
import models.distributions as dist
from models.regression import BNNLinear

torch.autograd.set_detect_anomaly(True)
class BayesianMnist(nn.Module):
    def __init__(self, input_dim: int = 784, output_dim: int=10,   var_gauss: int = 0.2,
                 mixture_params: Tuple = (0, 6, 1 / 4), normal_params: float = -3,
                 mixture_prior: bool = True) -> None:
        """

        :param hidden_dim: Hidden dimension of the model
        :param var_gauss: Variance of the Normal likelihood
        :type mixture_params: Params for mixture of Gaussian's prior
        :type normal_params: Params for Normal prior
        :type mixture_prior: Bool to use mixture as prior or normal
        """
        super().__init__()
        self.l1 = BNNLinear(in_features=input_dim,
                            out_features=output_dim,  mixture_params=mixture_params,
                            normal_params=normal_params, mixture_prior=mixture_prior)
    @property
    def log_prior(self) -> torch.Tensor:
        """
        Compute the Log Prior probability of the model in one forward pass.
        :return: Log prior probability of the model in one forward pass
        """
        log_prior = 0
        # Sum the prior over all layers of the model
        for layer in self.children():
            log_prior += layer.log_prior
        return log_prior

    @property
    def log_posterior(self) -> torch.Tensor:
        """
        Compute the Log posterior probability of the model in one forward pass.
        :return: Log posterior probability of the model in one forward pass
        """
        log_posterior = 0
        # Sum the posterior over all layers of the model
        for layer in self.children():
            log_posterior += layer.log_posterior
        return log_posterior

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        :rtype: object
        :param x: Input vector
        :return: Output of the model
        """
        x = F.relu(self.l1(x))
        out = F.softmax(x, dim=1)
        return out

    def energy_loss(self, x: torch.Tensor, target: torch.Tensor,
                    num_classes: int, samples: int) -> torch.Tensor:
        """
        Compute the Energy loss for a given batch
        :param x: Input vector
        :param target: Targets vector
        :param samples: Number of samples
        :return: Loss for the current batch
        """
        # Initialization

        batch_size = target.shape[0]
        log_priors = torch.zeros(samples).to(x.device)
        log_posteriors = torch.zeros(samples).to(x.device)
        log_likelihoods = torch.zeros(samples, batch_size).to(x.device)
        for i in range(samples):
            outputs = self.forward(x)  # Forward pass
            log_priors[i] = self.log_prior  # Compute log prior of current forward pass
            log_posteriors[i] = self.log_posterior  # Compute log posterior of current forward pass
            log_likelihoods[i] = Categorical(outputs).log_prob(target)  # log-likelihood

        total_loss = log_posteriors.mean() - log_priors.mean() -  1/(batch_size)*log_likelihoods.mean(axis=0).sum()
        return total_loss
