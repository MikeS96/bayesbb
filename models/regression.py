import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple

from models.layers import BNNLinear


class BayesianRegressor(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, ll_var: float = 0.2,
                 mixture_params: Tuple = (0, 6, 1 / 4), normal_params: float = -3,
                 mixture_prior: bool = True) -> None:
        """

        :param input_dim: Input dimension of the model
        :param hidden_dim: Hidden dimension of the model
        :param ll_var: Variance of the Normal likelihood
        :type mixture_params: Params for mixture of Gaussian's prior
        :type normal_params: Params for Normal prior
        :type mixture_prior: Bool to use mixture as prior or normal
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ll_var = ll_var
        self.l1 = BNNLinear(in_features=self.input_dim, out_features=self.hidden_dim, mixture_params=mixture_params,
                            normal_params=normal_params, mixture_prior=mixture_prior)
        self.l2 = BNNLinear(in_features=self.hidden_dim, out_features=1, mixture_params=mixture_params,
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
        out = self.l2(x)
        return out

    def energy_loss(self, x: torch.Tensor, target: torch.Tensor, samples: int) -> torch.Tensor:
        """
        Compute the Energy loss for a given batch
        :param x: Input vector
        :param target: Targets vector
        :param samples: Number of samples
        :return: Loss for the current batch
        """
        # Initialization
        batch_size = target.shape[0]
        outputs = torch.zeros(samples, batch_size).to(x.device)
        log_priors = torch.zeros(samples).to(x.device)
        log_posteriors = torch.zeros(samples).to(x.device)
        log_likelihoods = torch.zeros(samples).to(x.device)
        for i in range(samples):
            outputs[i] = self.forward(x).view(-1)  # Forward pass
            log_priors[i] = self.log_prior  # Compute log prior of current forward pass
            log_posteriors[i] = self.log_posterior  # Compute log posterior of current forward pass
            log_likelihoods[i] = Normal(outputs[i], np.sqrt(self.ll_var)).log_prob(
                target.view(-1)).sum()  # log-likelihood
        total_loss = log_posteriors.mean() - log_priors.mean() - log_likelihoods.mean()
        return total_loss


class BayesianRegressorHeteroscedastic(nn.Module):
    def __init__(self, hidden_dim: int = 64, ll_var: float = 0.2,
                 mixture_params: Tuple = (0, 6, 1 / 4), normal_params: float = -3,
                 mixture_prior: bool = True) -> None:
        """

        :param hidden_dim: Hidden dimension of the model
        :param ll_var: Variance of the Normal likelihood
        :type mixture_params: Params for mixture of Gaussian's prior
        :type normal_params: Params for Normal prior
        :type mixture_prior: Bool to use mixture as prior or normal
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ll_var = ll_var
        self.l1 = BNNLinear(in_features=1, out_features=self.hidden_dim, mixture_params=mixture_params,
                            normal_params=normal_params, mixture_prior=mixture_prior)
        self.l2 = BNNLinear(in_features=self.hidden_dim, out_features=self.hidden_dim, mixture_params=mixture_params,
                            normal_params=normal_params, mixture_prior=mixture_prior)
        # Now one of the outputs is the log variance of the current input
        self.l3 = BNNLinear(in_features=self.hidden_dim, out_features=2, mixture_params=mixture_params,
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
        x = F.relu(self.l2(x))
        out = self.l3(x)
        return out

    def energy_loss(self, x: torch.Tensor, target: torch.Tensor, samples: int, warmup: bool = False) -> torch.Tensor:
        """
        Compute the Energy loss for a given batch
        :param x: Input vector
        :param target: Targets vector
        :param samples: Number of samples
        :param warmup: Parameter needed for mean warmup and prevent instability
        :return: Loss for the current batch
        """
        # Initialization
        batch_size = target.shape[0]
        # The output will not have mean and variance (Heteroscendastic noise)
        outputs = torch.zeros(samples, batch_size, 2).to(x.device)
        log_priors = torch.zeros(samples).to(x.device)
        log_posteriors = torch.zeros(samples).to(x.device)
        log_likelihoods = torch.zeros(samples).to(x.device)
        for i in range(samples):
            outputs[i] = self.forward(x)  # Forward pass
            log_priors[i] = self.log_prior  # Compute log prior of current forward pass
            log_posteriors[i] = self.log_posterior  # Compute log posterior of current forward pass
            # Computing log-likelihood
            if warmup:
                log_likelihoods[i] = Normal(outputs[i, :, 0],
                                            np.sqrt(self.ll_var)).log_prob(target.view(-1)).sum()
            else:
                log_likelihoods[i] = Normal(outputs[i, :, 0],
                                            torch.sqrt(torch.exp(outputs[i, :, 1]))).log_prob(target.view(-1)).sum()
        total_loss = log_posteriors.mean() - log_priors.mean() - log_likelihoods.mean()
        return total_loss
