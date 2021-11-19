import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple

# File with distributions
import models.distributions as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BNNLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 mixture_params: Tuple = (0, 6, 1 / 4), normal_params: float = -3,
                 mixture_prior: bool = True) -> None:
        """
        Initialize Bayesian Linear Layer

        :param in_features: Input features of the Linear Layer
        :param out_features: Output features of the Linear Layer
        :type mixture_params: Params for mixture of Gaussian's prior
        :type normal_params: Params for Normal prior
        :type mixture_prior: Bool to use mixture as prior or normal
        """
        super().__init__()

        # Initialization
        self.in_features = in_features
        self.out_features = out_features

        # Initialize variational parameters for w
        self.w = dist.GaussianPosterior(in_features, out_features, bias=False)
        # Initialize variational parameters for b
        self.b = dist.GaussianPosterior(in_features, out_features, bias=True)
        # Initialize priors
        if mixture_prior:
            log_sigma1, log_sigma2, pi = mixture_params
            self.w_prior = dist.MixturePrior(log_sigma1, log_sigma2, pi)
            self.b_prior = dist.MixturePrior(log_sigma1, log_sigma2, pi)
        else:
            self.w_prior = dist.NormalPrior(normal_params)
            self.b_prior = dist.NormalPrior(normal_params)
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
    def __init__(self, hidden_dim: int = 64, var_gauss: int = 0.2,
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
        self.hidden_dim = hidden_dim
        self.var_gauss = var_gauss
        self.l1 = BNNLinear(in_features=1, out_features=self.hidden_dim, mixture_params=mixture_params,
                            normal_params=normal_params, mixture_prior=mixture_prior)
        self.l2 = BNNLinear(in_features=self.hidden_dim, out_features=1, mixture_params=mixture_params,
                            normal_params=normal_params, mixture_prior=mixture_prior)

    def log_prior(self) -> torch.Tensor:
        """
        Compute the Log Prior probability of the model in one forward pass.
        :return: Log prior probability of the model in one forward pass
        """
        log_prior = torch.zeros(1).to(device)
        # Sum the prior over all layers of the model
        for layer in self.children():
            log_prior += layer.log_prior
        return log_prior

    def log_posterior(self) -> torch.Tensor:
        """
        Compute the Log posterior probability of the model in one forward pass.
        :return: Log posterior probability of the model in one forward pass
        """
        log_posterior = torch.zeros(1).to(device)
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
        outputs = torch.zeros(samples, batch_size).to(device)
        log_priors = torch.zeros(samples).to(device)
        log_posteriors = torch.zeros(samples).to(device)
        log_likelihoods = torch.zeros(samples).to(device)
        for i in range(samples):
            outputs[i] = self.forward(x).view(-1)  # Forward pass
            log_priors[i] = self.log_prior()  # Compute log prior of current forward pass
            log_posteriors[i] = self.log_posterior()  # Compute log posterior of current forward pass
            log_likelihoods[i] = Normal(outputs[i], self.var_gauss).log_prob(target.view(-1)).sum()  # log-likelihood
        total_loss = log_posteriors.mean() - log_priors.mean() - log_likelihoods.mean()
        return total_loss
