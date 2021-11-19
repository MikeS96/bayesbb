import numpy as np

import torch
from torch.distributions import Normal
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaussianPosterior(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        """
        Initialize variational parameters of variational posterior distribution

        :param in_features: Input features of the Linear Layer
        :param out_features: Output features of the Linear Layer
        :param bias: Boolean to initialize vector of weights or matrix of weights
        """
        super().__init__()

        # Initialize parameters uniformly within a range
        if bias:
            # mu ~ U(-0.5, 0.5)
            self.mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.5, 0.5), requires_grad=True)
            # rho = log(exp(sigma) - 1) -- Init sigma ~ U(0.001, 1) or rho ~ U(-7, -4.5)
            self.rho = nn.Parameter(torch.Tensor(out_features).uniform_(-7, -4.5), requires_grad=True)
        else:
            # mu ~ U(-0.5, 0.5)
            self.mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.5, 0.5), requires_grad=True)
            # rho = log(exp(sigma) - 1) -- Init sigma ~ U(0.001, 1) or rho ~ U(-7, -4.5)
            self.rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-7, -4.5), requires_grad=True)

        # eps ~ N(0, 1)
        self.normal = Normal(0, 1)

    @property
    def rho2sigma(self) -> torch.Tensor:
        """
        Transform the variational parameter rho to sigma
        :return: Tensor of sigmas
        """
        sigma = torch.log(1 + torch.exp(self.rho))
        return sigma

    def sample(self) -> torch.Tensor:
        """
        Sample parameters from variational posterior using re-parameterization trick.

        :return: Tensor of Sampled parameters from variational posterior
        """
        # Sample parameter-free noise epsilon
        eps = self.normal.sample(self.mu.size()).to(device)
        param = self.mu + self.rho2sigma * eps
        return param

    def log_prob(self, w: torch.Tensor) -> torch.Tensor:
        """
         Compute the log probability of the weights given the gaussian posterior

        :param w: Parameters samples from the variational posterior
        :return: Log probability of given parameters in the variational posterior
        """
        # sum_w q(w | theta)
        logp_var_post = Normal(self.mu, self.rho2sigma).log_prob(w).sum()
        return logp_var_post


class MixturePrior:
    def __init__(self, log_sigma1: float = 0, log_sigma2: float = 6, pi: float = 1 / 4) -> None:
        """
        :param log_sigma1: -log(sigma1) parameter for 1 gaussian component
        :param log_sigma2: -log(sigma2) parameter for 2 gaussian component
        :param pi: Mix probability in mixture
        """
        self.sigma1 = torch.FloatTensor([np.exp(-log_sigma1)]).to(device)
        self.sigma2 = torch.FloatTensor([np.exp(-log_sigma2)]).to(device)
        self.pi = pi
        # Define mixture components
        self.normal1 = Normal(0, self.sigma1)
        self.normal2 = Normal(0, self.sigma2)

    def log_prob(self, w: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of the weights given a Mixture of normal priors.

        :param w: Set of weights drawn out of posterior distribution
        :return: log probability of given set of weights over mixture prior
        """
        p_prior_1 = torch.exp(self.normal1.log_prob(w))
        p_prior_2 = torch.exp(self.normal2.log_prob(w))
        logp_prior = torch.log(self.pi * p_prior_1 + (1 - self.pi) * p_prior_2).sum()
        return logp_prior


class NormalPrior:
    def __init__(self, log_sigma: float = -3) -> None:
        """
        :param log_sigma: -log(sigma) parameter for gaussian
        """
        # -log_simga = -3 -> sigma = 20
        self.sigma = torch.FloatTensor([np.exp(-log_sigma)]).to(device)
        # Define normal distribution
        self.normal = Normal(0, self.sigma)

    def log_prob(self, w: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of the weights given a normal prior.

        :param w: Set of weights drawn out of posterior distribution
        :return: log probability of given set of weights over gaussian prior
        """
        logp_prior = self.normal.log_prob(w).sum()
        return logp_prior
