import torch
from torch.distributions import Normal
from torch.distributions.uniform import Uniform

from typing import Tuple


def experiment(xmin: float = 0.0, xmax: float = 0.5, samples: int = 50, std: float = 0.02, h: bool = False) -> Tuple:
    """
    Method to replicate the function used in Bayes By Backprop paper.

    :param xmin: Min value in X to generate data
    :param xmax: Max value in X to generate data
    :param samples: Number of samples to generate in the given interval
    :param std: Standars deviation of the Gaussian (error noise)
    :param h: Produce Heteroscedastic samples
    :return: Vectors with X and Y data
    """
    x = torch.linspace(xmin, xmax, samples).reshape(-1, 1)
    if std == 0:
        eps = torch.zeros(x.size())
    elif h:
        eps = torch.distributions.Normal(torch.zeros(samples), Uniform(1e-20, std).sample([samples])).sample([1])
        eps = eps.reshape(-1, 1)
    else:
        eps = Normal(0, std).sample(x.size())
    y = x + 0.3 * torch.sin(2 * torch.pi * (x + eps)) + 0.3 * torch.sin(4 * torch.pi * (x + eps)) + eps
    return x, y
