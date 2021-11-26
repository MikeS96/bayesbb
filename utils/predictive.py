import numpy as np
from scipy.stats import entropy

from typing import Tuple


def mean_var_predictive(y_hat: np.array, var_hat: np.array) -> Tuple:
    """
    Compute the predictive mean and variance of the Bayesian regression
    :param y_hat: Predictions matrix over n samples of the bayesian regression model
    :param var_hat: Variance of the gaussian likelihood
    :return: Predictive mean and variance of the Bayesian regression
    """
    # Computing mean of predictive distribution
    predictive_mean = np.mean(y_hat, axis=0)
    # Computing variance of predictive distribution
    if var_hat.size > 1:
        predictive_var = np.mean(y_hat * y_hat, axis=0) - predictive_mean ** 2 + np.mean(var_hat, axis=0)
    else:
        predictive_var = np.mean(y_hat * y_hat, axis=0) - predictive_mean ** 2 + var_hat
    return predictive_mean, predictive_var


def mean_entropy_predictive(p_hat: np.array) -> Tuple:
    """
    Compute the predictive mean and entropy of the classification model
    :param p_hat: Predictions matrix over n samples of the bayesian classification model
    :return: Predictive mean and entropy of the Bayesian classification
    """
    # Computing mean of predictive distribution
    predictive_mean = np.mean(p_hat, axis=0)
    # Computing variance (entropy) of predictive distribution [nats]
    predictive_entropy = entropy(p_hat, base=2, axis=0)

    return predictive_mean, predictive_entropy
