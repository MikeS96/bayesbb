import numpy as np

from typing import Tuple


def mean_var_predictive(y_hat: np.array, var_hat: np.array) -> Tuple:
    """
    Compute the predictive mean and variance of the Bayesian regressor
    :param y_hat: Predictions matrix over n samples of the bayesian model
    :param var_hat: Variance of the gaussian likelihood
    :return: Predictive mean and variance of the Bayesian regressor
    """
    # Computing mean of predictive distribution
    predictive_mean = np.mean(y_hat, axis=0)
    # Computing variance of predictive distribution
    if var_hat.size > 1:
        predictive_var = np.mean(y_hat * y_hat, axis=0) - predictive_mean ** 2 + np.mean(var_hat, axis=0)
    else:
        predictive_var = np.mean(y_hat * y_hat, axis=0) - predictive_mean ** 2 + var_hat
    return predictive_mean, predictive_var
