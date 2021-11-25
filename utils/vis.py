import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from typing import Tuple

plt.style.use('seaborn-white')


def weights_histogram(model) -> Tuple:
    """

    :param model: Bayesian torch model
    :return: List with weights and biases of the model
    """
    # Initialize variables to store weights and biases of the model
    weights = []  # Only weights
    biases = []  # Only biases
    params = []  # Weights and biases
    for layer in model.children():
        # Samples weights from layers
        w = layer.w.sample().view(-1).cpu().detach().tolist()
        b = layer.b.sample().view(-1).cpu().detach().tolist()
        # Append weights to lists
        weights.extend(w)
        biases.extend(b)
        params.extend(w + b)
    fig, axs = plt.subplots(ncols=3, figsize=(8, 6))
    sns.histplot(data=np.array(weights), kde=True, ax=axs[0]).set(title='Histogram of Weights')
    sns.histplot(data=np.array(biases), kde=True, ax=axs[1]).set(title='Histogram of Biases')
    sns.histplot(data=np.array(params), kde=True, ax=axs[2]).set(title='Histogram of Parameters')
    plt.show()

    return weights, biases


def visualize_training(x_train: np.array, y_train: np.array, y_train_line: np.array) -> matplotlib.figure.Figure:
    """
    Visualize training data
    :param x_train: X samples of data
    :param y_train: Y samples of noisy data
    :param y_train_line: Ground truth Y samples without noise
    :return: Matplotlib figure
    """
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x_train, y_train_line, label='True function')
    plt.scatter(x_train, y_train, label='Noisy data points', marker='*', color='r')
    plt.title('Training data with {} samples'.format(x_train.shape[0]))
    plt.legend()
    plt.show()

    return fig


def visualize_inference(x_train: np.array, y_train: np.array,
                        x_test: np.array, y_test_true: np.array,
                        predictive_mean: np.array, predictive_var: np.array) -> matplotlib.figure.Figure:
    """
    Predictive mean and error of the bayesian regression
    :param x_train: X Training data
    :param y_train: Noisy Y target data
    :param x_test: X test data
    :param y_test_true: Ground truth Y without noise
    :param predictive_mean: Predictive mean of the model
    :param predictive_var: Predictive variance of the model
    :return: Matplotlib figure
    """
    # Compute stdv of samples
    predictive_std = np.sqrt(predictive_var)
    fig = plt.figure(figsize=(8, 6))
    # Plot original training samples
    plt.scatter(x_train, y_train, color='k', marker='*', label='Training data')
    # Plot predictive mean of the model
    plt.plot(x_test, predictive_mean, label='Mean Posterior Predictive')
    # Plot two standard deviations of the predictive stdv
    plt.fill_between(x_test.reshape(-1), predictive_mean + 2 * predictive_std, predictive_mean - 2 * predictive_std,
                     alpha=0.25, label='2 Stdv')
    # plt.fill_between(x_test.cpu().detach().numpy().reshape(-1), np.percentile(y_samp, 2.5, axis=0),
    #                  np.percentile(y_samp, 97.5, axis=0),
    #                  alpha=0.25, label='95% Confidence')
    # Ground truth
    plt.plot(x_test, y_test_true, label='Ground Truth')
    plt.legend()
    plt.title('Posterior Predictive mean with $2\sigma$')
    plt.show()

    return fig
