import numpy as np
import seaborn as sns
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
