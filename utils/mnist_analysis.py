import os
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from models.mnist import BayesianMnist
from utils.vis import signal_noise, plot_hist


train_data = torchvision.datasets.MNIST(root="~/data", train=True,
                                        download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor()]))

test_data = torchvision.datasets.MNIST(root="~/data", train=False,
                                       download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor()]))
#subset = list(range(0, 100))
#test_data = torch.utils.data.Subset(test_data, subset)

batch_size = 100
elbo_samples = 10
num_classes = 10
cuda = True
device = torch.device("cuda" if cuda else "cpu")
loader_test = DataLoader(test_data, batch_size,
                         shuffle=True, drop_last=True)

def load_checkpoint(checkpoint_path, device):
    model = BayesianMnist(28 * 28).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    return model

def test_model(model, loader_test, device, num_classes, elbo_samples, batch_size):
    num_test_batches = len(loader_test)
    total_test_accuracy = 0
    total_entropy = 0 
    for x_test, y_test in loader_test:
        x_test = x_test.reshape(batch_size, -1).to(device)
        y_test = y_test.to(device)
        test_loss, _ = model.energy_loss(
            x_test, y_test, num_test_batches, num_classes, elbo_samples)
        test_loss += test_loss.item()
        softmax_averaged, entropy = model.inference(x_test, 10, elbo_samples, batch_size)
        pred = softmax_averaged.argmax(axis=1)
        accuracy = sum(pred == y_test.to("cpu").numpy())
        total_test_accuracy += (accuracy / batch_size)
        total_entropy += np.sum(entropy)
    average_test_accuracy = total_test_accuracy / num_test_batches
    print("average test accuracy for checkpoint: {}".format(average_test_accuracy))

def remove_percentage_weights(percentage, weights_ratio, biases_ratio, model):
    value_weights = np.percentile(weights_ratio, percentage)
    value_biases = np.percentile(biases_ratio, percentage)
    for layer in model.children():

        # grabbing mu and rho for weights and biases
        weights_mu = layer.w.mu.cpu().detach()
        weights_rho = layer.w.rho.cpu().detach()
        biases_mu = layer.b.mu.cpu().detach()
        biases_rho = layer.b.rho.cpu().detach()
        # convert rhos to sigmas
        weights_sigma = np.log(np.exp(weights_rho) + 1)
        biases_sigma = np.log(np.exp(biases_rho) + 1)
        # get signal noise ratio for weights and biases
        ratio_weights = np.divide(np.abs(weights_mu), weights_sigma)
        ratio_biases = np.divide(np.abs(biases_mu), biases_sigma)
        # bool for weights and biases we will set to zero
        to_keep_weights_mu = ratio_weights > value_weights
        # if ratio is large, signal is large, so we keep 
        to_keep_biases_mu = ratio_biases > value_weights
        to_keep_weights_rho = weights_sigma > value_biases
        to_keep_biases_rho = biases_sigma > value_biases
        # 0 for remove, 1 for keep
        # set mus to zeros
        # set sigmas to zeros
        layer.w.mu = torch.nn.Parameter(np.multiply(to_keep_weights_mu, weights_mu))
        layer.w.rho = torch.nn.Parameter(np.multiply(to_keep_weights_rho, weights_rho))
        layer.b.mu = torch.nn.Parameter(np.multiply(to_keep_biases_mu, biases_mu))
        layer.b.rho = torch.nn.Parameter(np.multiply(to_keep_biases_rho, biases_rho))
    return model

        
def model_param_hist_plot(device, checkpoint_file):
    model = load_checkpoint(checkpoint_file, device)
    test_model(model, loader_test, device, num_classes, elbo_samples, batch_size)
    weights_ratio, biases_ratio = signal_noise(model)
    weights_to_plot = 10*np.log10(weights_ratio)
    biases_to_plot = 10*np.log10(biases_ratio)
    plot_hist(weights_to_plot)
    plot_hist(biases_to_plot)


# code to remove weights and test on test data
model = load_checkpoint("gaussian_prior.checkpoint", device)
weights_ratio, biases_ratio = signal_noise(model)
percentage_to_remove = 99
model_pruned = remove_percentage_weights(percentage_to_remove, weights_ratio, biases_ratio,  model)
model_pruned.to(device)
test_model(model_pruned, loader_test, device, num_classes, elbo_samples, batch_size)

plot_weights = False
if plot_weights:
    model_param_hist_plot("gaussian_prior.checkpoint", device)
