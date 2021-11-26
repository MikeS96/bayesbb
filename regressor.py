import numpy as np
import os

import torch
import torch.optim as optim

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.observers import FileStorageObserver
from neptune.new.integrations.sacred import NeptuneObserver
import neptune.new as neptune

from models.regression import BayesianRegressor
from utils.experiments import experiment
from utils.vis import visualize_training, weights_histogram, visualize_inference
from utils.predictive import mean_var_predictive

# Declare experiment
ex = Experiment('bayesian_regression')


@ex.config
def config():
    observe = True
    if observe:
        token = os.environ.get("NEPTUNE_API_TOKEN")
        nep_run = neptune.init(api_token=token, project="pgm/BayesianRegressor")
        ex.observers.append(FileStorageObserver("sacred_files"))
        ex.observers.append(NeptuneObserver(run=nep_run))
        print("*****Observing runs*****")
    else:
        print("*****Not oberving runs*****")
    epochs = 2000
    train_samples = 50
    x_min_train, x_max_train = 0.0, 0.5
    x_min_test, x_max_test = -0.2, 0.8
    elbo_samples = 6
    std = 0.02
    ll_var = 0.05 ** 2
    mixture_prior = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@ex.main
def train(epochs, train_samples, x_min_train, x_max_train,
          x_min_test, x_max_test, elbo_samples, std, ll_var, mixture_prior, device):
    # Training bayesian regressor (Homoscedastic)
    x_train, y_train = experiment(xmin=x_min_train, xmax=x_max_train, samples=train_samples, std=std)
    _, y_train_true = experiment(xmin=x_min_train, xmax=x_max_train, samples=train_samples, std=0)
    # Visualizing data sample
    _ = visualize_training(x_train.numpy(), y_train.numpy(), y_train_true.numpy())
    # ex.log_scalar('training/data_x', x_train.tolist())
    # ex.log_scalar('training/data_y', y_train.tolist())
    # ex.log_scalar('training/data_y_true', y_train_true.tolist())
    # Logging training Figure
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    model = BayesianRegressor(hidden_dim=128, ll_var=ll_var, mixture_prior=mixture_prior).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):  # loop over the dataset multiple times
        optimizer.zero_grad()
        # forward + backward + optimize
        loss = model.energy_loss(x_train, y_train, elbo_samples)
        loss.backward()
        optimizer.step()
        ex.log_scalar('training/loss', loss.item())

        if epoch % 100 == 0:
            print('Current epoch: {}/{}'.format(epoch + 1, epochs))
            print('Current Loss:', loss.item())
    print('Finished Training')

    # Prediction step
    test_samples = 500
    exp_samples = 100
    x_test, y_test = experiment(xmin=x_min_test, xmax=x_max_test, samples=test_samples, std=0.02)
    _, y_test_true = experiment(xmin=x_min_test, xmax=x_max_test, samples=test_samples, std=0)
    y_samp = np.zeros((exp_samples, test_samples))
    # Testing experiment
    x_test = x_test.to(device)
    for s in range(exp_samples):
        y_hat = model.forward(x_test).cpu().detach().numpy()
        y_samp[s, :] = y_hat.reshape(-1)

    # Computing mean and variance of predictive distribution
    predictive_mean, predictive_var = mean_var_predictive(y_samp, np.array(ll_var))
    _ = visualize_inference(x_train.cpu().detach().numpy(), y_train.cpu().detach().numpy(),
                            x_test.cpu().detach().numpy(), y_test_true.cpu().detach().numpy(),
                            predictive_mean, predictive_var)
    # Weights histograms
    _, _ = weights_histogram(model)
    return predictive_mean, predictive_var


ex.run_commandline()
nep_run.stop()
