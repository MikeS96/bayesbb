import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

import torch
import torch.optim as optim

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.observers import FileStorageObserver
from neptune.new.integrations.sacred import NeptuneObserver
import neptune.new as neptune

from models.regression import BayesianRegressor
from utils.experiments import experiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--observe", action="store_true", default=True)
args = parser.parse_args()
# Setting experiment
ex = Experiment('bayesian_regression')
# Set logging or not
if args.observe:
    token = os.environ.get("NEPTUNE_API_TOKEN")
    nep_run = neptune.init(api_token=token, project='mikes96/bbb')
    ex.observers.append(FileStorageObserver("sacred_files"))
    ex.observers.append(NeptuneObserver(run=nep_run))
    print("*****Observing runs*****")
else:
    print("*****Not oberving runs*****")


@ex.config
def config():
    epochs = 2000
    train_samples = 100
    x_min_train, x_max_train = 0.0, 0.5
    x_min_test, x_max_test = -0.2, 0.7
    elbo_samples = 6
    std = 0.02
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@ex.main
def train(epochs, train_samples, x_min_train, x_max_train, x_min_test, x_max_test, elbo_samples, std, device):
    x_train, y_train = experiment(xmin=x_min_train, xmax=x_max_train, samples=train_samples, std=std)
    _, y_train_true = experiment(xmin=x_min_train, xmax=x_min_train, samples=50, std=0)
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    model = BayesianRegressor(hidden_dim=128, ll_var=0.05, mixture_prior=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):  # loop over the dataset multiple times
        optimizer.zero_grad()
        # forward + backward + optimize
        loss = model.energy_loss(x_train, y_train, elbo_samples)
        loss.backward()
        optimizer.step()
        ex.log_scalar('training/epoch', epoch + 1)
        ex.log_scalar('training/loss', loss.item())

        if epoch % 100 == 0:
            print('Current epoch: {}/{}'.format(epoch + 1, epochs))
            print('Current Loss:', loss.item())
    print('Finished Training')


ex.run()
nep_run.stop()
