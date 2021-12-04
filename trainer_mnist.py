# run as python trainer_minist.py with "observe=False" without sacred logging
# run as python trainer_mnist.py with sacred / neptune logging
# write over config parameters using the command line ie python trainer_mnist.py with "elbo_samples=10"

import os
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.observers import FileStorageObserver
from neptune.new.integrations.sacred import NeptuneObserver
import neptune.new as neptune

from models.mnist import BayesianMnist

ex = Experiment()

train_data = torchvision.datasets.MNIST(root="~/data", train=True,
                                        download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor()]))

test_data = torchvision.datasets.MNIST(root="~/data", train=False,
                                       download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor()]))
#subset = list(range(0, 100))
#train_data = torch.utils.data.Subset(train_data, subset)
#test_data = torch.utils.data.Subset(test_data, subset)


@ex.config
def config():
    observe = True
    if observe:
        token = os.environ.get("NEPTUNE_API_TOKEN")
        nep_run = neptune.init(api_token=token, project='bbb')
        ex.observers.append(FileStorageObserver("sacred_files"))
        ex.observers.append(NeptuneObserver(run=nep_run))
        print("*****Observing runs*****")
    else:
        print("*****Not oberving runs*****")
    num_epochs = 100
    batch_size = 100
    elbo_samples = 4
    cuda = False

@ex.automain
def train(elbo_samples: int, batch_size: int, num_epochs: int,
          cuda: bool, mixture_prior: bool, checkpoint_name: str = None):
    device = torch.device("cuda" if cuda else "cpu")
    model = BayesianMnist(28 * 28, mixture_prior=mixture_prior).to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.01)
    num_classes = 10
    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
    num_train_batches = len(loader_train)
    num_test_batches = len(loader_test)
    for epoch in range(num_epochs):
        train_loss = 0
        total_accuracy = 0
        num_train_batches = len(loader_train)
        for x_train, y_train in loader_train:
            x_train = x_train.reshape(batch_size, -1).to(device)
            y_train = y_train.to(device)
            optimiser.zero_grad()
            loss, accuracy = model.energy_loss(
                x_train, y_train, num_train_batches, num_classes, elbo_samples)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
            total_accuracy += accuracy
        epoch_accuracy = total_accuracy / num_train_batches
        print("epoch number: {} TRAIN loss: {} ".format(epoch, train_loss))
        print("epoch number: {} TRAIN accuracy: {} ".format(epoch, epoch_accuracy))
        ex.log_scalar("train loss", loss.item())
        ex.log_scalar("train accuracy", epoch_accuracy)
        test_loss = 0
        total_test_accuracy = 0
        total_entropy = 0
        num_test_batches = len(loader_test)
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
        average_test_acc = total_test_accuracy/num_test_batches
        
        print(
            "epoch number {} TEST loss: {} ".format(epoch, test_loss))
        print(
            "epoch number {} TEST acc: {} ".format(
                epoch, average_test_acc.item()))
        print("epoch number {} entropy {}".format(epoch, total_entropy))
        ex.log_scalar("test loss", test_loss.item())
        ex.log_scalar("test accuracy", average_test_acc)
        ex.log_scalar("test entropy", total_entropy )
        if checkpoint_name is not None:
            torch.save(model.state_dict(), checkpoint_name + ".checkpoint")
        
nep_run.stop()
