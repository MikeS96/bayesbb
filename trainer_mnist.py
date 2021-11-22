# run as python trainer_minist.py without sacred logging
# run as python trainer_mnist.py -m <sacred_file_name> with sacred logging
import os

import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.observers import FileStorageObserver
from neptunecontrib.monitoring.sacred import NeptuneObserver
import neptune.new as neptune

from models.mnist import BayesianMnist

#token = os.environ.get("NEPTUNE_API_TOKEN")
#nep_run = neptune.init(api_token=token, project='bbb')



ex = Experiment()
ex.observers.append(FileStorageObserver)
#ex.observers.append(NeptuneObserver("bbb"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using")
print(device)
train_data = torchvision.datasets.MNIST(root="~/data", train=True,
                                        download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor()]
                                        ))

test_data = torchvision.datasets.MNIST(root="~/data", train=False,
                                       download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor()]
                                       ))

@ex.config
def config():
    num_epochs = 10
    batch_size = 28
    elbo_samples = 4
    cuda = False

@ex.automain
def train(elbo_samples, batch_size, num_epochs, cuda):
    device = torch.device("cuda" if cuda else "cpu")
    model = BayesianMnist(28*28).to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.01)
    num_classes = 10
    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
    num_train_batches = len(loader_train)
    num_test_batches = len(loader_test)

    for epoch in range(num_epochs):
        batch_number = 0
        epoch_loss = 0
        for x_train, y_train in loader_train:
            x_train = x_train.reshape(batch_size, -1).to(device)
            y_train = y_train.to(device)        
            optimiser.zero_grad()
            loss, accuracy = model.energy_loss(
                x_train, y_train, num_train_batches, num_classes, elbo_samples)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            batch_number += 1
            if batch_number % 400 == 0:
                print("batch number: {} batch TRAIN loss: {} ".format(batch_number, loss.item()))
                print("batch number: {} batch TRAIN accuracy: {} ".format(batch_number, accuracy))
                ex.log_scalar("batch train loss", loss.item())
                ex.log_scalar("batch train accuracy", accuracy)
                total_test_loss = 0
                for x_test, y_test in loader_test:
                    x_test = x_test.reshape(batch_size, -1).to(device)
                    y_test = y_test.to(device)
                    batch_test_loss, batch_test_accuracy = model.energy_loss(
                        x_test, y_test, num_test_batches, num_classes, elbo_samples)
                    total_test_loss += batch_test_loss
                print(
                    "batch number {} batch TEST loss: {} ".format(batch_number, batch_test_loss))
                print(
                    "batch number {} batch TEST acc: {} ".format(batch_number, batch_test_accuracy))
                ex.log_scalar("batch test loss", loss.item())
                ex.log_scalar("batch test accuracy", accuracy)

        print("total loss for epoch {} : {} ".format(epoch, epoch_loss))


train(num_epochs=10, batch_size=28,  elbo_samples=3, cuda=True)
