import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from models.mnist import BayesianMnist
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
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

model = BayesianMnist(28*28).to(device)
optimiser = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 10
batch_size = 28
elbo_samples = 3
num_classes = 10
loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

for epoch in range(num_epochs):
    batch_number = 0
    for x_train, y_train in loader_train:
        x_train = x_train.reshape(batch_size, -1).to(device)
        y_train = y_train.to(device)        
        optimiser.zero_grad()
        loss = model.energy_loss(x_train, y_train, num_classes, elbo_samples)
        loss.backward()
        optimiser.step()
        if batch_number % 200 == 0:
            print(loss.item())
        batch_number += 1
