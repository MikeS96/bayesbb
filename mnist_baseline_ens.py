# run as python trainer_minist.py with "observe=False" without sacred logging
# run as python trainer_mnist.py with sacred / neptune logging
# write over config parameters using the command line ie python trainer_mnist.py with "elbo_samples=10"

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from neptune.new.integrations.sacred import NeptuneObserver
import neptune.new as neptune
import ddu_dirty_mnist

from models.mnist import BayesianMnist
from models.baselines import BaselineMnist
from models.baselines import BaselineEnsembleMnist
from models.baselines import BaselineMnistWithDropout

ex = Experiment()

@ex.config
def config():
    observe = True
    if observe:
        token = os.environ.get("NEPTUNE_API_TOKEN")
        nep_run = neptune.init(api_token=token, project="pgm/mnistbl")
        ex.observers.append(FileStorageObserver("sacred_files"))
        ex.observers.append(NeptuneObserver(run=nep_run))
        print("*****Observing runs*****")
    else:
        print("*****Not oberving runs*****")
    num_epochs = 100
    batch_size = 100
    elbo_samples = 4
    train_samples = None
    dirty_mnist = False
    mixture_prior = True
    lr = 0.01
    cuda = True
    model_param = "ens" # "det", "ens", 
    

@ex.automain
def train(elbo_samples: int, batch_size: int, num_epochs: int, lr: float,
          cuda: bool, model_param: str, mixture_prior: bool = False, checkpoint_name: str = None,
          train_samples: int = None, dirty_mnist: bool = False):
    print("starting train")

    device = torch.device("cuda" if cuda else "cpu")
    device_str = "cuda" if cuda else "cpu"
    print(f"{device_str=}")
    if dirty_mnist:
        train_data = ddu_dirty_mnist.DirtyMNIST("~/data", train=True, download=True,
                                               transform=transforms.Compose([
                                                   transforms.Normalize((-0.0651),(0.8897))
                                               ]), device=device_str)

        test_data = ddu_dirty_mnist.DirtyMNIST("~/data", train=False, download=True,
                                               transform=transforms.Compose([
                                                   transforms.Normalize((-0.0651),(0.8897))
                                               ]), device=device_str)
    else:
        train_data = torchvision.datasets.MNIST(root="~/data", train=True,
                                        download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor()]))

        test_data = torchvision.datasets.MNIST(root="~/data", train=False,
                                       download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor()]))
    
    num_classes = 10
    if model_param == "ens":
        num_models = elbo_samples
        bl_en_mnist = BaselineEnsembleMnist(device, num_models=num_models)
        train_losses = np.zeros((num_models, num_epochs))
        train_accs = np.zeros((num_models, num_epochs))
        train_error = np.zeros((num_models, num_epochs))
        test_losses = np.zeros((num_models, num_epochs))
        test_accs = np.zeros((num_models, num_epochs))
        test_error = np.zeros((num_models, num_epochs))
        test_entropy = np.zeros((num_models, num_epochs))
        for model_idx, curr_mnist_model in enumerate(bl_en_mnist.models):
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(curr_mnist_model.parameters(), lr=0.01)
            # num_classes = 10
            # epochs = 3
            # batch_size = 100

            loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
            loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

            for epoch in range(num_epochs):  # loop over the dataset multiple times
                total_acc = 0
                train_loss = 0
                for x_train, y_train in loader_train:
                    x_train = x_train.reshape(batch_size, -1).to(device)
                    y_train = y_train.to(device)
                    optimizer.zero_grad()
                    output = curr_mnist_model(x_train)
                    loss = criterion(output, y_train)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    y_train_preds = output.max(1).indices
                    correct_preds = sum(y_train_preds == y_train)
                    total_acc += correct_preds/len(y_train)
                epoch_accuracy = total_acc / len(loader_train)
                print(f"Model index: {model_idx}/{len(bl_en_mnist.models)}")
                print("epoch number: {} TRAIN loss: {} ".format(epoch, train_loss))
                print("epoch number: {} TRAIN accuracy: {} ".format(epoch, epoch_accuracy))
                train_losses[model_idx][epoch] = loss.item()
                train_accs[model_idx][epoch] = epoch_accuracy.item()
                train_error[model_idx][epoch] = 1 - epoch_accuracy.item()
                # print(f"{train_losses=}")
                # print(f"{train_accs=}")

                test_loss = 0
                total_test_acc = 0
                total_entropy = 0
                for x_test, y_test in loader_test:
                    x_test = x_test.reshape(batch_size, -1).to(device)
                    y_test = y_test.to(device)

                    softmax_averaged, entropy = bl_en_mnist.inference(x_test)
                    softmax_averaged = torch.tensor(softmax_averaged).to(device)
                    loss = criterion(softmax_averaged, y_test)

                    test_loss += loss.item()

                    y_test_preds = softmax_averaged.argmax(axis=1)
                    # print(f"{y_test_preds.shape=}")
                    # print(f"{y_test.shape=}")
                    correct_preds = sum(y_test_preds == y_test)
                    # print(f"{y_test_preds.shape=}")
                    total_test_acc += correct_preds/len(y_test)
                    total_entropy += entropy
                    # print(f"{total_test_acc=}")
                epoch_accuracy = total_test_acc / len(loader_test)
                print("epoch number: {} TEST loss: {} ".format(epoch, test_loss))
                print("epoch number: {} TEST accuracy: {} ".format(epoch, epoch_accuracy))
                test_losses[model_idx][epoch] = loss.item()
                test_accs[model_idx][epoch] = epoch_accuracy.item()
                test_error[model_idx][epoch] = 1 - epoch_accuracy.item()
                test_entropy[model_idx][epoch] = np.sum(total_entropy)


            print('Model #{} Done Training'.format(model_idx))
        print("Now logging stuff to neptune!")

        for i in range(num_epochs):

            ex.log_scalar("train loss", np.mean(train_losses[:,i]))
            ex.log_scalar("train accuracy", np.mean(train_accs[:,i]))
            ex.log_scalar("train error", np.mean(train_error[:,i]))
            ex.log_scalar("test loss", np.mean(test_losses[:,i]))
            ex.log_scalar("test accuracy", np.mean(test_accs[:,i]))
            ex.log_scalar("test error", np.mean(test_error[:,i]))
            ex.log_scalar("test entropy", np.mean(test_entropy[:,i]))



        # bl_mnist_drp_model = BaselineMnistWithDropout().to(device)
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(bl_mnist_drp_model.parameters(), lr=0.01)
        # loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        # loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

        # correct_preds = 0
        # total_acc = 0
        # for epoch in range(num_epochs):  # loop over the dataset multiple times
        #     total_acc = 0
        #     train_loss = 0
        #     print(f"{loader_train=}")
        #     bl_mnist_drp_model= bl_mnist_drp_model.train()
        #     for x_train, y_train in loader_train:
        #         x_train = x_train.reshape(batch_size, -1).to(device)
        #         y_train = y_train.to(device)
        #         optimizer.zero_grad()
        #         output = bl_mnist_drp_model(x_train)
        #         loss = criterion(output, y_train)
        #         loss.backward()
        #         optimizer.step()
        #         train_loss += loss.item()
        #         y_train_preds = output.max(1).indices
        #         # print(f"{y_train_preds.shape=}")
        #         # print(f"{y_train.shape=}")
        #         correct_preds = sum(y_train_preds == y_train)
        #         # print(f"{y_train_preds.shape=}")
        #         total_acc += correct_preds/len(y_train)
        #         # print(f"{total_acc=}")
        #     epoch_accuracy = total_acc / len(loader_train)
        #     print("epoch number: {} TRAIN loss: {} ".format(epoch, train_loss))
        #     print("epoch number: {} TRAIN accuracy: {} ".format(epoch, epoch_accuracy))
        #     ex.log_scalar("train loss", loss.item())
        #     ex.log_scalar("train accuracy", epoch_accuracy)
        #     ex.log_scalar("train error", 1 - epoch_accuracy)
        #     # print('Finished Training')

        #     test_loss = 0
        #     total_test_acc = 0
        #     bl_mnist_drp_model= bl_mnist_drp_model.eval()
        #     for x_test, y_test in loader_test:
        #         x_test = x_test.reshape(batch_size, -1).to(device)
        #         y_test = y_test.to(device)
        #         # optimizer.zero_grad()
        #         output = bl_mnist_drp_model(x_test)
        #         loss = criterion(output, y_test)
        #         # loss.backward()
        #         # optimizer.step()
        #         test_loss += loss.item()
        #         y_test_preds = output.max(1).indices
        #         # print(f"{y_test_preds.shape=}")
        #         # print(f"{y_test.shape=}")
        #         correct_preds = sum(y_test_preds == y_test)
        #         # print(f"{y_test_preds.shape=}")
        #         total_test_acc += correct_preds/len(y_test)
        #         # print(f"{total_test_acc=}")
        #     epoch_accuracy = total_test_acc / len(loader_test)
        #     print("epoch number: {} TEST loss: {} ".format(epoch, test_loss))
        #     print("epoch number: {} TEST accuracy: {} ".format(epoch, epoch_accuracy))
        #     ex.log_scalar("test loss", loss.item())
        #     ex.log_scalar("test accuracy", epoch_accuracy)
        #     ex.log_scalar("test error", 1 - epoch_accuracy)
        # print('Finished')
    # num_classes = 10
    # if train_samples != None:
    #     subset = list(range(0, train_samples))
    #     train_data = torch.utils.data.Subset(train_data, subset)
    # loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    # loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # num_train_batches = len(loader_train)
    # num_test_batches = len(loader_test)
    # for epoch in range(num_epochs):
    #     train_loss = 0
    #     total_accuracy = 0
    #     num_train_batches = len(loader_train)
    #     for x_train, y_train in loader_train:
    #         x_train = x_train.reshape(batch_size, -1).to(device)
    #         y_train = y_train.to(device)
    #         optimiser.zero_grad()
    #         loss, accuracy = model.energy_loss(
    #             x_train, y_train, num_train_batches, num_classes, elbo_samples)
    #         loss.backward()
    #         optimiser.step()
    #         train_loss += loss.item()
    #         total_accuracy += accuracy
    #     epoch_accuracy = total_accuracy / num_train_batches
    #     print("epoch number: {} TRAIN loss: {} ".format(epoch, train_loss))
    #     print("epoch number: {} TRAIN accuracy: {} ".format(epoch, epoch_accuracy))
    #     ex.log_scalar("train loss", loss.item())
    #     ex.log_scalar("train accuracy", epoch_accuracy)
    #     ex.log_scalar("train error", 1 - epoch_accuracy)
    #     test_loss = 0
    #     total_test_accuracy = 0
    #     total_entropy = 0
    #     num_test_batches = len(loader_test)
    #     for x_test, y_test in loader_test:
    #         x_test = x_test.reshape(batch_size, -1).to(device)
    #         y_test = y_test.to(device)
    #         test_loss, _ = model.energy_loss(
    #             x_test, y_test, num_test_batches, num_classes, elbo_samples)
    #         test_loss += test_loss.item()
    #         softmax_averaged, entropy = model.inference(x_test, 10, elbo_samples, batch_size)
    #         pred = softmax_averaged.argmax(axis=1)
    #         accuracy = sum(pred == y_test.to("cpu").numpy())
    #         total_test_accuracy += (accuracy / batch_size)
    #         total_entropy += np.sum(entropy)
    #     average_test_acc = total_test_accuracy/num_test_batches
        
    #     print(
    #         "epoch number {} TEST loss: {} ".format(epoch, test_loss))
    #     print(
    #         "epoch number {} TEST acc: {} ".format(
    #             epoch, average_test_acc.item()))
    #     print("epoch number {} entropy {}".format(epoch, total_entropy))
    #     ex.log_scalar("test loss", test_loss.item())
    #     ex.log_scalar("test accuracy", average_test_acc)
    #     ex.log_scalar("test error", 1 - average_test_acc)
    #     ex.log_scalar("test entropy", total_entropy )
    #     if checkpoint_name is not None:
    #         torch.save(model.state_dict(), checkpoint_name + ".checkpoint")
        
nep_run.stop()
