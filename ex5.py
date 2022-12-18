import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from torch.nn import Conv2d, LeakyReLU, MaxPool2d, Linear, Dropout, Sequential
from torch.nn.functional import log_softmax, nll_loss

from gcommand_dataset import GCommandLoader

EPOCHS = 4
LEARNING_RATE = 0.001
BATCH_SIZE = 100

device = torch.device("cuda:0" if cuda.is_available() else "cpu")  # use GPU if can


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # layer 1
        self.layer1 = Sequential(
            Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            LeakyReLU(0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2))
        # layer 2
        self.layer2 = Sequential(
            Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            LeakyReLU(0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2))
        # layer 3
        self.layer3 = Sequential(
            Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            LeakyReLU(0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2))
        # layer 4
        self.layer4 = Sequential(
            Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            LeakyReLU(0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2))
        # layer 5
        self.layer5 = Sequential(
            Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            LeakyReLU(0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2))
        # fully_connected
        self.fully_connected = Sequential(
            Linear(480, 512),
            Dropout(0.4),
            LeakyReLU(0.2, inplace=True),
            Linear(512, 256),
            Dropout(0.4),
            LeakyReLU(0.2, inplace=True),
            Linear(256, 30))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(x.size(0), -1)
        x = self.fully_connected(x)
        return log_softmax(x, dim=1)


def Train(model, optimizer, train_loader):
    model.train()  # we are training
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # zeroing the gradient
            output = model(data)  # forward propagation
            loss = nll_loss(output, target)  # set loss to Negative Log Likelihood
            loss.backward()  # backward propagation
            optimizer.step()  # update


def Test(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)  # forward propagation
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            predictions.append(pred)
    return predictions


def argSort(files_names):
    for i in range(len(files_names)):
        files_names[i] = int(files_names[i].split(".")[0])
    arg_sort = np.argsort(files_names)
    return arg_sort


def main():
    train_set = GCommandLoader('./gcommands/train')
    test_set = GCommandLoader('./gcommands/test')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, pin_memory=True)

    # MY Model
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    Train(model, optimizer, train_loader)
    predictions = Test(model, test_loader)

    # get file names
    files_names = []
    for index in range(len(test_loader)):
        # find file name
        file_name = test_set.spects[index][0]
        file_name = file_name.split("/")[-1]
        files_names.append(file_name)

    # sort
    arg_sort = argSort(files_names.copy())
    files_names = [files_names[i] for i in arg_sort]
    predictions = [predictions[i] for i in arg_sort]

    # print to output
    output_file = open("test_y", "w")
    for index in range(len(files_names)):
        # get prediction
        pred = predictions[index][0].item()
        pred = train_set.classes[pred]
        # make line to output
        line = files_names[index] + "," + str(pred) + "\n"
        output_file.write(line)
    output_file.close()


if __name__ == "__main__":
    main()
