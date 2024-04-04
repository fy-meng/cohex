import os.path

import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Subset

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

batch_size = 128
num_epochs = 2
device = torch.device('cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.log(), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output.log(), target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class MNISTScenario:
    def __init__(self, digits=(7, 9), test_size=1000):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('mnist_data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=batch_size, shuffle=True)

        test_dataset = datasets.MNIST('mnist_data', train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor()
                                      ]))
        test_indices = [idx for idx, target in enumerate(test_dataset.targets) if target in digits]
        self.test_loader = torch.utils.data.DataLoader(
            Subset(test_dataset, test_indices),
            batch_size=test_size, shuffle=False
        )

        self.model = Net().to(device)

        if os.path.exists('scenario/mnist_cnn.pth'):
            self.model.load_state_dict(torch.load('scenario/mnist_cnn.pth'))
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
            for epoch in range(1, num_epochs + 1):
                train(self.model, device, train_loader, optimizer, epoch)
                test(self.model, device, self.test_loader)
            torch.save(self.model.state_dict(), 'scenario/mnist_cnn.pth')

        self.model.eval()

        batch = next(iter(self.test_loader))
        self.X, self.y = batch

    def pred_from_rgb(self, X):
        X = rgb2gray(X)
        if len(X.shape) == 4:
            X = X.transpose((1, 0, 2, 3))
        else:
            X = X[:, np.newaxis]
        return self.model(X).detach().numpy()

    def next(self):
        self.X, self.y = next(iter(self.test_loader))
