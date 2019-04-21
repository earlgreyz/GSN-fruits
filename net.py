import torch
from torch.nn.parameter import Parameter
from torch import nn, cuda
import torch.nn.functional as F


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-8):
        super(BatchNorm, self).__init__()
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _, C, _, _ = x.shape
        mean = torch.mean(x, dim=(0, 2, 3)).reshape((1, C, 1, 1))
        variance = torch.mean((x - mean) ** 2, dim=(0, 2, 3))
        x = (x - mean) / ((variance.reshape((1, C, 1, 1)) + self.eps) ** 0.5)
        x = self.weight.reshape((1, C, 1, 1)) * x +  self.bias.reshape((1, C, 1, 1))
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4)
        #self.bn2 = nn.BatchNorm2d(16)
        self.bn2 = BatchNorm(16)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)
        #self.bn3 = nn.BatchNorm2d(32)
        self.bn3 = BatchNorm(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc4 = nn.Linear(1152, 512)
        self.fc5 = nn.Linear(512, 95)

    def forward(self, x):
        in_size = x.size(0)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(self.bn2(F.relu(self.conv2(x))))
        x = self.pool3(self.bn3(F.relu(self.conv3(x))))
        x = x.view(in_size, -1)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

