import torch
from torch import nn, cuda
import torch.nn.functional as F


bias2 = torch.tensor([-0.0403, 0.0712, -0.0283, 0.0456, 0.0496, 0.0114, -0.0790, 0.0305, 0.0038, 0.0061, 0.0490, 0.0827, -0.0060, -0.0316, -0.0299, 0.0084], requires_grad=False)
weight2 = torch.tensor([ 0.2413, 0.6167, 0.6191, 0.4878, 0.3865, 0.2764, 0.3128, 0.1325, 0.0697, 0.2582, 0.2585, 0.8624, 0.1757, 0.1494, -0.0030, 0.4738], requires_grad=False)

bias3 = torch.tensor([-0.0489, -0.0594, -0.0484, -0.0495, -0.0581, -0.0351, -0.0112, -0.0379, -0.0054, -0.0834, -0.0420, 0.0336, -0.0220, -0.0594, -0.0666, -0.0389, -0.0984, -0.0907, 0.0583, -0.0292, -0.0413, -0.0003, -0.0367, -0.0105, -0.0594, -0.0256, -0.0166, -0.0682, -0.0382, -0.0786, -0.0640, -0.0362], requires_grad=False)
weight3 = torch.tensor([ 0.5748, 0.1579, 0.8463, 0.1358, 0.1177, 0.9955, 0.0610, 0.0372, 0.6816, 0.7851, 0.8415, -0.0650, 0.9060, 0.3908, 0.4781, 0.5364, 0.8652, 0.7174, 0.1153, 0.8199, 0.7481, 0.1380, 0.3621, 0.7918, 0.1101, 0.0585, 0.0512, 0.5135, 0.3539, 0.2333, 0.2198, 0.3139], requires_grad=False)


class BatchNorm:
    def __init__(self, num_features, weight, bias, eps=1e-8):
        self.eps = eps

        self.weight = weight.reshape((1, num_features, 1, 1))
        self.bias = bias.reshape((1, num_features, 1, 1))

        if cuda.is_available():
            self.weight, self.bias = self.weight.to('cuda'), self.bias.to('cuda')

    def __call__(self, x):
        _, C, _, _ = x.shape
        mean = torch.mean(x, dim=(0, 2, 3)).reshape((1, C, 1, 1))
        var = torch.mean((x - mean) ** 2, dim=(0, 2, 3))
        X_hat = (x - mean) / (var.reshape((1, C, 1, 1)) + self.eps).pow(0.5)
        out = self.weight * X_hat + self.bias
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(16)
        #self.bn2 = BatchNorm(16, weight=weight2, bias=bias2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)
        self.bn3 = nn.BatchNorm2d(32)
        #self.bn3 = BatchNorm(32, weight=weight3, bias=bias3)
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

