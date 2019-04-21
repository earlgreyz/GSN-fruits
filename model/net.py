from torch import nn
import torch.nn.functional as F

from model.batchnorm import BatchNorm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4)
        self.bn2 = BatchNorm(16)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)
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
