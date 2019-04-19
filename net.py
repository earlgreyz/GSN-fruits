import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=10)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(40)
        self.fc1 = nn.Linear(1960, 500)
        self.fc2 = nn.Linear(500, 95)

    def forward(self, x):
        in_size = x.size(0)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.bn1(self.conv3(x))))
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

