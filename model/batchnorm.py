import torch
from torch import nn
from torch.nn.parameter import Parameter


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
        variance = torch.mean((x - mean) ** 2, dim=(0, 2, 3)).reshape((1, C, 1, 1))
        x = (x - mean) / ((variance + self.eps) ** 0.5)
        x = self.weight.reshape((1, C, 1, 1)) * x + self.bias.reshape((1, C, 1, 1))
        return x
