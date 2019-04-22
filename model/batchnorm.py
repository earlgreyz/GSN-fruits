import torch
from torch import nn
from torch.nn.parameter import Parameter


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(BatchNorm, self).__init__()
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches', torch.tensor(0, dtype=torch.long))
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _, C, _, _ = x.shape
        if self.training:
            mean = torch.mean(x, dim=(0, 2, 3))
            variance = torch.mean((x - mean.reshape((1, C, 1, 1))) ** 2, dim=(0, 2, 3))
            self.running_mean = (self.num_batches * self.running_mean + mean) / (self.num_batches + 1)
            self.running_var = (self.num_batches * self.running_var + variance) / (self.num_batches + 1)
            self.num_batches += 1
        else:
            mean, variance = self.running_mean, self.running_var
        mean, variance = mean.reshape((1, C, 1, 1)), variance.reshape((1, C, 1, 1))
        x = (x - mean) / ((variance + self.eps) ** 0.5)
        x = self.weight.reshape((1, C, 1, 1)) * x + self.bias.reshape((1, C, 1, 1))
        return x

