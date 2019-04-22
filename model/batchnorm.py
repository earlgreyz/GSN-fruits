import torch
from torch import nn
from torch.nn.parameter import Parameter


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum
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
            self._update_running_stats(mean, variance, self.momentum)
        else:
            mean, variance = self.running_mean, self.running_var
        mean, variance = mean.reshape((1, C, 1, 1)), variance.reshape((1, C, 1, 1))
        x = (x - mean) / ((variance + self.eps) ** 0.5)
        x = self.weight.reshape((1, C, 1, 1)) * x + self.bias.reshape((1, C, 1, 1))
        return x

    def _update_running_stats(self, mean, variance, momentum):
        if self.num_batches == 0:
            self.running_mean = mean
            self.running_var = variance
        else:
            self.running_mean = (1 - momentum) * self.running_mean + momentum * mean
            self.running_var = (1 - momentum) * self.running_var + momentum * variance
        self.num_batches += 1
