import click

import torch
from torch import cuda
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import transforms

from fruit import FruitNet


def test(net: FruitNet, data_path: str, batch_size: int) -> float:
    trans = transforms.Compose([transforms.ToTensor(), ])

    test_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=trans)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with click.progressbar(test_loader) as bar:
        for inputs, labels in bar:
            if cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            mean = inputs.sum(dim=0, keepdim=True) / inputs.size(0)
            minimum = inputs.min(dim=0, keepdim=True)
            maximum = inputs.max(dim=0, keepdim=True)
            variance = ((inputs - mean) ** 2).sum(dim=0, keepdim=True) / inputs.size(0)
            inputs = (inputs - mean) / ((variance + 1e-8) ** 0.5)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total
