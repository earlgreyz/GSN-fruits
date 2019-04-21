import click

import torch
from torch import cuda
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import transforms

from net import Net


def test(net: Net, data_path: str, batch_size: int) -> float:
    trans = transforms.Compose([transforms.ToTensor(), ])

    test_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=trans)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    correct = 0
    total = 0

    with click.progressbar(test_loader) as bar:
        for inputs, labels in bar:
            if cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total
