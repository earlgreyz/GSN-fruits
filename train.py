import click

from torch import nn, cuda
from torch.optim import adam
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import transforms

from fruit import FruitNet


def train(net: FruitNet, data_path: str, batch_size: int, num_epochs: int, learning_rate: float):
    trans = transforms.Compose([transforms.ToTensor(), ])

    train_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=trans)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = adam.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        def show_loss(item):
            return '[{}, {:3f}]'.format(epoch + 1, running_loss / 500)

        with click.progressbar(train_loader, item_show_func=show_loss) as bar:
            for inputs, labels in bar:
                if cuda.is_available():
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')

                mean = inputs.sum(dim=0, keepdim=True) / inputs.size(0)
                minimum = inputs.min(dim=0, keepdim=True)
                maximum = inputs.max(dim=0, keepdim=True)
                variance = ((inputs - mean) ** 2).sum(dim=0, keepdim=True) / inputs.size(0)
                inputs = (inputs - mean) / ((variance + 1e-8) ** 0.5)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
