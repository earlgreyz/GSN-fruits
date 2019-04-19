from torch import nn, cuda
from torch.optim import adam
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import transforms

from fruit import FruitNet


def train(model: FruitNet, data_path: str, batch_size: int, num_epochs: int, learning_rate: float):
    trans = transforms.Compose([transforms.ToTensor(), ])

    train_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=trans)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = adam.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            print('\r[{}, {:5d}] loss: {:3f}'.format(epoch + 1, i + 1, running_loss / 100), end='')

            inputs, labels = data
            if cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print('\r[{}, {:5d}] loss: {:3f}'.format(epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
