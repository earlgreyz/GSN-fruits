import torch

from torch import nn

import matplotlib.pyplot as plt


def _plot_image(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')


def gradient(net, image, label):
    criterion = nn.CrossEntropyLoss()

    input = image.clone()
    input.requires_grad_()
    outputs = net(input)
    loss = criterion(outputs, label)
    loss.backward()

    heatmap = input.grad.squeeze()
    min_heatmap = torch.min(heatmap)
    range_heatmap = torch.max(heatmap) - min_heatmap
    heatmap = (heatmap - min_heatmap) / range_heatmap

    image = image.squeeze()

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (20, 20)

    fig.add_subplot(1, 2, 1)
    _plot_image(image.transpose(0, 1).transpose(1, 2), 'Original')

    fig.add_subplot(1, 2, 2)
    _plot_image(heatmap.transpose(0, 1).transpose(1, 2), 'Heatmap')

    plt.show()
