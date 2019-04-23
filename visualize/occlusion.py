import click

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from visualize import utils


def occlustion(net, image, label, k=10, stride=4):
    _, C, W, H = image.shape
    width = int((W - k) / stride + 1)
    height = int((H - k) / stride + 1)
    heatmap = torch.zeros(width, height)
    image = image.data

    with click.progressbar(range(0, H - k + 1, stride)) as bar:
        for y, i in enumerate(bar):
            for x, j in enumerate(range(0, W - k + 1, stride)):
                tmp = image.clone()
                tmp[:, :, j:j + k, i:i + k] = 0
                softmax = F.softmax(net(tmp), dim=1).data[0]
                heatmap[y, x] = softmax[label]

    image = image.squeeze()

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (20, 20)

    fig.add_subplot(1, 2, 1)
    utils.plot_image(image.transpose(0, 1).transpose(1, 2), 'Original')

    fig.add_subplot(1, 2, 2)
    utils.plot_image(heatmap, 'Heatmap')

    plt.show()
    return heatmap
