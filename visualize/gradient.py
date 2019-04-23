from torch import nn

import matplotlib.pyplot as plt

from visualize import utils


def gradient(net, image, label):
    criterion = nn.CrossEntropyLoss()

    input = image.clone()
    input.requires_grad_()
    outputs = net(input)
    loss = criterion(outputs, label)
    loss.backward()

    heatmap = input.grad.squeeze()
    heatmap = utils.normalize_image(heatmap)

    image = image.squeeze()

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (20, 20)

    fig.add_subplot(1, 2, 1)
    utils.plot_image(image.transpose(0, 1).transpose(1, 2), 'Original')

    fig.add_subplot(1, 2, 2)
    utils.plot_image(heatmap.transpose(0, 1).transpose(1, 2), 'Heatmap')

    plt.show()
    return heatmap
