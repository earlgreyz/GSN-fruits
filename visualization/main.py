import click

import torch
from torch import cuda

from PIL import Image
import torchvision.transforms.functional as F

from model.net import Net
from visualization.classes import classes
from visualization.occlusion import occlustion


def load_image(path):
    image = Image.open(path)
    image = F.to_tensor(image)
    return image.unsqueeze(0)


@click.command()
@click.argument('images', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--model', '-m', default='artifacts/model.state')
@click.option('--occlusion-window', '-k', default=10)
@click.option('--occlusion-stride', '-s', default=5)
def main(model: str, images: [str], occlusion_window: int, occlusion_stride: int):
    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
    click.secho('Using device={}'.format(device), fg='blue')

    net = Net()
    net.to(device)

    click.secho('Loading model from \'{}\''.format(model), fg='yellow')
    net.load_state_dict(torch.load(model, map_location=device))
    net.eval()

    for path in images:
        image = load_image(path).to(device)
        output = net(image)
        _, predicted = torch.max(output.data, 1)
        click.echo('Image \'{}\' most likely represents a \'{}\''.format(path, classes[predicted]))
        occlustion(net, image, predicted, k=occlusion_window, stride=occlusion_stride)


if __name__ == '__main__':
    main()
