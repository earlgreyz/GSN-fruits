import torch

import click
from torch import cuda

from fruit import FruitNet
from train import train
from test import test

num_epochs = 5
batch_size = 100
learning_rate = 0.01


@click.command()
@click.option('--load-model', '-m', default=None)
@click.option('--save-model', '-s', default=None)
@click.option('--train-dataset', '-d', default='./dataset/Training')
@click.option('--test-dataset', '-t', default='./dataset/Test')
@click.option('--no-train', is_flag=True, default=False)
@click.option('--no-test', is_flag=True, default=False)
def main(load_model: str, save_model: str, train_dataset: str, test_dataset: str, no_train: bool, no_test: bool):
    net = FruitNet()

    if load_model is not None:
        net = torch.load(load_model)

    device = torch.device("cuda:0" if cuda.is_available() else "cpu")
    net.to(device)
    click.echo(f'Using device={device}', color='blue')

    if not no_train:
        click.echo(f'Training net using {train_dataset}')
        net.train()
        train(net, data_path=train_dataset, batch_size=100, num_epochs=5, learning_rate=0.01)

    if not no_train and save_model is not None:
        click.echo(f'Saving model as \'{save_model}\'')
        torch.save(net, save_model)

    if not no_test:
        click.echo(f'Testing net using {test_dataset}')
        net.eval()
        accuracy = test(net, data_path=test_dataset, batch_size=100)
        click.echo(f'Net accuracy={accuracy}', color='green')


if __name__ == '__main__':
    main()
