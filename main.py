import torch

import click
from torch import cuda

from net import Net
from train import train
from test import test


@click.command()
@click.option('--load-model', '-m', default=None)
@click.option('--save-model', '-s', default=None)
@click.option('--train-dataset', '-d', default='./dataset/Training')
@click.option('--test-dataset', '-t', default='./dataset/Test')
@click.option('--no-train', is_flag=True, default=False)
@click.option('--no-test', is_flag=True, default=False)
@click.option('--epochs', '-e', default=5)
@click.option('--batch-size', '-b', default=100)
@click.option('--learning-rate', '-l', default=0.01)
def main(load_model: str, save_model: str,
            train_dataset: str, test_dataset: str,
            no_train: bool, no_test: bool,
            epochs: int, batch_size: int, learning_rate: float):
    net = Net()

    if load_model is not None:
        net.load_state_dict(torch.load(load_model))

    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
    net.to(device)
    click.echo('Using device={}'.format(device), color='blue')

    if not no_train:
        click.echo('Training net using {}'.format(train_dataset))
        net.train()
        train(net, data_path=train_dataset, batch_size=batch_size, num_epochs=epochs, learning_rate=learning_rate)

    if not no_train and save_model is not None:
        click.echo('Saving model as \'{}\''.format(save_model))
        torch.save(net.state_dict(), save_model)

    if not no_test:
        click.echo('Testing net using {}'.format(test_dataset))
        net.eval()
        accuracy = test(net, data_path=test_dataset, batch_size=100)
        click.echo('Net accuracy={}'.format(accuracy), color='green')


if __name__ == '__main__':
    main()
