import torch

import click
from torch import cuda

from model.net import Net
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
    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
    click.secho('Using device={}'.format(device), fg='blue')

    net = Net()
    net.to(device)

    if load_model is not None:
        click.secho('Loading model from \'{}\''.format(load_model), fg='yellow')
        net.load_state_dict(torch.load(load_model, map_location=device))

    if not no_train:
        click.echo('Training model using {}'.format(train_dataset))
        net.train()
        train(net, data_path=train_dataset, batch_size=batch_size, num_epochs=epochs, learning_rate=learning_rate)

    if not no_train and save_model is not None:
        click.secho('Saving model as \'{}\''.format(save_model), fg='yellow')
        torch.save(net.state_dict(), save_model)

    if not no_test:
        click.echo('Testing model using {}'.format(test_dataset))
        net.eval()
        accuracy = test(net, data_path=test_dataset, batch_size=batch_size)
        color = 'green' if accuracy > 97. else 'red'
        click.secho('Accuracy={}'.format(accuracy), fg=color)


if __name__ == '__main__':
    main()
