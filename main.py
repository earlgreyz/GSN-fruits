import torch

import click

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
    model = FruitNet()

    if load_model is not None:
        model = torch.load(load_model)

    if not no_train:
        model.train()
        train(model, data_path=train_dataset, batch_size=100, num_epochs=5, learning_rate=0.01)

    if not no_train and save_model is not None:
        torch.save(model, save_model)

    if not no_test:
        model.eval()
        accuracy = test(model, data_path=test_dataset, batch_size=100)
        click.echo(click.style('Model accuracy={}'.format(accuracy), fg='green'))


if __name__ == '__main__':
    main()
