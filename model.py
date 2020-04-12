import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import config


class Preprocess:

    def __init__(self):
        self.num_batches = None                                     # Init in create_data_loader

    def read_mnist(self):
        """
        Download and store mnist after preprocessing.
        :return: utils.data.Dataset object which can be passed to Dataloader directly.
        """
        compose = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

        return datasets.MNIST(root=config.out_dir, train=True, transform=compose, download=True)

    def create_data_loader(self, data):
        data_loader = DataLoader(data, batch_size=config.dataloader_batch_size, shuffle=True,)
        self.num_batches = len(data_loader)

        return data_loader


class Discriminator(nn.module):
    def __init__(self):
        super().__init__()

        self.hidden0 = nn.Sequential(
            nn.Linear(config.n_features, config.h1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(config.h1, config.h2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(config.h2, config.h3),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.output = nn.Sequential(
            nn.Linear(config.n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)

        return x

class Generator(nn.Module):
    super().__init__()

    


def main():
    pre = Preprocess()
    data = pre.read_mnist()
    data_loader = pre.create_data_loader(data)

    discriminator = Discriminator()


if __name__ == "__main__":
    main()
