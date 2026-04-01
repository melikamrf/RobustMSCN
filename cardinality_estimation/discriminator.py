import torch
from torch import nn


class LatentDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout=0.1):
        super(LatentDiscriminator, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        dims = [input_dim] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
