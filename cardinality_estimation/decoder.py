import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.0):
        super(Decoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = []

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
