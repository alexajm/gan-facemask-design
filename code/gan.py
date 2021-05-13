import numpy as np
import torch
from torch import nn, optim

class Generator(nn.Module):
    def __init__(self, learning_rate=1e-3):
        # Initialize
        super(Generator, self).__init__()
        self.input_dim = 100
        self.output_dim = (64, 64, 3)
        self.learning_rate = learning_rate

        # Network layers
        # Input: 100-dim latent vector
        # Output: 64x64x3 RGB image
        self.h1 = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            )
        self.output = nn.Sequential(
            nn.Linear(16, np.prod(output_dim)),
            nn.Sigmoid(),
            )

        # Optimization
        self.optimizer = optim.Adam(self.parameters(), learning_rate)

    def forward(self, latent_=None):
        latent = latent_ if latent_ is not None else np.random.rand(100, 1)
        layer1 = self.h1(latent)
        output = self.output(layer1)
        return torch.reshape(output, output_dim)
