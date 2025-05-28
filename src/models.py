import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
            nn.Tanh()          # because inputs were scaled
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)
