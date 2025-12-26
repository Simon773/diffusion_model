import torch
import torch.nn as nn


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        # Convulutional Block  1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # Time Embedding MLP
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))

        # Convolutional Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, t):
        x1 = self.conv1(x)
        time_emb = self.time_mlp(t)

        # time_emb[:, :, None, None] adds two dimensions
        # to match the spatial dimensions of x1 for broadcasting
        # ex : if time_emb shape is (128, 128), it becomes (128, 128, 1, 1)
        x2 = x1 + time_emb[:, :, None, None]

        x = self.conv2(x2)

        return x
