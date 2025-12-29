import time

import torch
import torch.nn as nn

from modules.double_conv_block import DoubleConvBlock
from modules.embeddings import SinusoidalPositionalEmbedding


class Down(nn.Module):
    """
    Class for implementing the downsampling part of the U-Net architecture.
    Maxpool kernel 2*2 with stride 2 followed by DoubleConv.
    """

    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(in_channels, out_channels, time_embedding_dim)

    def forward(self, x, t):
        x = self.maxpool(x)
        x = self.conv(x, t)
        return x


class Up(nn.Module):
    """
    Class for up part in Unet.
    Upsampling followed by DoubleConv. We divided by 2 the number of input channels
    """

    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConvBlock(in_channels, out_channels, time_embedding_dim)

    def forward(self, x, x_skip, t):
        x = self.up(x)
        # concatenation for skip connection
        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x, t)


class Unet(nn.Module):
    """
    U-Net architecture implementation.
    """

    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),  # better than ReLU with smoothness
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        self.positional_embedding = SinusoidalPositionalEmbedding(time_embedding_dim)

        self.input_conv = DoubleConvBlock(
            in_channels, 64, time_emb_dim=time_embedding_dim
        )

        self.down1 = Down(64, 128, time_embedding_dim)
        self.down2 = Down(128, 256, time_embedding_dim)
        self.down3 = Down(256, 512, time_embedding_dim)
        self.down4 = Down(512, 1024, time_embedding_dim)

        self.bottom = DoubleConvBlock(1024, 1024, time_embedding_dim)

        self.up1 = Up(1024, 512, time_embedding_dim)
        self.up2 = Up(512, 256, time_embedding_dim)
        self.up3 = Up(256, 128, time_embedding_dim)
        self.up4 = Up(128, 64, time_embedding_dim)

        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.positional_embedding.forward(t)
        t_emb = self.time_mlp(t_emb)

        x1 = self.input_conv(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x5 = self.down4(x4, t_emb)

        x_bottom = self.bottom(x5, t_emb)

        x = self.up1(x_bottom, x4, t_emb)
        x = self.up2(x, x3, t_emb)
        x = self.up3(x, x2, t_emb)
        x = self.up4(x, x1, t_emb)
        output = self.output_conv(x)
        return output
