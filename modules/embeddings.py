import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, time):
        """
        Args: time (Tensor): Tenseur 1D contenant les indices de temps (Batch_Size,)

        Returns:
            Tensor: Embeddings of size (Batch_Size, dim)
        """
        device = time.device
        half_dim = self.embedding_dim // 2  # for having 2i or 2i+1 dimensions
        # using property of exp and log to compute the sinusoidal frequencies
        # (10000^(2i/dim) = exp(log(10000) * (2i/dim)))
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # To have a embedding of size (Batch_Size, dim)
        embeddings = time[:, None] * embeddings[None, :]

        full_emb = torch.zeros(time.shape[0], self.embedding_dim, device=device)
        full_emb[:, 0::2] = embeddings.sin()
        full_emb[:, 1::2] = embeddings.cos()

        return full_emb
