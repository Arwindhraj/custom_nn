import torch
from torch import nn


# Activation functions

class SiLU(nn.Module):
    # SiLU activation https://arxiv.org/pdf/1606.08415.pdf
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
