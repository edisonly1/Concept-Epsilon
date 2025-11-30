from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class LeakCNN(nn.Module):
    """
    Simple 1D CNN over time.

    Input:  [batch, in_channels (sensors), T]
    Output: [batch, num_classes]  (0 = no leak, 1..K = leak at node)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, T/2]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, T/4]

        # Global average over time: [B, 64]
        x = x.mean(dim=-1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
