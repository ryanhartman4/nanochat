"""
NCA Data Generator for pre-pre-training.

Generates synthetic Neural Cellular Automata trajectories as a pre-training dataset.
Based on Han et al. 2026 (arXiv:2603.10055).

Usage:
    python -m scripts.nca_generate --num-tokens 164000000 --seq-len 2048 --alphabet-size 2 --output /path/to/output
"""

import os
import gzip
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class NCARule(nn.Module):
    """NCA transition rule: 3x3 conv (circular padding) -> ReLU -> 1x1 conv.

    Maps (B, alphabet_size, H, W) -> (B, alphabet_size, H, W) logits.
    Uses circular padding internally for periodic (toroidal) boundaries.
    """
    def __init__(self, alphabet_size, hidden_size=16):
        super().__init__()
        self.conv1 = nn.Conv2d(alphabet_size, hidden_size, kernel_size=3, padding=0, bias=False)
        self.conv2 = nn.Conv2d(hidden_size, alphabet_size, kernel_size=1, bias=False)
        for p in self.parameters():
            nn.init.normal_(p, std=0.5)

    def forward(self, grid):
        padded = F.pad(grid, (1, 1, 1, 1), mode='circular')
        x = self.conv1(padded)
        x = F.relu(x)
        x = self.conv2(x)
        return x


def create_nca_rule(alphabet_size, hidden_size=16):
    """Create a random NCA transition rule as a small neural network.

    Architecture: 3x3 conv (circular padding) -> ReLU -> 1x1 conv (logits)
    This maps the one-hot grid state to next-step logits for each cell.

    Args:
        alphabet_size: Number of possible cell states (n in the paper)
        hidden_size: Hidden layer size in the MLP (paper uses 16)

    Returns:
        NCARule module that maps (B, alphabet_size, 12, 12) -> (B, alphabet_size, 12, 12)
    """
    return NCARule(alphabet_size, hidden_size)
