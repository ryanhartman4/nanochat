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


@torch.no_grad()
def simulate_trajectory(rule, alphabet_size, grid_size=12, num_steps=56, tau=1e-3):
    """Simulate an NCA trajectory from a random initial state.

    Args:
        rule: NCA transition rule (from create_nca_rule)
        alphabet_size: Number of cell states
        grid_size: Grid width/height (paper uses 12)
        num_steps: Number of simulation steps
        tau: Softmax temperature for stochastic sampling (paper uses 1e-3)

    Returns:
        Tensor of shape (num_steps+1, alphabet_size, grid_size, grid_size) — one-hot grid states
    """
    # Random initial state: sample uniformly from alphabet
    state_indices = torch.randint(0, alphabet_size, (1, grid_size, grid_size))
    grid = F.one_hot(state_indices, alphabet_size).permute(0, 3, 1, 2).float()  # (1, alphabet, H, W)

    grids = [grid.squeeze(0)]  # collect initial state

    for _ in range(num_steps):
        logits = rule(grid)  # (1, alphabet, H, W)
        # Stochastic sampling with temperature
        probs = F.softmax(logits / tau, dim=1)  # (1, alphabet, H, W)
        # Sample: reshape to (H*W, alphabet), sample, reshape back
        flat_probs = probs.squeeze(0).permute(1, 2, 0).reshape(-1, alphabet_size)  # (H*W, alphabet)
        samples = torch.multinomial(flat_probs, 1).squeeze(-1)  # (H*W,)
        grid = F.one_hot(samples, alphabet_size).float().reshape(grid_size, grid_size, alphabet_size)
        grid = grid.permute(2, 0, 1).unsqueeze(0)  # (1, alphabet, H, W)
        grids.append(grid.squeeze(0))

    return torch.stack(grids)  # (num_steps+1, alphabet, H, W)


def tokenize_trajectory(grids, alphabet_size):
    """Convert a trajectory of one-hot grids into a flat token sequence.

    Uses 2x2 non-overlapping patches. Each patch of 4 cells maps bijectively
    to a token ID in [0, alphabet_size^4).

    Mapping: token = c0 * n^3 + c1 * n^2 + c2 * n + c3
    where c0..c3 are the cell states in the 2x2 patch (row-major) and n = alphabet_size.

    Args:
        grids: Tensor of shape (T, alphabet_size, H, W) — one-hot grid states
        alphabet_size: Number of cell states per cell

    Returns:
        1D tensor of token IDs, length T * (H/2) * (W/2)
    """
    T, n, H, W = grids.shape
    assert H % 2 == 0 and W % 2 == 0, f"Grid {H}x{W} must be even for 2x2 patches"

    # Convert one-hot to cell state indices: (T, H, W)
    cell_states = grids.argmax(dim=1)  # (T, H, W)

    # Extract 2x2 patches: reshape to (T, H//2, 2, W//2, 2) then combine
    patches = cell_states.reshape(T, H // 2, 2, W // 2, 2)
    c0 = patches[:, :, 0, :, 0]  # top-left
    c1 = patches[:, :, 0, :, 1]  # top-right
    c2 = patches[:, :, 1, :, 0]  # bottom-left
    c3 = patches[:, :, 1, :, 1]  # bottom-right

    # Bijective mapping: mixed-radix encoding
    n = alphabet_size
    token_ids = c0 * (n**3) + c1 * (n**2) + c2 * n + c3  # (T, H//2, W//2)

    # Flatten: row-major serialization
    return token_ids.reshape(-1).long()
