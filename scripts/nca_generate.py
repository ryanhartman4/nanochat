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


def gzip_compression_ratio(tokens):
    """Compute gzip compression ratio for a token sequence.

    Returns r = compressed_size / raw_size. Lower r = more compressible = simpler.
    """
    # Convert tokens to bytes. Use uint8 if max value fits, else uint16.
    np_tokens = tokens.numpy()
    if np_tokens.max() < 256:
        raw_bytes = np_tokens.astype('uint8').tobytes()
    else:
        raw_bytes = np_tokens.astype('<u2').tobytes()
    compressed = gzip.compress(raw_bytes)
    return len(compressed) / len(raw_bytes)


def passes_complexity_filter(tokens, min_ratio=0.50):
    """Return True if the token sequence passes the complexity filter.

    Rejects trivial (highly compressible) sequences. Keeps sequences with
    compression ratio above min_ratio.
    """
    return gzip_compression_ratio(tokens) > min_ratio


def generate_dataset(num_tokens, seq_len, alphabet_size, output_dir, min_gzip_ratio=0.50, grid_size=12):
    """Generate a full NCA dataset.

    Generates trajectories with random rules, filters by complexity, tokenizes,
    and packs into sequences of seq_len.

    Args:
        num_tokens: Target total number of tokens to generate
        seq_len: Sequence length per sample (should match main training, e.g. 2048)
        alphabet_size: NCA alphabet size (2 or 4)
        output_dir: Directory to save output
        min_gzip_ratio: Minimum gzip compression ratio for filtering
        grid_size: NCA grid size (paper uses 12)
    """
    patches_per_grid = (grid_size // 2) ** 2  # 36 for 12x12
    steps_per_seq = seq_len // patches_per_grid  # 56 for seq_len=2048
    num_sequences = (num_tokens + seq_len - 1) // seq_len  # ceiling division

    all_sequences = []
    generated = 0
    rejected = 0

    while generated < num_sequences:
        rule = create_nca_rule(alphabet_size)
        grids = simulate_trajectory(rule, alphabet_size, grid_size=grid_size, num_steps=steps_per_seq)
        tokens = tokenize_trajectory(grids, alphabet_size)

        # Truncate or pad to exact seq_len
        if len(tokens) >= seq_len:
            tokens = tokens[:seq_len]
        else:
            # Pad with zeros (padding token)
            tokens = F.pad(tokens, (0, seq_len - len(tokens)), value=0)

        if passes_complexity_filter(tokens, min_ratio=min_gzip_ratio):
            all_sequences.append(tokens)
            generated += 1
        else:
            rejected += 1

    dataset = torch.stack(all_sequences)  # (num_sequences, seq_len)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "nca_data.pt")
    torch.save(dataset, output_path)
    print(f"Generated {generated} sequences ({rejected} rejected by filter), saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NCA pre-pre-training dataset")
    parser.add_argument("--num-tokens", type=int, required=True, help="Target number of tokens to generate")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length per sample")
    parser.add_argument("--alphabet-size", type=int, default=2, choices=[2, 4, 10], help="NCA alphabet size")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--min-gzip-ratio", type=float, default=0.50, help="Minimum gzip compression ratio")
    args = parser.parse_args()

    generate_dataset(
        num_tokens=args.num_tokens,
        seq_len=args.seq_len,
        alphabet_size=args.alphabet_size,
        output_dir=args.output,
        min_gzip_ratio=args.min_gzip_ratio,
    )
