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


def create_nca_rule(alphabet_size, hidden_size=16, device="cpu"):
    """Create a random NCA transition rule as a small neural network.

    Args:
        alphabet_size: Number of possible cell states (n in the paper)
        hidden_size: Hidden layer size in the MLP (paper uses 16)
        device: Device to place rule on ("cpu", "cuda", etc.)

    Returns:
        NCARule module on the specified device
    """
    return NCARule(alphabet_size, hidden_size).to(device)


@torch.no_grad()
def simulate_trajectory(rule, alphabet_size, grid_size=12, num_steps=56, tau=1e-3,
                        batch_size=1, device="cpu"):
    """Simulate NCA trajectories from random initial states.

    Supports batched simulation: multiple trajectories with the same rule
    run in parallel via batched conv2d.

    Args:
        rule: NCA transition rule (from create_nca_rule)
        alphabet_size: Number of cell states
        grid_size: Grid width/height (paper uses 12)
        num_steps: Number of simulation steps
        tau: Softmax temperature for stochastic sampling (paper uses 1e-3)
        batch_size: Number of trajectories to simulate in parallel
        device: Device for simulation tensors

    Returns:
        Tensor of shape (batch_size, num_steps+1, alphabet_size, grid_size, grid_size)
    """
    B = batch_size
    # Random initial states: (B, grid_size, grid_size) indices
    state_indices = torch.randint(0, alphabet_size, (B, grid_size, grid_size), device=device)
    grid = F.one_hot(state_indices, alphabet_size).permute(0, 3, 1, 2).float()  # (B, alphabet, H, W)

    grids = [grid]  # collect initial state: (B, alphabet, H, W)

    for _ in range(num_steps):
        logits = rule(grid)  # (B, alphabet, H, W) — batched conv2d
        # Stochastic sampling with temperature
        probs = F.softmax(logits / tau, dim=1)  # (B, alphabet, H, W)
        # Sample: reshape to (B*H*W, alphabet), sample, reshape back
        flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, alphabet_size)  # (B*H*W, alphabet)
        samples = torch.multinomial(flat_probs, 1).squeeze(-1)  # (B*H*W,)
        grid = F.one_hot(samples, alphabet_size).float().reshape(B, grid_size, grid_size, alphabet_size)
        grid = grid.permute(0, 3, 1, 2)  # (B, alphabet, H, W)
        grids.append(grid)

    return torch.stack(grids, dim=1)  # (B, num_steps+1, alphabet, H, W)


def tokenize_trajectory(grids, alphabet_size):
    """Convert one-hot grids into flat token sequences.

    Uses 2x2 non-overlapping patches. Each patch of 4 cells maps bijectively
    to a token ID in [0, alphabet_size^4).

    Supports both single and batched input:
      - Single: grids shape (T, alphabet_size, H, W) -> returns (T * H/2 * W/2,)
      - Batched: grids shape (B, T, alphabet_size, H, W) -> returns (B, T * H/2 * W/2)

    Args:
        grids: Tensor of grid states (see shapes above)
        alphabet_size: Number of cell states per cell

    Returns:
        Token IDs tensor (1D for single, 2D for batched)
    """
    batched = grids.dim() == 5
    if not batched:
        grids = grids.unsqueeze(0)  # add batch dim

    B, T, n, H, W = grids.shape
    assert H % 2 == 0 and W % 2 == 0, f"Grid {H}x{W} must be even for 2x2 patches"

    # Convert one-hot to cell state indices: (B, T, H, W)
    cell_states = grids.argmax(dim=2)

    # Extract 2x2 patches
    patches = cell_states.reshape(B, T, H // 2, 2, W // 2, 2)
    c0 = patches[:, :, :, 0, :, 0]  # top-left
    c1 = patches[:, :, :, 0, :, 1]  # top-right
    c2 = patches[:, :, :, 1, :, 0]  # bottom-left
    c3 = patches[:, :, :, 1, :, 1]  # bottom-right

    # Bijective mapping: mixed-radix encoding
    token_ids = c0 * (alphabet_size**3) + c1 * (alphabet_size**2) + c2 * alphabet_size + c3

    # Flatten time and spatial dims: (B, T * H/2 * W/2)
    token_ids = token_ids.reshape(B, -1).long()

    if not batched:
        return token_ids.squeeze(0)  # remove batch dim for single input
    return token_ids


def gzip_compression_ratio(tokens):
    """Compute gzip compression ratio for a token sequence.

    Returns r = compressed_size / raw_size. Lower r = more compressible = simpler.
    Expects a CPU tensor (moves to CPU if needed).
    """
    # Ensure CPU for .numpy()
    if tokens.is_cuda:
        tokens = tokens.cpu()
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


def _resolve_device(device_str):
    """Resolve device string: 'auto' picks cuda if available, else cpu."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def generate_dataset(num_tokens, seq_len, alphabet_size, output_dir, min_gzip_ratio=0.50,
                     grid_size=12, trajectories_per_rule=128, device="auto"):
    """Generate a full NCA dataset with batched simulation.

    Creates random NCA rules, simulates multiple trajectories per rule in
    a batched forward pass, filters by gzip complexity, and saves.

    Args:
        num_tokens: Target total number of tokens to generate
        seq_len: Sequence length per sample (should match main training, e.g. 2048)
        alphabet_size: NCA alphabet size (2 or 4)
        output_dir: Directory to save output
        min_gzip_ratio: Minimum gzip compression ratio for filtering
        grid_size: NCA grid size (paper uses 12)
        trajectories_per_rule: Trajectories simulated per rule (controls diversity vs speed)
        device: Device for simulation ("auto", "cpu", "cuda")
    """
    device = _resolve_device(device) if isinstance(device, str) else device
    print(f"NCA generation on device: {device}")

    patches_per_grid = (grid_size // 2) ** 2  # 36 for 12x12
    # We want complete grids only. E.g. seq_len=2048, 36 tok/grid:
    # 2048 // 36 = 56 grids → 55 steps → 56 * 36 = 2016 tokens → pad 32.
    grids_per_seq = seq_len // patches_per_grid
    steps_per_seq = grids_per_seq - 1  # simulate N-1 steps + initial = N grids
    num_sequences = (num_tokens + seq_len - 1) // seq_len  # ceiling division

    all_sequences = []
    generated = 0
    rejected = 0
    num_rules = 0

    while generated < num_sequences:
        # How many trajectories to simulate this batch
        remaining = num_sequences - generated
        # Over-generate to account for gzip rejection (~50% acceptance)
        batch_size = min(trajectories_per_rule, remaining * 2)

        rule = create_nca_rule(alphabet_size, device=device)
        num_rules += 1

        # Batched simulation: (B, T+1, alphabet, H, W)
        grids = simulate_trajectory(
            rule, alphabet_size, grid_size=grid_size,
            num_steps=steps_per_seq, batch_size=batch_size, device=device,
        )

        # Batched tokenization: (B, T * patches_per_grid)
        tokens_batch = tokenize_trajectory(grids, alphabet_size)  # (B, raw_tokens)

        # Truncate or pad each sequence to exact seq_len
        raw_len = tokens_batch.shape[1]
        if raw_len >= seq_len:
            tokens_batch = tokens_batch[:, :seq_len]
        else:
            tokens_batch = F.pad(tokens_batch, (0, seq_len - raw_len), value=0)

        # Move to CPU for gzip filtering (gzip needs numpy)
        tokens_cpu = tokens_batch.cpu()

        # Filter each sequence individually
        for i in range(tokens_cpu.shape[0]):
            if generated >= num_sequences:
                break
            seq = tokens_cpu[i]
            if passes_complexity_filter(seq, min_ratio=min_gzip_ratio):
                all_sequences.append(seq)
                generated += 1
            else:
                rejected += 1

        if num_rules % 10 == 0:
            print(f"Progress: {generated}/{num_sequences} sequences "
                  f"({num_rules} rules, {rejected} rejected)")

    dataset = torch.stack(all_sequences)  # (num_sequences, seq_len)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "nca_data.pt")
    torch.save(dataset, output_path)
    print(f"Generated {generated} sequences from {num_rules} rules "
          f"({rejected} rejected by filter), saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NCA pre-pre-training dataset")
    parser.add_argument("--num-tokens", type=int, required=True, help="Target number of tokens to generate")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length per sample")
    parser.add_argument("--alphabet-size", type=int, default=2, choices=[2, 4, 10], help="NCA alphabet size")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--min-gzip-ratio", type=float, default=0.50, help="Minimum gzip compression ratio")
    parser.add_argument("--trajectories-per-rule", type=int, default=128,
                        help="Trajectories per rule (higher=faster, lower=more rule diversity)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device for simulation (auto picks cuda if available)")
    args = parser.parse_args()

    generate_dataset(
        num_tokens=args.num_tokens,
        seq_len=args.seq_len,
        alphabet_size=args.alphabet_size,
        output_dir=args.output,
        min_gzip_ratio=args.min_gzip_ratio,
        trajectories_per_rule=args.trajectories_per_rule,
        device=args.device,
    )
