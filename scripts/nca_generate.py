"""
NCA Data Generator for pre-pre-training.

Generates synthetic Neural Cellular Automata trajectories as a pre-training dataset.
Based on Han et al. 2026 (arXiv:2603.10055).

Two modes:
  Epoch mode (recommended):
    python -m scripts.nca_generate --num-rules 1000 --num-epochs 100 --output /path/to/output

  Legacy token-count mode:
    python -m scripts.nca_generate --num-tokens 164000000 --output /path/to/output
"""

import os
import json
import gzip
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class NCARule(nn.Module):
    """NCA transition rule matching paper architecture (Han et al. 2026).

    3-layer design with 4-channel spatial bottleneck:
      Conv(n_states→4, 3x3) → Conv(4→16, 1x1) → ReLU → Conv(16→n_states, 1x1)

    The bottleneck forces rules to express neighborhood information through
    4 channels, biasing toward simpler, more structured dynamics.
    """
    def __init__(self, alphabet_size, hidden_size=16):
        super().__init__()
        self.conv1 = nn.Conv2d(alphabet_size, 4, kernel_size=3, padding=0, bias=False)
        self.conv2 = nn.Conv2d(4, hidden_size, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(hidden_size, alphabet_size, kernel_size=1, bias=False)
        for p in self.parameters():
            nn.init.normal_(p, std=0.5)

    def forward(self, grid):
        padded = F.pad(grid, (1, 1, 1, 1), mode='circular')
        x = self.conv1(padded)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x


def create_nca_rule(alphabet_size, hidden_size=16, device="cpu"):
    """Create a random NCA transition rule as a small neural network."""
    return NCARule(alphabet_size, hidden_size).to(device)


@torch.no_grad()
def simulate_trajectory(rule, alphabet_size, grid_size=12, num_steps=56, tau=1e-3,
                        batch_size=1, device="cpu"):
    """Simulate NCA trajectories from random initial states.

    Supports batched simulation: multiple trajectories with the same rule
    run in parallel via batched conv2d.

    Returns:
        Tensor of shape (batch_size, num_steps+1, alphabet_size, grid_size, grid_size)
    """
    B = batch_size
    state_indices = torch.randint(0, alphabet_size, (B, grid_size, grid_size), device=device)
    grid = F.one_hot(state_indices, alphabet_size).permute(0, 3, 1, 2).float()

    grids = [grid]
    for _ in range(num_steps):
        logits = rule(grid)
        probs = F.softmax(logits / tau, dim=1)
        flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, alphabet_size)
        samples = torch.multinomial(flat_probs, 1).squeeze(-1)
        grid = F.one_hot(samples, alphabet_size).float().reshape(B, grid_size, grid_size, alphabet_size)
        grid = grid.permute(0, 3, 1, 2)
        grids.append(grid)

    return torch.stack(grids, dim=1)


def tokenize_trajectory(grids, alphabet_size):
    """Convert one-hot grids into flat token sequences using 2x2 patches.

    Each grid frame is wrapped with delimiter tokens:
      [START, patch_0, ..., patch_35, END]  (38 tokens per grid for 12x12)
    Patch IDs are in [0, alphabet_size^4), START = alphabet_size^4, END = alphabet_size^4 + 1.
    """
    batched = grids.dim() == 5
    if not batched:
        grids = grids.unsqueeze(0)

    B, T, n, H, W = grids.shape
    assert H % 2 == 0 and W % 2 == 0, f"Grid {H}x{W} must be even for 2x2 patches"

    cell_states = grids.argmax(dim=2)
    patches = cell_states.reshape(B, T, H // 2, 2, W // 2, 2)
    c0 = patches[:, :, :, 0, :, 0]
    c1 = patches[:, :, :, 0, :, 1]
    c2 = patches[:, :, :, 1, :, 0]
    c3 = patches[:, :, :, 1, :, 1]

    token_ids = c0 * (alphabet_size**3) + c1 * (alphabet_size**2) + c2 * alphabet_size + c3
    patches_per_grid = (H // 2) * (W // 2)
    token_ids = token_ids.reshape(B, T, patches_per_grid).long()

    # Wrap each grid frame with START/END delimiter tokens
    START_TOKEN = alphabet_size ** 4
    END_TOKEN = alphabet_size ** 4 + 1
    start_col = torch.full((B, T, 1), START_TOKEN, dtype=token_ids.dtype, device=token_ids.device)
    end_col = torch.full((B, T, 1), END_TOKEN, dtype=token_ids.dtype, device=token_ids.device)
    token_ids = torch.cat([start_col, token_ids, end_col], dim=2)

    token_ids = token_ids.reshape(B, -1)

    if not batched:
        return token_ids.squeeze(0)
    return token_ids


def gzip_compression_ratio(tokens):
    """Compute gzip compression ratio for a token sequence.

    Returns r = compressed_size / raw_size. Lower r = more compressible = simpler.
    """
    if tokens.is_cuda:
        tokens = tokens.cpu()
    np_tokens = tokens.numpy()
    if np_tokens.max() < 256:
        raw_bytes = np_tokens.astype('uint8').tobytes()
    else:
        raw_bytes = np_tokens.astype('<u2').tobytes()
    compressed = gzip.compress(raw_bytes)
    return len(compressed) / len(raw_bytes)


def passes_complexity_filter(tokens, min_ratio=0.50, max_ratio=1.0):
    """Return True if the token sequence falls within the target complexity band.

    Gzip ratio is a tunable knob, not a fixed threshold. The paper (Han et al. 2026,
    Section 5.2) shows optimal NCA complexity is domain-dependent:
      - Code:      benefits from simpler dynamics (30-40% gzip band)
      - Math/text: benefits from higher complexity  (50%+ gzip band)
    The reference repo exposes both lower and upper bounds for this reason.
    Tuning this band is a primary lever for domain-targeted NCA pre-pre-training.
    """
    r = gzip_compression_ratio(tokens)
    return r > min_ratio and r <= max_ratio


def _resolve_device(device_str):
    """Resolve device string: 'auto' picks cuda if available, else cpu."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _compute_seq_params(seq_len, grid_size=12):
    """Compute NCA simulation parameters from sequence length."""
    patches_per_grid = (grid_size // 2) ** 2  # 36 for 12x12
    grid_len = patches_per_grid + 2  # +2 for START/END delimiters
    grids_per_seq = seq_len // grid_len
    steps_per_seq = grids_per_seq - 1
    return patches_per_grid, grids_per_seq, steps_per_seq, grid_len


def build_rule_pool(num_rules, alphabet_size, seq_len, min_gzip_ratio=0.50,
                    max_gzip_ratio=1.0, grid_size=12, device="cpu"):
    """Create a pool of complexity-filtered NCA rules.

    Each rule is a small random neural network that defines unique dynamics.
    Rules are filtered by generating a test trajectory and checking gzip complexity
    falls within [min_gzip_ratio, max_gzip_ratio].

    Returns:
        List of NCARule modules, steps_per_seq
    """
    _, _, steps_per_seq, _ = _compute_seq_params(seq_len, grid_size)

    rules = []
    rejected = 0
    while len(rules) < num_rules:
        rule = create_nca_rule(alphabet_size, device=device)
        # Quick complexity check with one trajectory
        grids = simulate_trajectory(rule, alphabet_size, grid_size=grid_size,
                                    num_steps=steps_per_seq, batch_size=1, device=device)
        tokens = tokenize_trajectory(grids, alphabet_size)[0][:seq_len]
        if passes_complexity_filter(tokens, min_ratio=min_gzip_ratio, max_ratio=max_gzip_ratio):
            rules.append(rule)
        else:
            rejected += 1
        if (len(rules) + rejected) % 100 == 0:
            print(f"  Rule pool: {len(rules)}/{num_rules} accepted, {rejected} rejected")

    print(f"Built rule pool: {num_rules} rules ({rejected} rejected by gzip filter, "
          f"band=[{min_gzip_ratio:.2f}, {max_gzip_ratio:.2f}])")
    return rules, steps_per_seq


def generate_epoch_dataset(num_rules, num_epochs, seq_len, alphabet_size, output_dir,
                           min_gzip_ratio=0.50, max_gzip_ratio=1.0, grid_size=12, device="auto"):
    """Generate a structured NCA dataset: num_epochs fresh trajectories per rule.

    Creates a fixed pool of rules, then generates num_epochs different trajectories
    from each rule (different random initial states each time). This gives the model
    fresh data each epoch while maintaining rule diversity.

    Output:
        nca_data.pt: tensor of shape (num_epochs * num_rules, seq_len)
        nca_meta.json: {num_rules, num_epochs, seq_len, alphabet_size}

    During training, epoch k = data[k*num_rules : (k+1)*num_rules].
    """
    device = _resolve_device(device) if isinstance(device, str) else device
    print(f"NCA epoch generation on device: {device}")
    print(f"Config: {num_rules} rules × {num_epochs} epochs = {num_rules * num_epochs} sequences")

    # Build complexity-filtered rule pool
    rules, steps_per_seq = build_rule_pool(
        num_rules, alphabet_size, seq_len, min_gzip_ratio, max_gzip_ratio, grid_size, device)

    all_sequences = []
    epoch_rejects = 0
    for epoch in range(num_epochs):
        epoch_seqs = []
        for rule in rules:
            # Fresh trajectory: new random initial state each time
            # Retry if this trajectory fails gzip filter (rule passed but init state may be degenerate)
            for _attempt in range(5):
                grids = simulate_trajectory(rule, alphabet_size, grid_size=grid_size,
                                            num_steps=steps_per_seq, batch_size=1, device=device)
                tokens = tokenize_trajectory(grids, alphabet_size)[0][:seq_len]
                if passes_complexity_filter(tokens, min_ratio=min_gzip_ratio, max_ratio=max_gzip_ratio):
                    break
                epoch_rejects += 1
            # Pad if needed
            if tokens.shape[0] < seq_len:
                tokens = F.pad(tokens, (0, seq_len - tokens.shape[0]), value=0)
            epoch_seqs.append(tokens.cpu())
        all_sequences.extend(epoch_seqs)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} generated ({len(all_sequences)} total sequences)")

    dataset = torch.stack(all_sequences)  # (num_epochs * num_rules, seq_len)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(dataset, os.path.join(output_dir, "nca_data.pt"))
    _, _, _, grid_len = _compute_seq_params(seq_len, grid_size)
    meta = {"num_rules": num_rules, "num_epochs": num_epochs,
            "seq_len": seq_len, "alphabet_size": alphabet_size, "grid_len": grid_len}
    with open(os.path.join(output_dir, "nca_meta.json"), "w") as f:
        json.dump(meta, f)

    total_tokens = dataset.shape[0] * dataset.shape[1]
    print(f"Generated {dataset.shape[0]} sequences ({total_tokens/1e6:.0f}M tokens), "
          f"saved to {output_dir} ({epoch_rejects} per-epoch gzip retries)")
    print(f"Data size: {dataset.element_size() * dataset.nelement() / 1e6:.0f}MB")


def generate_dataset(num_tokens, seq_len, alphabet_size, output_dir, min_gzip_ratio=0.50,
                     max_gzip_ratio=1.0, grid_size=12, trajectories_per_rule=128, device="auto"):
    """Legacy: Generate NCA dataset by token count (old interface)."""
    device = _resolve_device(device) if isinstance(device, str) else device
    print(f"NCA generation on device: {device}")

    patches_per_grid, grids_per_seq, steps_per_seq, _ = _compute_seq_params(seq_len, grid_size)
    num_sequences = (num_tokens + seq_len - 1) // seq_len

    all_sequences = []
    generated = 0
    rejected = 0
    num_rules = 0

    while generated < num_sequences:
        remaining = num_sequences - generated
        batch_size = min(trajectories_per_rule, remaining * 2)
        rule = create_nca_rule(alphabet_size, device=device)
        num_rules += 1
        grids = simulate_trajectory(
            rule, alphabet_size, grid_size=grid_size,
            num_steps=steps_per_seq, batch_size=batch_size, device=device,
        )
        tokens_batch = tokenize_trajectory(grids, alphabet_size)
        raw_len = tokens_batch.shape[1]
        if raw_len >= seq_len:
            tokens_batch = tokens_batch[:, :seq_len]
        else:
            tokens_batch = F.pad(tokens_batch, (0, seq_len - raw_len), value=0)
        tokens_cpu = tokens_batch.cpu()
        for i in range(tokens_cpu.shape[0]):
            if generated >= num_sequences:
                break
            seq = tokens_cpu[i]
            if passes_complexity_filter(seq, min_ratio=min_gzip_ratio, max_ratio=max_gzip_ratio):
                all_sequences.append(seq)
                generated += 1
            else:
                rejected += 1
        if num_rules % 10 == 0:
            print(f"Progress: {generated}/{num_sequences} sequences "
                  f"({num_rules} rules, {rejected} rejected)")

    dataset = torch.stack(all_sequences)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "nca_data.pt")
    torch.save(dataset, output_path)
    print(f"Generated {generated} sequences from {num_rules} rules "
          f"({rejected} rejected by filter), saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NCA pre-pre-training dataset")
    # Epoch mode (recommended)
    parser.add_argument("--num-rules", type=int, default=16000, help="Number of unique NCA rules (epoch mode, default 16000)")
    parser.add_argument("--num-epochs", type=int, default=100, help="Trajectories per rule = training epochs")
    # Legacy token-count mode
    parser.add_argument("--num-tokens", type=int, default=0, help="Target tokens (legacy mode, use --num-rules instead)")
    # Shared
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length per sample")
    parser.add_argument("--alphabet-size", type=int, default=10, choices=[2, 4, 10], help="NCA alphabet size")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--min-gzip-ratio", type=float, default=0.50,
                        help="Minimum gzip compression ratio (lower bound of complexity band)")
    parser.add_argument("--max-gzip-ratio", type=float, default=1.0,
                        help="Maximum gzip compression ratio (upper bound; 1.0 = no upper limit). "
                             "Paper shows optimal band is domain-dependent: 0.30-0.40 for code, 0.50+ for math/text")
    parser.add_argument("--trajectories-per-rule", type=int, default=128,
                        help="Trajectories per rule (legacy mode only)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device for simulation")
    args = parser.parse_args()

    if args.num_tokens > 0:
        # Legacy mode (explicit --num-tokens overrides default --num-rules)
        generate_dataset(
            num_tokens=args.num_tokens,
            seq_len=args.seq_len,
            alphabet_size=args.alphabet_size,
            output_dir=args.output,
            min_gzip_ratio=args.min_gzip_ratio,
            max_gzip_ratio=args.max_gzip_ratio,
            trajectories_per_rule=args.trajectories_per_rule,
            device=args.device,
        )
    elif args.num_rules > 0:
        # Epoch mode (default: 16000 rules)
        generate_epoch_dataset(
            num_rules=args.num_rules,
            num_epochs=args.num_epochs,
            seq_len=args.seq_len,
            alphabet_size=args.alphabet_size,
            output_dir=args.output,
            min_gzip_ratio=args.min_gzip_ratio,
            max_gzip_ratio=args.max_gzip_ratio,
            device=args.device,
        )
    else:
        parser.error("Specify either --num-rules (epoch mode) or --num-tokens (legacy mode)")
