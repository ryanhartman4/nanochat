"""Tests for NCA data generator. Run: python -m pytest tests/test_nca_generate.py -v"""
import torch
from scripts.nca_generate import create_nca_rule, simulate_trajectory, tokenize_trajectory, passes_complexity_filter, gzip_compression_ratio, generate_dataset


def test_create_nca_rule_output_shape():
    """NCA rule should map (B, alphabet, 12, 12) -> (B, alphabet, 12, 12) logits."""
    alphabet_size = 2
    rule = create_nca_rule(alphabet_size)
    grid = torch.zeros(1, alphabet_size, 12, 12)
    grid[0, 0, :, :] = 1.0  # all cells in state 0
    logits = rule(grid)
    assert logits.shape == (1, alphabet_size, 12, 12), f"Expected (1, {alphabet_size}, 12, 12), got {logits.shape}"


def test_create_nca_rule_periodic_boundaries():
    """Corner cells should receive neighbor info from opposite edges (toroidal)."""
    alphabet_size = 2
    rule = create_nca_rule(alphabet_size)
    grid = torch.zeros(1, alphabet_size, 12, 12)
    grid[0, 1, 0, 0] = 1.0  # cell (0,0) in state 1
    logits = rule(grid)
    assert not torch.allclose(logits[0, :, 11, 11], torch.zeros(alphabet_size)), \
        "Periodic boundary not working: corner neighbor sees zero logits"


def test_simulate_trajectory_shape_single():
    """Single trajectory should produce (1, T+1, alphabet, H, W)."""
    alphabet_size = 2
    rule = create_nca_rule(alphabet_size)
    num_steps = 10
    grids = simulate_trajectory(rule, alphabet_size, grid_size=12, num_steps=num_steps, batch_size=1)
    assert grids.shape == (1, num_steps + 1, alphabet_size, 12, 12)


def test_simulate_trajectory_shape_batched():
    """Batched simulation should produce (B, T+1, alphabet, H, W)."""
    alphabet_size = 2
    rule = create_nca_rule(alphabet_size)
    num_steps = 10
    batch_size = 8
    grids = simulate_trajectory(rule, alphabet_size, grid_size=12, num_steps=num_steps, batch_size=batch_size)
    assert grids.shape == (batch_size, num_steps + 1, alphabet_size, 12, 12)


def test_simulate_trajectory_stochastic():
    """With tau=1e-3, different seeds should produce different trajectories."""
    alphabet_size = 2
    rule = create_nca_rule(alphabet_size)
    torch.manual_seed(42)
    g1 = simulate_trajectory(rule, alphabet_size, grid_size=12, num_steps=5, tau=1e-3)
    torch.manual_seed(43)
    g2 = simulate_trajectory(rule, alphabet_size, grid_size=12, num_steps=5, tau=1e-3)
    assert not torch.equal(g1, g2), "Identical trajectories with different seeds"


def test_tokenize_single_shape():
    """Single trajectory tokenization produces 1D output with delimiters."""
    alphabet_size = 2
    num_grids = 56
    grids = torch.zeros(num_grids, alphabet_size, 12, 12)
    grids[:, 0, :, :] = 1.0
    tokens = tokenize_trajectory(grids, alphabet_size)
    assert tokens.dim() == 1
    assert tokens.shape[0] == 56 * 38  # 56 grids * (36 patches + 2 delimiters) = 2128


def test_tokenize_batched_shape():
    """Batched tokenization produces (B, tokens) output with delimiters."""
    alphabet_size = 2
    B = 4
    num_grids = 56
    grids = torch.zeros(B, num_grids, alphabet_size, 12, 12)
    grids[:, :, 0, :, :] = 1.0
    tokens = tokenize_trajectory(grids, alphabet_size)
    assert tokens.dim() == 2
    assert tokens.shape == (B, 56 * 38)


def test_tokenize_grid_vocab_range():
    """Token IDs should be in [0, alphabet^4 + 2) including START/END delimiters."""
    alphabet_size = 2
    nca_vocab_size = alphabet_size ** 4 + 2  # 16 patches + START + END = 18
    grids = torch.zeros(56, alphabet_size, 12, 12)
    grids[:, 0, :, :] = 1.0
    tokens = tokenize_trajectory(grids, alphabet_size)
    assert tokens.min() >= 0, f"Negative token ID: {tokens.min()}"
    assert tokens.max() < nca_vocab_size, f"Token {tokens.max()} >= vocab size {nca_vocab_size}"


def test_tokenize_bijective():
    """Different 2x2 patches should map to different token IDs."""
    alphabet_size = 2
    g1 = torch.zeros(1, alphabet_size, 12, 12)
    g1[0, 0, :, :] = 1.0
    g2 = g1.clone()
    g2[0, 0, 0, 0] = 0.0; g2[0, 1, 0, 0] = 1.0
    t1 = tokenize_trajectory(g1, alphabet_size)
    t2 = tokenize_trajectory(g2, alphabet_size)
    # t1[0] and t2[0] are both START delimiter; compare first patch token at index 1
    assert t1[0] == alphabet_size ** 4, "First token should be START delimiter"
    assert t1[1] != t2[1], "Different patches mapped to same token — not bijective"


def test_gzip_filter_rejects_trivial():
    """Constant trajectory should be rejected."""
    tokens = torch.zeros(2048, dtype=torch.long)
    assert not passes_complexity_filter(tokens, min_ratio=0.50), \
        "Constant sequence should be rejected"


def test_gzip_filter_accepts_complex():
    """Random-looking tokens should pass filter."""
    torch.manual_seed(42)
    tokens = torch.randint(0, 16, (2048,))
    ratio = gzip_compression_ratio(tokens)
    assert ratio > 0.50, f"Random tokens should pass filter, got ratio={ratio:.3f}"
    assert passes_complexity_filter(tokens, min_ratio=0.50)


import os
import tempfile


def test_generate_dataset_produces_valid_file():
    """Full pipeline: generate NCA dataset, save, and verify."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_dataset(
            num_tokens=4096,
            seq_len=2048,
            alphabet_size=2,
            output_dir=tmpdir,
            min_gzip_ratio=0.50,
            device="cpu",
        )
        output_path = os.path.join(tmpdir, "nca_data.pt")
        assert os.path.exists(output_path), f"Output file not found at {output_path}"

        data = torch.load(output_path, weights_only=True)
        assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D"
        assert data.shape[1] == 2048, f"Expected seq_len=2048, got {data.shape[1]}"
        nca_vocab = 2 ** 4 + 2  # +2 for START/END delimiters
        assert data.min() >= 0 and data.max() < nca_vocab, \
            f"Token range [{data.min()}, {data.max()}] outside [0, {nca_vocab})"
        assert data.shape[0] >= 1, f"Expected at least 1 sequence, got {data.shape[0]}"
