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
    # Place a single active cell at (0,0) — with periodic boundaries,
    # its neighbors include cells at (11,11), (11,0), (0,11), etc.
    grid = torch.zeros(1, alphabet_size, 12, 12)
    grid[0, 1, 0, 0] = 1.0  # cell (0,0) in state 1
    logits = rule(grid)
    # Corner neighbor (11, 11) should see non-zero logits due to periodic padding
    assert not torch.allclose(logits[0, :, 11, 11], torch.zeros(alphabet_size)), \
        "Periodic boundary not working: corner neighbor sees zero logits"


def test_simulate_trajectory_shape():
    """Simulate should produce a sequence of grid states."""
    alphabet_size = 2
    rule = create_nca_rule(alphabet_size)
    num_steps = 10
    grids = simulate_trajectory(rule, alphabet_size, grid_size=12, num_steps=num_steps)
    # Should return (num_steps+1, alphabet_size, 12, 12) — initial state + num_steps updates
    assert grids.shape == (num_steps + 1, alphabet_size, 12, 12)


def test_simulate_trajectory_stochastic():
    """With tau=1e-3, simulation should be near-deterministic but not exactly."""
    alphabet_size = 2
    rule = create_nca_rule(alphabet_size)
    torch.manual_seed(42)
    g1 = simulate_trajectory(rule, alphabet_size, grid_size=12, num_steps=5, tau=1e-3)
    torch.manual_seed(43)
    g2 = simulate_trajectory(rule, alphabet_size, grid_size=12, num_steps=5, tau=1e-3)
    # Same rule, different seeds — should produce different trajectories (stochastic)
    assert not torch.equal(g1, g2), "Identical trajectories with different seeds — simulation may not be stochastic"


def test_tokenize_grid_shape():
    """Tokenize a grid trajectory into a flat token sequence."""
    alphabet_size = 2
    num_steps = 56  # 57 grids total (initial + 56 steps)
    # Create fake trajectory: (57, alphabet, 12, 12)
    grids = torch.zeros(num_steps + 1, alphabet_size, 12, 12)
    grids[:, 0, :, :] = 1.0  # all cells in state 0
    tokens = tokenize_trajectory(grids, alphabet_size)
    # 6x6 patches per grid * 57 grids = 2052 tokens, but we need seq_len tokens
    # The function should return a flat 1D tensor
    assert tokens.dim() == 1
    assert tokens.shape[0] == 57 * 36  # 57 grids * 36 patches each = 2052


def test_tokenize_grid_vocab_range():
    """Token IDs should be in [0, alphabet^4)."""
    alphabet_size = 2
    nca_vocab_size = alphabet_size ** 4  # 16
    grids = torch.zeros(57, alphabet_size, 12, 12)
    grids[:, 0, :, :] = 1.0
    tokens = tokenize_trajectory(grids, alphabet_size)
    assert tokens.min() >= 0, f"Negative token ID: {tokens.min()}"
    assert tokens.max() < nca_vocab_size, f"Token {tokens.max()} >= vocab size {nca_vocab_size}"


def test_tokenize_bijective():
    """Different 2x2 patches should map to different token IDs."""
    alphabet_size = 2
    # Create two grids with different cell states at patch (0,0)
    g1 = torch.zeros(1, alphabet_size, 12, 12)
    g1[0, 0, :, :] = 1.0  # all state 0
    g2 = g1.clone()
    g2[0, 0, 0, 0] = 0.0; g2[0, 1, 0, 0] = 1.0  # cell (0,0) changed to state 1
    t1 = tokenize_trajectory(g1, alphabet_size)
    t2 = tokenize_trajectory(g2, alphabet_size)
    # First token (patch at 0,0) should differ
    assert t1[0] != t2[0], "Different patches mapped to same token — not bijective"


def test_gzip_filter_rejects_trivial():
    """Constant trajectory (all same state) should have low compression ratio and be rejected."""
    tokens = torch.zeros(2048, dtype=torch.long)  # all zeros — trivially compressible
    assert not passes_complexity_filter(tokens, min_ratio=0.50), \
        "Constant sequence should be rejected (high compressibility = low ratio)"


def test_gzip_filter_accepts_complex():
    """Random-looking tokens should have high compression ratio and pass."""
    # Create a pseudo-structured sequence (not truly random, but varied)
    torch.manual_seed(42)
    tokens = torch.randint(0, 16, (2048,))
    ratio = gzip_compression_ratio(tokens)
    # Random data typically has ratio > 0.8
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
        )
        # Verify output file exists
        output_path = os.path.join(tmpdir, "nca_data.pt")
        assert os.path.exists(output_path), f"Output file not found at {output_path}"

        # Load and verify
        data = torch.load(output_path, weights_only=True)
        assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D"
        assert data.shape[1] == 2048, f"Expected seq_len=2048, got {data.shape[1]}"
        nca_vocab = 2 ** 4  # alphabet_size^4
        assert data.min() >= 0 and data.max() < nca_vocab, \
            f"Token range [{data.min()}, {data.max()}] outside [0, {nca_vocab})"
        # Should have at least 1 sequence (4096 / 2048 = 2 sequences minimum)
        assert data.shape[0] >= 1, f"Expected at least 1 sequence, got {data.shape[0]}"
