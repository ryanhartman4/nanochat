"""Tests for NCA data generator. Run: python -m pytest tests/test_nca_generate.py -v"""
import torch
from scripts.nca_generate import create_nca_rule, simulate_trajectory


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
