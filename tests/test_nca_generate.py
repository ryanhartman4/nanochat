"""Tests for NCA data generator. Run: python -m pytest tests/test_nca_generate.py -v"""
import torch
from scripts.nca_generate import create_nca_rule


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
