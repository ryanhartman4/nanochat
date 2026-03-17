"""
Integration test for NCA stage in base_train.

Tests the NCA training functions in isolation (without running the full training script).
Run: python -m pytest tests/test_nca_integration.py -v
"""
import torch
import torch.nn as nn
from nanochat.gpt import GPT, GPTConfig


def _build_tiny_model():
    """Build a tiny model for testing (depth=4, small dims)."""
    config = GPTConfig(sequence_len=512, vocab_size=32, n_layer=4, n_head=2, n_kv_head=2, n_embd=64)
    with torch.device("meta"):
        model = GPT(config, pad_vocab_size_to=1)  # no padding for test simplicity
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def test_nca_layer_swap_and_restore():
    """Verify temporary NCA layers can be swapped in and originals restored."""
    model = _build_tiny_model()
    orig_vocab = model.config.vocab_size
    orig_wte_shape = model.transformer.wte.weight.shape
    orig_head_shape = model.lm_head.weight.shape

    # Import the NCA functions (to be implemented)
    from scripts.base_train_nca import swap_to_nca_layers, restore_text_layers

    nca_vocab_size = 16
    saved = swap_to_nca_layers(model, nca_vocab_size)

    # config.vocab_size should be the TRUE NCA vocab (for correct logit slicing in forward())
    # wte/lm_head are padded to multiple of 64 for tensor core alignment (GPT convention)
    padded_nca_vocab = ((nca_vocab_size + 63) // 64) * 64
    assert model.config.vocab_size == nca_vocab_size, \
        f"config.vocab_size should be {nca_vocab_size} (true vocab), got {model.config.vocab_size}"
    assert model.transformer.wte.weight.shape[0] == padded_nca_vocab
    assert model.lm_head.weight.shape[0] == padded_nca_vocab

    # Restore
    restore_text_layers(model, saved)
    assert model.config.vocab_size == orig_vocab
    assert model.transformer.wte.weight.shape == orig_wte_shape
    assert model.lm_head.weight.shape == orig_head_shape


def test_nca_transfer_preserves_transformer_blocks():
    """After transfer, attention + MLP weights should be NCA-trained, embeddings reinitialized.

    Paper (Han et al. 2026): keep all transformer weights (attention + MLP),
    reinit only vocab-dependent layers (embeddings, value_embeds, scalars).
    """
    model = _build_tiny_model()

    from scripts.base_train_nca import swap_to_nca_layers, transfer_nca_to_text

    nca_vocab_size = 16
    saved = swap_to_nca_layers(model, nca_vocab_size)

    # Manually perturb ALL weights to simulate NCA training
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.add_(1.0)

    # Snapshot NCA-trained weights before transfer
    nca_attn = {k: v.clone() for k, v in model.state_dict().items() if '.attn.' in k}
    nca_mlp = {k: v.clone() for k, v in model.state_dict().items() if '.mlp.' in k}
    SCALAR_KEYS = {'resid_lambdas', 'x0_lambdas', 'smear_gate.weight', 'smear_lambda', 'backout_lambda'}
    nca_scalars = {k: v.clone() for k, v in model.state_dict().items() if k in SCALAR_KEYS}

    # Transfer
    transfer_nca_to_text(model, saved, ddp=False)

    # Attention weights should be preserved
    for k in nca_attn:
        assert torch.allclose(model.state_dict()[k], nca_attn[k]), \
            f"Attention weight {k} was not preserved during transfer"

    # MLP weights should also be preserved (paper keeps all transformer blocks)
    for k in nca_mlp:
        assert torch.allclose(model.state_dict()[k], nca_mlp[k]), \
            f"MLP weight {k} was not preserved during transfer"

    # Scalars should be preserved (co-adapted with attention during NCA training)
    for k in nca_scalars:
        assert torch.allclose(model.state_dict()[k], nca_scalars[k]), \
            f"Scalar {k} was not preserved during transfer"

    # Embeddings should be reinitialized (different from NCA-trained values)
    assert model.transformer.wte.weight.shape == saved['wte'].weight.shape, \
        "Embedding shape not restored to text size"
