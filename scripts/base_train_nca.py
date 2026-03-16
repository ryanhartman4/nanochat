"""
NCA pre-pre-training helpers for base_train.py.

Provides layer swap, training loop, and transfer protocol functions.
Kept separate for testability — called from base_train.py.
"""

import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from nanochat.common import print0, COMPUTE_DTYPE
from nanochat.gpt import Linear  # custom Linear that casts weights to match input dtype


def swap_to_nca_layers(model, nca_vocab_size):
    """Swap text embedding/head layers with NCA-sized temporary layers.

    Returns a dict of saved originals needed for restore_text_layers().
    """
    device = model.get_device()
    n_embd = model.config.n_embd

    # Save originals
    saved = {
        'wte': model.transformer.wte,
        'lm_head': model.lm_head,
        'value_embeds': {k: v for k, v in model.value_embeds.items()},
        'vocab_size': model.config.vocab_size,
    }

    # Pad NCA vocab to multiple of 64, matching GPT.__init__ convention
    # The model's forward() slices logits to config.vocab_size, so the lm_head
    # and wte must use padded size while config.vocab_size stays at the true NCA vocab.
    padded_nca_vocab = ((nca_vocab_size + 63) // 64) * 64

    # Create NCA-sized layers (use custom Linear for dtype-casting consistency)
    nca_wte = nn.Embedding(padded_nca_vocab, n_embd).to(device)
    nca_head = Linear(n_embd, padded_nca_vocab, bias=False).to(device)
    torch.nn.init.normal_(nca_wte.weight, mean=0.0, std=0.8)
    torch.nn.init.normal_(nca_head.weight, mean=0.0, std=0.001)

    # Cast to compute dtype if needed (matching init_weights behavior)
    if COMPUTE_DTYPE != torch.float16:
        nca_wte.to(dtype=COMPUTE_DTYPE)

    # Swap into model
    model.transformer.wte = nca_wte
    model.lm_head = nca_head
    # Set vocab_size to the TRUE NCA vocab (not padded). GPT.forward() slices logits
    # to config.vocab_size — this ensures cross-entropy targets a 16-way softmax (n=2),
    # not a 64-way softmax with 48 phantom classes.
    model.config.vocab_size = nca_vocab_size

    # Swap value_embeds to NCA-sized
    head_dim = n_embd // model.config.n_head
    kv_dim = model.config.n_kv_head * head_dim
    for key in list(model.value_embeds.keys()):
        nca_ve = nn.Embedding(padded_nca_vocab, kv_dim).to(device)
        s = 3**0.5 * n_embd**-0.5
        torch.nn.init.uniform_(nca_ve.weight, -s, s)
        if COMPUTE_DTYPE != torch.float16:
            nca_ve.to(dtype=COMPUTE_DTYPE)
        model.value_embeds[key] = nca_ve

    return saved


def restore_text_layers(model, saved):
    """Restore original text-sized layers from saved dict."""
    model.transformer.wte = saved['wte']
    model.lm_head = saved['lm_head']
    for k, v in saved['value_embeds'].items():
        model.value_embeds[k] = v
    model.config.vocab_size = saved['vocab_size']


def transfer_nca_to_text(model, saved, ddp=False):
    """Execute NCA -> text transfer protocol.

    Following Han et al. 2026: keep all transformer weights (attention + MLP + layernorm),
    reinit only the vocab-dependent layers (embeddings, value_embeds, scalars).

    1. Deep copy all transformer block weights (attention + MLP)
    2. All-reduce across ranks (if DDP)
    3. Restore text-sized modules (embeddings, value_embeds)
    4. Full reinit (resets everything to text-sized defaults)
    5. Load NCA-trained transformer weights back (attention + MLP + layernorm)
    """
    # 1. Deep copy all transformer block weights — attention AND MLP AND layernorm
    # Paper ablation (Fig 5): attention is the primary transfer mechanism, but keeping
    # MLP/layernorm is the paper's default config that produced headline results.
    block_state = {k: v.clone() for k, v in model.state_dict().items()
                   if '.attn.' in k or '.mlp.' in k}

    # 2. Average weights across ranks
    if ddp:
        for v in block_state.values():
            dist.all_reduce(v, op=dist.ReduceOp.AVG)

    # 3. Restore text-sized modules
    restore_text_layers(model, saved)

    # 4. Full reinit (operates on text-sized modules now)
    model.init_weights()

    # 5. Restore NCA-trained transformer weights (attention + MLP)
    # init_weights() already set layernorm to correct defaults (weight=1, bias=0),
    # and nanochat uses parameterless RMSNorm, so no layernorm state to transfer.
    model.load_state_dict(block_state, strict=False)


def run_nca_stage(model, nca_data_path, nca_steps, nca_lr, nca_batch_size,
                  seq_len, alphabet_size, ddp, ddp_rank, ddp_world_size, device, wandb_run):
    """Run the full NCA pre-pre-training stage.

    Args:
        model: GPT model (uncompiled, pre-FP8)
        nca_data_path: Path to NCA dataset directory
        nca_steps: Number of NCA training steps
        nca_lr: Learning rate for NCA AdamW
        nca_batch_size: Per-device batch size
        seq_len: Sequence length (should match NCA data)
        alphabet_size: NCA alphabet size (for computing vocab)
        ddp: Whether distributed training is active
        ddp_rank: Current rank
        ddp_world_size: World size
        device: Target device
        wandb_run: Wandb run for logging
    """
    nca_vocab_size = alphabet_size ** 4
    print0(f"NCA pre-pre-training: {nca_steps} steps, vocab={nca_vocab_size}, lr={nca_lr}")

    # Load NCA dataset
    data_path = os.path.join(nca_data_path, "nca_data.pt")
    nca_data = torch.load(data_path, weights_only=True).to(device)
    # Truncate sequences to model's sequence length (NCA data may be longer)
    if nca_data.shape[1] > seq_len:
        nca_data = nca_data[:, :seq_len]
    num_sequences = nca_data.shape[0]
    # Clamp batch size to available sequences
    nca_batch_size = min(nca_batch_size, num_sequences)
    print0(f"Loaded NCA data: {num_sequences} sequences, shape {nca_data.shape}")

    # Swap to NCA layers
    saved = swap_to_nca_layers(model, nca_vocab_size)

    # Create NCA optimizer (plain AdamW over all parameters)
    nca_optimizer = torch.optim.AdamW(model.parameters(), lr=nca_lr, betas=(0.9, 0.999))

    # NCA training loop
    warmup_steps = max(1, nca_steps // 10)
    model.train()

    for step in range(nca_steps):
        # LR schedule: linear warmup, then constant
        lr_mult = min(1.0, (step + 1) / warmup_steps)
        for pg in nca_optimizer.param_groups:
            pg['lr'] = nca_lr * lr_mult

        # Get batch: shard across ranks via offset + stride
        batch_start = (step * nca_batch_size * ddp_world_size + ddp_rank * nca_batch_size) % num_sequences
        indices = [(batch_start + i) % num_sequences for i in range(nca_batch_size)]
        x = nca_data[indices]  # (B, seq_len)
        # Next-token prediction: input is x[:, :-1], target is x[:, 1:]
        # .contiguous() needed because slicing creates non-contiguous views
        # and GPT.forward() uses .view() which requires contiguity
        inputs = x[:, :-1].contiguous()
        targets = x[:, 1:].contiguous()

        loss = model(inputs, targets)
        loss.backward()
        nca_optimizer.step()
        nca_optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0:
            loss_val = loss.item()
            print0(f"NCA step {step:04d}/{nca_steps} | loss: {loss_val:.4f} | lr: {nca_lr * lr_mult:.6f}")
            wandb_run.log({"nca/loss": loss_val, "nca/step": step})

    # Transfer: keep attention, reinit rest
    print0("NCA transfer: preserving attention weights, reinitializing everything else")
    transfer_nca_to_text(model, saved, ddp=ddp)

    # Cleanup
    del nca_optimizer, nca_data
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print0("NCA pre-pre-training complete")
