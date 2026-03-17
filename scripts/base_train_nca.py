"""
NCA pre-pre-training helpers for base_train.py.

Provides layer swap, training loop, and transfer protocol functions.
Kept separate for testability — called from base_train.py.
"""

import gc
import json
import math
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
    padded_nca_vocab = ((nca_vocab_size + 63) // 64) * 64

    # Create NCA-sized layers
    nca_wte = nn.Embedding(padded_nca_vocab, n_embd).to(device)
    nca_head = Linear(n_embd, padded_nca_vocab, bias=False).to(device)
    torch.nn.init.normal_(nca_wte.weight, mean=0.0, std=0.8)
    torch.nn.init.normal_(nca_head.weight, mean=0.0, std=0.001)

    if COMPUTE_DTYPE != torch.float16:
        nca_wte.to(dtype=COMPUTE_DTYPE)

    model.transformer.wte = nca_wte
    model.lm_head = nca_head
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


def transfer_nca_to_text(model, saved, ddp=False, transfer_mode="full"):
    """Execute NCA -> text transfer protocol.

    transfer_mode='full' (paper default): keep attention + MLP weights, reinit embeddings/scalars.
    transfer_mode='attn-only': keep only attention weights, reinit MLPs + embeddings/scalars.

    Both modes also preserve nanochat's learnable scalars (resid_lambdas, x0_lambdas,
    smear_gate, smear_lambda, backout_lambda). These are not vocab-dependent — they
    control residual stream scaling and were co-adapted with attention during NCA training.
    Reiniting them creates a mismatch: attention weights expect the NCA-trained scalar
    values, but get the default init values instead, causing a BPB penalty at transfer.
    """
    # Nanochat scalars that are co-adapted with attention weights during NCA training.
    # These are NOT vocab-dependent and should be preserved across transfer.
    SCALAR_KEYS = {'resid_lambdas', 'x0_lambdas', 'smear_gate.weight', 'smear_lambda', 'backout_lambda'}

    # Deep copy transformer block weights based on transfer mode
    if transfer_mode == "attn-only":
        block_state = {k: v.clone() for k, v in model.state_dict().items()
                       if '.attn.' in k or k in SCALAR_KEYS}
        print0(f"NCA transfer (attn-only): preserving {len(block_state)} tensors (attention + scalars), reinitializing MLPs")
    else:
        block_state = {k: v.clone() for k, v in model.state_dict().items()
                       if '.attn.' in k or '.mlp.' in k or k in SCALAR_KEYS}
        print0(f"NCA transfer (full): preserving {len(block_state)} tensors (attention + MLP + scalars)")

    # Average weights across ranks
    if ddp:
        for v in block_state.values():
            dist.all_reduce(v, op=dist.ReduceOp.AVG)

    # Restore text-sized modules
    restore_text_layers(model, saved)

    # Full reinit (operates on text-sized modules now)
    model.init_weights()

    # Restore NCA-trained weights (block weights + scalars)
    # nanochat uses parameterless RMSNorm, so no layernorm state to transfer.
    model.load_state_dict(block_state, strict=False)


def run_nca_stage(model, nca_data_path, nca_lr, nca_batch_size,
                  seq_len, alphabet_size, ddp, ddp_rank, ddp_world_size, device, wandb_run,
                  nca_steps=0, transfer_mode="full"):
    """Run the full NCA pre-pre-training stage.

    Loads pre-generated NCA data from disk. If nca_meta.json exists (epoch mode),
    trains for multiple epochs with fresh data each epoch. Otherwise falls back to
    step-based training on a flat dataset.
    """
    nca_vocab_size = alphabet_size ** 4 + 2  # +2 for START/END delimiter tokens

    # Load dataset
    data_path = os.path.join(nca_data_path, "nca_data.pt")
    meta_path = os.path.join(nca_data_path, "nca_meta.json")
    nca_data = torch.load(data_path, weights_only=True).to(device)
    if nca_data.shape[1] > seq_len:
        nca_data = nca_data[:, :seq_len]

    # Determine mode: epoch (has metadata) or legacy (flat dataset + nca_steps)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        num_rules = meta["num_rules"]
        num_epochs = meta["num_epochs"]
        grid_len = meta.get("grid_len", 38)  # 36 patches + 2 delimiters; default for backward compat
        assert nca_data.shape[0] == num_rules * num_epochs, \
            f"Data shape {nca_data.shape[0]} != {num_rules}*{num_epochs}"
        rank_rules = num_rules // max(ddp_world_size, 1)
        steps_per_epoch = rank_rules // nca_batch_size
        total_steps = steps_per_epoch * num_epochs
        print0(f"NCA pre-pre-training (epoch mode): {num_epochs} epochs × {num_rules} rules "
               f"= {total_steps} steps, vocab={nca_vocab_size}, lr={nca_lr}")
    else:
        # Legacy: flat dataset, train for nca_steps
        num_rules = nca_data.shape[0]
        num_epochs = 1
        grid_len = 38  # default: (12//2)^2 + 2 for 12x12 grid with delimiters
        total_steps = nca_steps if nca_steps > 0 else num_rules // nca_batch_size
        steps_per_epoch = total_steps
        print0(f"NCA pre-pre-training (legacy): {total_steps} steps, vocab={nca_vocab_size}, lr={nca_lr}")

    print0(f"Loaded NCA data: {nca_data.shape[0]} sequences, shape {nca_data.shape}")

    # Swap to NCA layers
    saved = swap_to_nca_layers(model, nca_vocab_size)

    # Create NCA optimizer — paper uses Adam with no weight decay
    nca_optimizer = torch.optim.AdamW(model.parameters(), lr=nca_lr, betas=(0.9, 0.999), weight_decay=0.0)
    warmup_steps = max(1, total_steps // 10)
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        # Slice this epoch's data
        epoch_start = epoch * num_rules
        epoch_end = epoch_start + num_rules
        epoch_data = nca_data[epoch_start:epoch_end]

        # Shuffle within epoch — broadcast from rank 0 so all ranks agree
        perm = torch.randperm(epoch_data.shape[0], device=device)
        if ddp:
            dist.broadcast(perm, src=0)
        epoch_data = epoch_data[perm]

        # Shard for DDP: each rank gets a non-overlapping slice
        if ddp:
            rank_size = epoch_data.shape[0] // ddp_world_size
            epoch_data = epoch_data[ddp_rank * rank_size:(ddp_rank + 1) * rank_size]

        for i in range(0, epoch_data.shape[0] - nca_batch_size + 1, nca_batch_size):
            if nca_steps > 0 and global_step >= nca_steps:
                break  # legacy mode: stop at nca_steps

            # LR schedule: linear warmup, then constant
            # LR schedule: 10% linear warmup + cosine decay (matches paper)
            if global_step < warmup_steps:
                lr_mult = (global_step + 1) / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in nca_optimizer.param_groups:
                pg['lr'] = nca_lr * lr_mult

            x = epoch_data[i:i + nca_batch_size]
            inputs = x[:, :-1].contiguous()
            targets = x[:, 1:].contiguous()
            targets[:, :grid_len] = -1  # mask first grid frame (ICL context, not training signal)

            loss = model(inputs, targets)
            loss.backward()
            nca_optimizer.step()
            nca_optimizer.zero_grad(set_to_none=True)

            if global_step % 50 == 0:
                print0(f"NCA epoch {epoch+1:02d}/{num_epochs} step {global_step:05d}/{total_steps} "
                       f"| loss: {loss.item():.4f} | lr: {nca_lr * lr_mult:.6f}")
                wandb_run.log({"nca/loss": loss.item(), "nca/step": global_step, "nca/epoch": epoch})
            global_step += 1

        if nca_steps > 0 and global_step >= nca_steps:
            break

    print0(f"NCA training complete: {global_step} steps across {epoch+1} epochs")

    # Transfer: keep attention + MLP, reinit embeddings/scalars
    transfer_nca_to_text(model, saved, ddp=ddp, transfer_mode=transfer_mode)

    # Cleanup
    del nca_optimizer, nca_data
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print0("NCA pre-pre-training complete")
