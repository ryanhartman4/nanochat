# Handoff Document
Generated: 2026-03-16

## Summary

Project NEXUS-Sprint: targeting sub-1.0hr Time-to-GPT-2 on 8xH100 (leaderboard record: 1.65hr / CORE 0.2626). This session redesigned NCA tokenization to match the paper, ran an H200 sweep (all NCA configs underperformed baseline), identified a scalar mismatch as the likely cause, fixed it, and launched a second sweep to validate.

## Current Status

### Completed This Session

**NCA tokenization redesign (matching paper + reference repo):**
- Added START/END delimiter tokens per grid frame (vocab 10000 → 10002)
- Added min_grid ICL masking (first grid frame targets masked, not trained on)
- `_compute_seq_params` updated for 38 tokens/grid (36 patches + 2 delimiters)
- `grid_len` saved in `nca_meta.json` for training loop
- Gzip filter upgraded to tunable band (`--min-gzip-ratio` + `--max-gzip-ratio`)
- Default `--num-rules` set to 16,000 (paper scale); speedrun uses 2,000
- All 14 NCA tests pass

**H200 sweep results (from GPU Claude, prior session):**
- All full-transfer (attn+MLP) NCA configs performed WORSE than baseline (best: +0.0015 BPB)
- Attn-only transfer at 4K rules was best NCA result: +0.0007 vs baseline
- GPU Claude added `--nca-transfer-mode` flag (full/attn-only)
- Key finding: MLP transfer hurts at higher rule counts; attention carries the transferable signal

**Scalar mismatch identified and fixed:**
- nanochat's learnable scalars (`resid_lambdas`, `x0_lambdas`, `smear_gate`, `smear_lambda`, `backout_lambda`) were being reinit'd during transfer while attention weights were preserved
- These scalars control `x = resid_lambdas[i] * x + x0_lambdas[i] * x0` — co-adapted with attention during NCA training
- Reiniting them creates a mismatch: attention expects NCA-trained scalar values, gets default init
- The paper's Llama has NO learnable residual scalars — this problem is nanochat-specific
- Fix: preserve all 5 scalar parameters alongside attention/MLP weights during transfer

**Gzip cap at 0.80:**
- Added upper bound to filter out near-random chaotic sequences (ratio > 0.80)
- Paper shows optimal complexity is domain-dependent; band [0.50, 0.80] is more selective

### In Progress

**Scalar-fix sweep running on single H200** (`bash runs/nca_sweep_scalars.sh`):
- 5 sequential d12 experiments (~3.5 hours total):
  - Baseline (no NCA)
  - 2K rules × 50ep (3,100 NCA steps)
  - 4K rules × 50ep (6,250 NCA steps)
  - 6K rules × 50ep (9,375 NCA steps)
  - 8K rules × 50ep (12,500 NCA steps)
- All use attn-only transfer + scalar preservation + gzip band [0.50, 0.80]
- Results will land in `~/.cache/nanochat/sweep_scalars/bpb_*.csv`
- Key question: does any config beat baseline (0.8538)?

### Not Started
- d24 leaderboard attempt on 8xH100 (blocked on sweep results)
- Leaderboard rules require 8xH100 — H200 results are for config selection only

## Recent Changes (This Session)

| Commit | Description |
|--------|-------------|
| `220a3a9` | fix: cap gzip at 0.80, fresh data dirs for scalar sweep |
| `0e27ccc` | feat: add scalar-fix sweep script (2K-8K rules, attn-only + scalars) |
| `555579b` | fix: preserve learnable scalars during NCA transfer |
| `307d780` | docs: update handoff, archive old version |
| `37e4944` | feat: rewrite NCA sweep for H200 with epoch mode configs |
| `2b99ab0` | feat: redesign NCA tokenization (delimiters, ICL masking, gzip band) |

### Key Files Modified

| File | Changes |
|------|---------|
| `scripts/nca_generate.py` | Delimiter tokens, grid_len, gzip band, default 16K rules |
| `scripts/base_train_nca.py` | Vocab +2, grid_len masking, scalar preservation in transfer |
| `tests/test_nca_generate.py` | Updated for delimiters (shapes, vocab range, bijective) |
| `tests/test_nca_integration.py` | Transfer test verifies attn + MLP + scalars preserved |
| `runs/speedrun.sh` | Bumped to 2000 rules |
| `runs/nca_sweep_scalars.sh` | New sweep: 2K-8K rules with scalar fix + gzip cap |
| `docs/modifications.md` | Updated with scalar mismatch finding + all new decisions |

## Next Steps

### After sweep completes
1. **If NCA beats baseline:** Pick best config, run d24 leaderboard attempt on 8xH100 with `--nca-transfer-mode=attn-only`
2. **If NCA still doesn't help at d12:** Try d24 directly — the paper's smallest model is 400M, our d12 is only 124M. Scale may be the issue, not the approach.
3. **If d24 also doesn't help:** NCA likely doesn't transfer to nanochat's architecture. Pivot to other modifications (hybrid attention, data mix optimization).

### Leaderboard attempt parameters (when ready)
```bash
# On 8xH100:
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 \
    --nca-data=$NANOCHAT_BASE_DIR/nca_data \
    --nca-lr=1e-4 --nca-alphabet-size=10 --nca-batch-size=8 \
    --nca-transfer-mode=attn-only \
    --run=$WANDB_RUN
```
Note: on 8xH100, effective NCA batch = 8×8 = 64 (vs paper's 16). Consider gradient accumulation if results are marginal.

## Context & Notes

### Why NCA has underperformed so far
Three compounding issues, discovered and addressed progressively:
1. **Missing tokenization structure** (fixed: delimiters + ICL masking)
2. **MLP contamination** (fixed: attn-only transfer mode)
3. **Scalar mismatch** (fixed: preserve learnable scalars) ← current sweep validates this

### Paper vs nanochat architecture gap
The paper uses standard Llama (no learnable scalars, weight-tied embeddings, standard residual connections). nanochat has `resid_lambdas`, `x0_lambdas`, `smear_gate`, `value_embeds`, separate wte/lm_head. Every transfer workaround addresses a nanochat-specific issue that doesn't exist in the paper's setup.

### Hardware constraints
- **Leaderboard:** Must run on 8xH100 (Karpathy's requirement for comparability)
- **Sweeps:** Any GPU works (H200 is cheaper: $3.50/hr vs $28/hr for 8xH100)
- **d24 OOM:** nca_batch_size=8 and device-batch-size=16 are the max at d24 on H100 (77.7GB/GPU)
- **H200 advantage:** 141GB allows batch_size=32 for NCA, giving effective batch 32 (closer to paper's 16)

### Sweep data locations (on H200 instance)
- `~/.cache/nanochat/sweep_scalars_nca-{2000,4000,6000,8000}-50ep/` — NCA datasets (gzip 0.50-0.80)
- `~/.cache/nanochat/sweep_scalars/bpb_*.csv` — results CSVs
- Data is ephemeral — if the instance is terminated, regenerate with `bash runs/nca_sweep_scalars.sh`
