# Handoff: x0 Re-Injection Experiment
**Branch:** `exp/attention_improvements`
**Date:** 2026-03-18

## What Changed

Two edits on the `exp/attention_improvements` branch:

### 1. `nanochat/gpt.py` line 237-240 — x0_lambdas init changed to flat 1.0
**Before:** Linear decay 0.20 → 0.05 (prescribed curve)
```python
self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))
```
**After:** Flat 1.0 at every layer (let optimizer discover the curve)
```python
self.x0_lambdas.data[i] = 1.0
```

### 2. `scripts/base_train.py` line 621-625 — Per-layer scalar logging
Added logging of `x0_lambdas` and `resid_lambdas` for all layers to wandb every training step. Look for panels `scalars/x0_lambda_*` and `scalars/resid_lambda_*`.

## What to Run

Single d12 run on H200 (or single H100):
```bash
OMP_NUM_THREADS=4 uv run python -m scripts.base_train \
    --depth=12 \
    --run="d12-x0-flat1" \
    --model-tag="d12-x0-flat1" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
```

This should take ~30-35 min on a single H200 (matching prior d12 baseline runs).

## What to Compare Against

**Baseline:** d12 H200, BPB = 0.8538 (from NCA sweep, default x0_lambdas init 0.20→0.05)

The baseline used the same hardware and training config, so BPB is directly comparable.

## What to Watch

### Primary metric
- **Final BPB** — compare to 0.8538 baseline. Any improvement (even 0.001) is signal.

### wandb scalar panels (the interesting part)
- **`scalars/x0_lambda_*`** — These start at 1.0 everywhere. Watch what shape the optimizer learns:
  - If they all decay toward ~0.1-0.2 → model wants the original prescribed decay, 1.0 init was too high but self-corrected
  - If they hold near 1.0 → model wants MUCH more input re-injection than we were giving it
  - If a spike emerges at L/2 (layer 6) → model wants a midpoint boost specifically
  - If a U-shape emerges (high at layer 0, low in middle, high near end) → model wants input signal at both ends
  - If they go near 0.0 → model doesn't want input re-injection at all (current 0.20→0.05 was already too much)
- **`scalars/resid_lambda_*`** — These start at 1.15→1.05 decay. Watch if they change in response to the x0 change.

### Early warning signs
- **Loss spike in first 100 steps** — The flat 1.0 init is 5-8x higher than default at most layers. If training is unstable early, note it but let it run — the optimizer should correct within a few hundred steps.
- **BPB > 0.90 at step 500** — If it's significantly behind baseline trajectory at step 500, the init may be too aggressive. Note the value but let it finish.

## Why This Matters

nanochat already has x0 re-injection via `x0_lambdas`, but we prescribed a linear decay (0.20→0.05) without evidence that this is optimal. By starting flat at 1.0, the optimizer tells us the *actual* ideal re-injection curve. This informs whether we need the heavier AttnRes/DCA machinery or whether just tuning the existing scalars is enough.

Three papers (AttnRes arXiv:2603.15031, DCA arXiv:2502.06785, MoDA arXiv:2603.15619) all confirm that depth signal dilution is a real bottleneck. This experiment tests the cheapest possible fix before we invest days in the full solutions.

## After the Run

Report back:
1. Final BPB (vs 0.8538 baseline)
2. Screenshot or values of the learned x0_lambda curve (all 12 values at final step)
3. Screenshot or values of the learned resid_lambda curve
4. Any training instability observed in the first 100-500 steps
5. Wall-clock time for the run
