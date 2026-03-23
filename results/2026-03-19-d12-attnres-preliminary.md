# d12 Block Attention Residuals — Preliminary Results

**Date:** 2026-03-19
**Branch:** `exp/attention_improvements`
**Hardware:** Single H100 80GB HBM3
**Status:** Run in progress (~50% complete at time of report)

## Summary

Implemented Block AttnRes (arXiv:2603.15031) — replaces resid_lambdas + x0_lambdas + backout_lambda with softmax attention over depth. Each sub-layer (attention + MLP) gets a learned pseudo-query (zero-init), blocks default to n_layer // 3 (d12: 4 blocks of 3 layers).

**Early result:** 1.7x throughput regression and BPB behind baseline at step 500.

## Throughput

| Metric | Baseline | AttnRes | Delta |
|--------|----------|---------|-------|
| tok/sec | 546k | 320k | **-41%** |
| MFU | 42% | 24.5% | **-42%** |
| Est. wall time | ~35 min | ~60 min | **+71%** |

## BPB Trajectory (partial)

| Step | Baseline | AttnRes | Delta |
|------|----------|---------|-------|
| 500 | 1.009 | 1.147 | **+0.138** |

BPB gap at step 500 is large. Zero-init queries start as uniform averaging — the model needs time to specialize the depth attention patterns. Open question: does BPB eventually catch up?

## Throughput Diagnosis

The throughput hit is **NOT** from SDPA fallback (FA3 confirmed active). The bottleneck is the AttnRes forward loop:

- `_attn_res()` called **25 times per forward** (2 per layer × 12 layers + 1 final)
- Each call does `torch.stack(sources)` + `rms_norm` + `einsum` + `softmax`
- Rebuilds Python lists and restacks every sub-layer — pure memory traffic waste
- `torch.compile` captures the whole forward in one graph with zero graph breaks, so it's kernel count and memory movement, not Dynamo overhead

## Recommended Throughput Fixes (priority order)

1. **Cache completed blocks as stacked tensor** — stop rebuilding lists and restacking every sub-layer
2. **Batch inter-block queries** — for d12 with 4 blocks, each block has 6 queries reading the same block reps. Precompute once per block
3. **Reduce attn_res_num_blocks** from 4 to 2 as a quick knob to cut overhead
4. **Paper-faithful two-phase block schedule** — one batched inter-block pass per block, then cheap sequential merges via online softmax

## Open Questions

- Does BPB eventually catch up despite the slow start? (zero-init needs time to specialize)
- If final BPB is still worse than baseline AND throughput is 1.7x slower, AttnRes is a net loss
- Should we try throughput-optimized version before concluding, or pivot to DCA?

## Context

This run followed the x0 re-injection experiment (flat 1.0 init), which showed BPB 0.8548 vs baseline 0.8538 — no improvement. The learned scalar curves were informative (non-monotonic, negative values at L9/L11) but confirmed that fixed scalar blending is not a bottleneck at d12 scale. AttnRes was the next step in the validation order.

## Run Command

```bash
OMP_NUM_THREADS=4 uv run python -m scripts.base_train \
    --depth=12 --run="d12-attnres" --model-tag="d12-attnres" \
    --core-metric-every=999999 --sample-every=-1 --save-every=-1
```
