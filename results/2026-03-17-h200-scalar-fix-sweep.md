# NCA Scalar-Fix Sweep — H200 d12 Results (2026-03-17)

## Hypothesis

Preserving learnable scalars (`resid_lambdas`, `x0_lambdas`, `smear_gate`, `smear_lambda`,
`backout_lambda`) during NCA-to-LM transfer would fix the convergence penalty observed in
the prior sweep. These scalars control residual stream scaling and were co-adapted with
attention during NCA training but reset to defaults during transfer.

## Setup

- **Hardware:** Single H200 (143GB)
- **Model:** d12 (124M params)
- **LM Training:** ~2,205 steps, ~33 min per run
- **Eval:** BPB logged every 100 steps
- **NCA batch size:** 32
- **Transfer mode:** attn-only + scalar preservation (new)
- **Gzip band:** [0.50, 0.80] (capped at 0.80 to filter chaotic sequences)
- **Data generation:** Batched per-rule with post-hoc gzip filtering (faster than per-epoch retry)

## Changes from Prior Sweep

1. **Scalar preservation:** Learnable scalars now kept alongside attention weights during transfer
2. **Gzip cap at 0.80:** Prior sweep used default 1.0 (no upper cap)
3. **Batch generation:** All epochs per rule generated in one GPU call, then gzip-filtered
4. **Dropped 8K config:** NCA training time (~162 min) exceeds speedrun budget

## Results

| Config | Final BPB | Min BPB | vs Baseline | Wall Time (LM) |
|--------|-----------|---------|-------------|-----------------|
| **Baseline** | 0.8841 | **0.8837** | — | 33.24 min |
| 2K×50ep | 0.8927 | 0.8926 | +0.0089 | 33.37 min |
| 4K×50ep | 0.8879 | 0.8878 | +0.0041 | 33.26 min |
| 6K×50ep | 0.8857 | 0.8854 | +0.0017 | 33.30 min |

**No config beat baseline.** Scalar fix did not resolve the convergence penalty.

## Convergence Trajectories

| Step | Baseline | 2K | 4K | 6K |
|------|----------|------|------|------|
| 100 | 1.382 | 1.514 | 1.510 | 1.504 |
| 500 | 1.005 | 1.024 | 1.020 | 1.014 |
| 1000 | 0.942 | 0.955 | 0.951 | 0.947 |
| 1500 | 0.904 | 0.915 | 0.911 | 0.908 |
| 2000 | 0.885 | 0.895 | 0.890 | 0.888 |
| 2205 | 0.884 | 0.893 | 0.888 | 0.886 |

## Comparison with Prior Sweep (without scalar fix)

| Config | Prior (no scalar fix) | This Sweep (scalar fix) |
|--------|----------------------|------------------------|
| Baseline | 0.8538 | 0.8837 |
| 2K attn-only | 0.8564 (+0.0026) | 0.8926 (+0.0089) |
| 4K attn-only | 0.8545 (+0.0007) | 0.8878 (+0.0041) |
| 6K attn-only | — | 0.8854 (+0.0017) |

Note: Different hardware setup (prior was 8xH100, this is single H200). Baselines differ
by 0.03 BPB, making direct comparison of gaps unreliable.

## Key Findings

### 1. Scalar preservation did not help

All configs performed worse relative to baseline than the prior sweep's best result
(4K attn-only at +0.0007). The scalar fix hypothesis is rejected — the convergence
penalty is not caused by scalar reinitialization.

### 2. More rules still helps

Consistent with prior sweep: 2K (+0.0089) > 4K (+0.0041) > 6K (+0.0017). Each doubling
of rule count roughly halves the gap. But even extrapolating to 16K rules, we'd only
approach baseline — not beat it.

### 3. NCA configs never cross over

All NCA configs start ~0.12-0.13 behind baseline at step 100 and converge at the same
rate as baseline. The gap narrows early but plateaus. NCA provides a strictly worse
initialization for language modeling at this scale.

### 4. Gzip cap may have hurt

The 0.80 gzip cap filtered ~7-8% of trajectories (46 usable epochs instead of 50).
Prior sweep with no cap got better results, though on different hardware.

## Conclusion

NCA pre-pre-training does not provide a convergence speedup for nanochat's d12 model on
a single GPU. The approach adds wall time (data generation + NCA training) while producing
a worse initialization than random. For the speedrun goal of sub-1.0hr Time-to-GPT-2,
NCA should be deprioritized in favor of other modifications.

## Data on Disk

All NCA datasets at `~/.cache/nanochat/sweep_scalars_nca-*/`:
- `sweep_scalars_nca-2000-50ep/` — 2K rules (92,775 sequences kept from 100,000)
- `sweep_scalars_nca-4000-50ep/` — 4K rules
- `sweep_scalars_nca-6000-50ep/` — 6K rules

Results CSVs at `~/.cache/nanochat/sweep_scalars/bpb_*.csv`
