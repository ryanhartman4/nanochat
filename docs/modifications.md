<!-- SYNC NOTE: This file is mirrored at two locations and MUST be kept in sync:
     1. /Models/docs/modifications.md (project root, outside git)
     2. /Models/Speedrun/docs/modifications.md (inside git repo, pushed to remote)
     When editing, update BOTH copies. The repo copy is the source of truth for remote Claudes. -->

# NEXUS-Sprint Modifications Tracker

**Target:** Sub-1.0hr Time-to-GPT-2 (upper bound: 1.25hr) | **Baseline:** 1.65hr (autoresearch r2) | **CORE threshold:** 0.256525

## Modification Status

| ID | Change | Priority | Status | Impact | Risk | Notes |
|----|--------|----------|--------|--------|------|-------|
| A.2.1 | **Smaller Tokenizer** — Retrain BPE at 24K vocab | HIGH | **Reverted** | LOW | LOW | **Tested on 8xH100 d24 run (2026-03-16). Reverted to 32K.** At D-24 (D=1536, V=24576), D/V only improved from 0.047 to 0.063 — marginal gradient flow gain. Meanwhile fewer bytes per token meant less content per 2048-token sequence, hurting CORE tasks that need knowledge coverage. Final CORE 0.2542 fell just short of 0.2565 threshold. See `results/2026-03-16-d24-8xh100-nca-24k.md`. |
| A.2.2 | **NCA Pre-Pre-Training** — Short NCA stage before main training to bootstrap attention induction circuits | — | **Failed** | NONE | LOW | **Extensively tested on H200 (d12) and 8xH100 (d24). No improvement found.** H200 sweep (2K-4K rules, full & attn-only transfer, 50-100 epochs) showed all NCA configs equal or worse than baseline. Best result: 4K rules attn-only at +0.0007 BPB — neutral at best. Full transfer actively hurts (MLPs overfit to NCA features). Convergence rate identical to baseline in plateau regime. Paper's results (1.6x speedup at 1.6B/16K rules/100K steps) likely require larger model capacity. Code archived on `exp/NCA`. See `results/2026-03-16-h200-nca-sweep.md`. |
| A.2.3 | **Multi-Token Prediction** — 2-4 auxiliary prediction heads on transformer trunk | LOW | Backlogged | HIGH | MEDIUM | DEPRIORITIZED: Reports from other speedrun participants indicate MTP heads blow up memory with minimal convergence gains at nanochat's model scale. Revisit if tokenizer + NCA gains plateau. Sequential fwd/bwd would keep memory constant but adds per-step overhead (~10-15%). |
| A.2.4 | **Hybrid Attention (Gated DeltaNet)** — Replace 75% of attention layers with linear attention | LOW | Possible Improvement | MEDIUM | MEDIUM | Production Triton kernels exist: [`fla-core`](https://github.com/fla-org/flash-linear-attention) (pip-installable, H100-benchmarked, powers Qwen3.5) and [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) (NVIDIA official, ICLR 2025). ~45K tok/sec at 1.3B on H100. **Key blocker:** `torch.compile` fails on DeltaNet (recurrent state updates cause graph breaks). nanochat relies on `torch.compile` for speed — need selective compilation (DeltaNet layers excluded, softmax layers compiled). Estimated 2-3 days integration, not weeks. Risk downgraded from HIGH to MEDIUM given kernel availability. |
| A.2.4b | **SDDF Hybrid Pattern** — Conservative hybrid: S(softmax) D(DeltaNet) D(DeltaNet) F(full softmax) | LOW | Possible Improvement | MEDIUM | MEDIUM | Middle-ground variant of A.2.4. Only replaces 50% of layers with DeltaNet (vs 75%). S layers keep proven short-window softmax for local patterns. ~29% compute savings vs SSSL. Lower risk than full DDDF. Same `torch.compile` constraint applies to D layers. |
| A.2.4c | **Split-Model Hybrid** — DeltaNet in front half (layers 1-12), traditional softmax in back half (13-24) | LOW | Possible Improvement | MEDIUM | MEDIUM | Exploits observation that early layers learn local features (forgiving of approximation) while later layers need precise attention for quality. Back half stays identical to nanochat's proven architecture. Easier to A/B test and bisect issues. `torch.compile` can apply to back half normally. |
| A.2.5 | **Data Mix Optimization** — Classify ClimbMix documents and resample for code/math-heavy mix | LOW | Possible Improvement | MEDIUM | LOW | ClimbMix arrives pre-mixed with no source labels. Requires document classification (heuristic or model-based) then resampling into new parquet shards. Low risk since data format stays the same. Could also explore shard-level composition differences. |

## Attention Architecture Visual Reference

See [attention_explainer.html](attention_explainer.html) for an interactive comparison of SSSL, SDDF, and DDDF patterns including compute savings, memory footprint, and scaling charts.

## Validation Order

1. ~~**A.2.1 Tokenizer** — Retrain existing BPE at 24K vocab~~ **REVERTED** (2026-03-16)
2. ~~**A.2.2 NCA** — Pre-pre-training stage~~ **FAILED** (2026-03-17)
3. Next candidate TBD — evaluate remaining modifications (A.2.4 hybrid attention, A.2.5 data mix, or new ideas)
4. Validate on d24, leaderboard submission

## Ideas Pipeline (2026-03-17)

Papers and concepts to evaluate for next round of modifications. Not yet analyzed for feasibility within nanochat constraints.

| # | Idea | References | Notes |
|---|------|------------|-------|
| 1 | **Hard-Max Attention / C-Compiler LLM** — Discrete attention as program execution | [Percepta blog](https://www.percepta.ai/blog/can-llms-be-computers) | Already in tech spec as ref #1 (Tzamos et al. 2026) |
| 2 | **Mixture of Depths** — Dynamically skip layers per token | [arXiv:2603.15619](https://arxiv.org/abs/2603.15619) | Could reduce per-step compute without losing quality on easy tokens |
| 3 | **Deep Cross Attention / Attention Residuals** — Residual connections across attention layers | [arXiv:2502.06785](https://arxiv.org/abs/2502.06785), [Attention Residuals PDF](https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf), [arXiv:2603.15031](https://arxiv.org/abs/2603.15031) | MoonshotAI. Could improve gradient flow through attention stack |
| 4 | **Looping Layers** — Single wide layer looped N times instead of N independent layers | [Tweet thread](https://x.com/yichen4nlp/status/2029864063162011689) | Attention residuals + MoE reuse both point here. "Less complicated = better" thesis. Strict generalization of current paradigm. Related to weight tying across depth. |
| 5 | **Recursive Prompts** — Re-inject original prompt midway through forward pass | (speculative) | Like a residual connection at the input level. Could help with instruction following / staying on task |
| 6 | **Small RL on Itself** — Self-improvement via RL at small scale | [arXiv:2505.19590](https://arxiv.org/pdf/2505.19590) | Could boost quality without extra data |
| 7 | **Gaussian Noise Injection** — Add noise during training to improve generalization | [arXiv:2603.12228](https://arxiv.org/pdf/2603.12228), [RandOpt repo](https://github.com/sunrainyg/RandOpt) | Regularization technique. Low integration cost if it works |
| 8 | **Synthetic Data for Pre-Training** — Use LLM-generated data for initial training | [NVIDIA Nemotron-3 Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf) | Could improve data quality/coverage vs raw web crawl |
| 9 | **Attention Layer Shortcuts / Looped Attention** — Skip connections across attention layers, or loop attention 3x instead of deeper stacks | (speculative) | Related to ideas #3 and #4. Reduce depth while maintaining effective receptive field |

## Future Improvements (Not Yet Implemented)

### Pinpoint CORE Crossing Step
The current training script evaluates CORE at the end of training only. To measure the exact speedup from NCA, use `--core-metric-every=250` on validation runs so we can identify which step first crosses CORE >= 0.256525. Compare that step number against the baseline's crossing point to compute the precise convergence speedup ratio. This is essential for d24 validation and leaderboard timing.

### CORE Auto-Stop (BPB-Triggered)
For the leaderboard submission, add early stopping to `base_train.py` — but NOT by running CORE every N steps. CORE eval takes 3-5 minutes per evaluation (22 tasks, few-shot inference across thousands of examples). At `--core-metric-every=250` that's ~32 minutes of eval overhead on a 25-minute training run, which would negate the NCA speedup entirely.

**Better approach — BPB as a proxy trigger:** BPB (bits per byte) is already computed every 250 steps by default (`--eval-every`) and costs ~2-5 seconds (one forward pass on val split, no backward, no generation). Use BPB to estimate when CORE threshold is near, then run exactly one CORE eval to confirm.

Implementation:
1. Calibrate BPB→CORE mapping from validation runs (the current d12 run establishes one data point)
2. During leaderboard run: monitor BPB each eval step (already free)
3. Once BPB drops below the calibrated threshold, trigger a single CORE eval
4. If CORE >= 0.256525, stop training and record wall-clock
5. Total CORE overhead: one eval (~4 min) instead of 8+ (~32 min)

**Caveat:** BPB and CORE aren't perfectly correlated. BPB measures raw prediction quality while CORE measures downstream task performance. Need 2-3 calibration data points (different step counts / model configs) to establish a reliable threshold. The d12 validation run provides the first data point.

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-14 | Target updated from sub-1.2hr to sub-1.0hr | Back-of-envelope math: NCA saves ~30min off 1.65hr baseline. Upper bound 1.25hr if gains don't fully stack. |
| 2026-03-14 | MTP moved to backlog | Other participants report memory blowup with minimal gains at nanochat scale. Not worth the risk for initial submission. |
| 2026-03-14 | NCA vocab: test both n=2 (16 patch tokens) and n=4 (256 patch tokens) | Paper (Han et al. 2026) found n=2 scales best — simpler dynamics produce more consistent transferable structure. 256 tokens (n=4) is our original plan. A/B test both on d12. Only need CORE >= 0.26, so simpler NCA is likely fine. |
| 2026-03-14 | NCA data pre-generated as auxiliary dataset | Self-contained generation adds latency to each run. Pre-generating keeps training script clean and allows reuse across experiments. |
| 2026-03-14 | NCA integrated inline in base_train.py | Competition rules — wall-clock timer must capture everything in a single script invocation. |
| 2026-03-14 | Tokenizer simplified: retrain existing BPE at smaller vocab, no custom corpus | nanochat's `tok_train.py` already supports `--vocab-size`. ClimbMix training data is already well-curated. Higher BPE merges overfit to corpus-specific patterns — fewer merges = more generalizable + better gradient flow. |
| 2026-03-14 | Start at 24K vocab, 16K as optional bump | 24K is conservative — still a meaningful bottleneck reduction from 32K. 16K is more aggressive (longer sequences = more compute per step). Try 16K only if 24K passes CORE easily and time is comfortable. |
| 2026-03-14 | All DeltaNet variants marked as possible improvements | At T=2048 with FA3 on H100s, DeltaNet's O(n) advantage may not beat FA3's optimized O(n²) without custom CUDA kernels. Three variants explored: DDDF (aggressive, 43% savings), SDDF (conservative, 29% savings), split-model (front DeltaNet / back softmax). All conditional on tokenizer + NCA proving out first. |
| 2026-03-14 | DeltaNet risk downgraded from HIGH to MEDIUM | Production Triton kernels exist via `fla-core` (pip install) and NVlabs reference. H100-benchmarked, powers Qwen3.5. No custom kernel work needed. Main engineering challenge is `torch.compile` incompatibility — DeltaNet's recurrent state updates cause graph breaks. Need selective compilation strategy. Estimated 2-3 days integration. Kept backlogged — focus remains on tokenizer + NCA. |
| 2026-03-14 | NCA default switched from n=2 to n=4 | H100 generation testing revealed n=2 (16 tokens) has 98% gzip rejection rate at seq_len=2048 — generation takes 25+ min. n=4 (256 tokens) has 55% rejection, generates 164M tokens in <1 min on CUDA. At nanochat's d12/d24 scale (124-350M params), the paper's n=2 scaling advantage is marginal. For a speedrun where wall-clock is everything, n=4 is the pragmatic choice. 80K sequences from 1,390 unique rules still provides ample dynamical diversity. |
| 2026-03-14 | NCA generator batched + GPU-accelerated | Original single-trajectory-per-rule code took 2+ hrs for 164M tokens. Batched to 128 trajectories per rule with CUDA conv2d. Also added --device flag, CUDA-safe gzip (CPU transfer before .numpy()), and OMP_NUM_THREADS override in speedrun.sh (global =1 cripples CPU convs). |
| 2026-03-16 | 24K tokenizer reverted to 32K | d24 8xH100 run: CORE 0.2542 in 95 min — just below 0.2565 threshold. 24K vocab reduces bytes per token, meaning each 2048-token context covers less text. For CORE's knowledge-heavy tasks (MMLU, ARC), content coverage per step matters more than the marginal D/V gradient flow improvement (0.047→0.063) at d24 scale. Gradient bottleneck paper (Godey & Artzi 2026) shows the real lever is D (head rank), not V (vocab size). |
| 2026-03-16 | D-24 OOM constraints documented | D-24 + FP8 uses ~77.7GB/GPU. device-batch-size=32 OOMs. NCA batch-size=32 also OOMs. Fixed at device-batch-size=16 (grad accum 2) and nca-batch-size=8. Both D-24 and D-26 are stuck at the same batch size, so no throughput advantage from choosing smaller model. |
| 2026-03-16 | NCA contribution unconfirmed | The 95 min / CORE 0.2542 result is faster than Karpathy's 99 min but lower CORE. Without a matched non-NCA baseline at identical config (32K, d24, FP8, ratio 8.0), we cannot isolate NCA's contribution. Next run: 32K vocab + NCA to compare against Karpathy's 32K-without-NCA baseline. |
| 2026-03-16 | Head avoidance (gradient bottleneck bypass) prototyped | Branch `exp/head-avoidance`: every K-th step, bypass LM head and compute cosine proxy loss against lm_head.weight[targets] in D-dimensional decoder space. d12 H200 test: BPB 0.887 final (vs 0.863 for NCA run). Promising but needs matched baseline comparison. DDP-safe: zero-fills None grads before optimizer step. FP8-safe: dequantizes lm_head weight before indexing. |
| 2026-03-16 | NCA switched from n=4 to n=10 (paper's actual default) | We were using n=4 based on a secondary finding that "n=2 scales best at 1.6B." But the paper's headline results (1.6x speedup, 6% improvement) used n=10 with LR 1e-4. We optimized for a scaling finding instead of reproducing the validated config. n=10 gives 10,000 patch tokens — much richer dynamics for attention circuit bootstrapping. LR also changed from 3e-4 to 1e-4 (paper default). |
| 2026-03-16 | NCA step count sweep planned | Before the next d24 leaderboard run, sweep NCA steps on d12 (one per GPU, 8 parallel): 250, 500, 750, 1000 steps at n=10/lr=1e-4. d12 trains in ~10-15 min per GPU, so full sweep in ~15 min. Goal: find the step count that maximizes convergence speedup relative to wall-clock overhead. |
| 2026-03-15 | NCA tokenization redesigned to match paper | Previous implementation was missing delimiter tokens and min_grid masking — two structural elements from Han et al. 2026 and the reference repo (`danihyunlee/nca-pre-pretraining`). Added: (1) START/END tokens wrapping each grid frame (vocab 10000→10002), (2) first grid frame masked in targets (ICL context, not training signal), (3) gzip filter as tunable band with `--max-gzip-ratio` (paper shows optimal complexity is domain-dependent: 30-40% for code, 50%+ for math/text). Speedrun bumped from 1000→2000 rules (~3,100 NCA steps, ~10 min vs old 500 steps/5 min). Transfer test fixed to assert MLP preservation (not reinit). |
| 2026-03-15 | Effective batch size deviation identified | Paper uses effective batch 16 for pre-pre-training. Our DDP across 8 GPUs with nca_batch_size=8 gives effective batch 64 (4x paper). This means fewer gradient updates per data seen (~3,100 steps vs paper's ~100K). Fixing this would require gradient accumulation in the NCA loop or single-GPU NCA training — both add wall-clock time. Noted as a potential improvement if current results underperform. |
| 2026-03-16 | H200 sweep: NCA full transfer underperforms baseline at d12 | All full-transfer (attn+MLP) NCA configs performed worse than baseline (best: +0.0015 BPB). Attn-only transfer at 4K rules was much better (+0.0007), suggesting MLPs overfit to NCA-specific features at higher rule counts. Added `--nca-transfer-mode` flag (full/attn-only). More rules → better attention init but worse MLP contamination. See `results/2026-03-16-h200-nca-sweep.md`. |
| 2026-03-16 | Scalar mismatch identified as likely cause of slow convergence | nanochat's learnable scalars (`resid_lambdas`, `x0_lambdas`, `smear_gate`, `smear_lambda`, `backout_lambda`) were being reinit'd during NCA transfer while attention weights were preserved. These scalars control residual stream scaling (`x = resid_lambdas[i] * x + x0_lambdas[i] * x0`) and are co-adapted with attention during NCA training. Reiniting them creates a mismatch: attention expects NCA-trained scalar values but gets default init. The paper's Llama has no such learnable scalars, so this problem doesn't exist for them. Fix: preserve scalars alongside attention/MLP weights during transfer. All NCA configs started behind baseline — this mismatch is the most likely explanation. |
| 2026-03-17 | **NCA marked as FAILED**, code archived to `exp/NCA` | After extensive sweeps (H200 d12: 2K-4K rules, full & attn-only transfer, 50-100 epochs; 8xH100 d24: n=4/n=10, 500 steps), no configuration improved over baseline. Best result was neutral (+0.0007 BPB). Root causes: (1) NCA-trained MLPs overfit to rule-specific features and hurt LM training, (2) even attn-only transfer shows no speedup — convergence rate is identical to baseline in plateau regime, (3) nanochat's d12/d24 models are likely too small for NCA's benefits to manifest (paper used 1.6B params). Master reset to upstream; NCA code preserved on `exp/NCA` branch. |
