# Deep Supervision: Shared-Weight Auxiliary Head at Layer N/2

## Problem
The LM head (D=768 -> V=32768) creates a gradient bottleneck. Lower layers get
weak, compressed gradients that pass through the full stack + final head.
Our proxy loss experiments (substitutive and additive cosine similarity) all
regressed vs baseline, likely because cosine similarity lacks contrastive signal
and/or the bottleneck isn't severe enough at this scale for a proxy to help.

## Proposal
Add a cheap auxiliary prediction head at the halfway layer (layer 6 for d=12)
that shares weights with `lm_head`. This gives lower layers a direct CE gradient
path that skips layers 6-11 entirely.

## Design

### Forward pass changes (gpt.py)
1. During the transformer block loop, after layer `n_layer // 2`, save `hidden_mid = x`
2. Continue the rest of the forward normally (layers 7-11, backout, norm, lm_head -> main CE)
3. Also compute: `aux_logits = norm(hidden_mid) @ lm_head.weight.T` (shared weights)
4. Apply same softcap (15) and vocab slicing as main head
5. `aux_ce = F.cross_entropy(aux_logits, targets)`
6. Return `main_ce + lambda * aux_ce` (when aux is enabled)

### Gradient flow
- Main CE backward: head -> layers 11->0 (unchanged)
- Aux CE backward: shared lm_head -> directly into layers 5->0
- Lower layers get TWO gradient paths: normal deep path + shortcut
- lm_head gets gradient from both paths (shared weights)

### Training loop changes (base_train.py)
- New CLI args:
  - `--aux-head-lambda` (float, default 0.0 = disabled, suggest 0.1-0.3)
  - `--aux-head-layer` (int, default -1 = auto n_layer//2)
- Pass lambda/layer to model forward, or configure on the model object
- No special EMA handling needed (single combined loss, always CE-based)
- Log `[+AUX]` tag on steps where aux is active

### Model changes (gpt.py)
- No new parameters except possibly a learnable scale scalar (optional)
- `norm()` before aux projection reuses the existing `norm` function
- The `backout_lambda` subtraction should NOT be applied to `hidden_mid` --
  backout is specific to the final head's needs

### Init
- No weight init changes needed (shared weights, norm is parameterless)
- If adding a learnable scale: init to 1.0 (neutral)

## Design choices

### Lambda: 0.1-0.3
Aux is a helper signal, not a replacement. Too high and it could distort
upper-layer training by over-weighting lower-layer representations.

### Softcap: reuse 15
Same as main head for consistency. The aux logits will be noisier (layer 6
representations aren't as refined), but softcap prevents gradient explosions.

### Every step (not periodic)
Unlike the proxy approach, this uses real CE loss -- no loss function mismatch.
The overhead is one extra D*V matmul per step, which is worth it for continuous
gradient signal to lower layers.

### Optional: decay lambda over training
Aux matters most early (helping lower layers form representations) and less
late (representations are settled). Could multiply lambda by `(1 - step/total_steps)`
or just keep it constant for simplicity.

## Overhead estimate
- Extra forward: (B*T, D) x (D, V) matmul = ~25M multiply-adds per token
- Extra backward through same matmul + layers 0-5
- Total: ~30-40% overhead on forward+backward portion
- For d=12 at ~29 min baseline, expect ~38-40 min wall time
- Break-even: need BPB improvement worth ~10 min of extra baseline training

## Comparison to proxy loss experiments

| Approach | Final BPB | vs Baseline | Issue |
|----------|-----------|-------------|-------|
| Baseline | 0.8603 | -- | -- |
| Proxy K=3 start=0% | ~0.900 | +0.040 | CE steps wasted, no contrastive signal |
| Proxy K=3 start=75% | 0.8706 | +0.010 | Same issues, less exposure |
| Proxy K=10 start=75% | 0.8619 | +0.002 | Minimal proxy, minimal damage |
| Additive proxy (λ=0.1) | 0.8603 | +0.000 (+2% wall time) | Neutral BPB, proxy signal is noise |
| **Deep supervision (this)** | TBD | TBD | Real CE, direct lower-layer gradients |

Key advantage: deep supervision uses the same loss function (CE + softmax
competition) rather than cosine similarity. The gradient signal is semantically
identical to what the model already optimizes, just delivered via a shorter path.
