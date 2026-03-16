# Head Avoidance / Gradient Bottleneck Experiments — d=12 Findings

Paper: Godey & Artzi 2026, arXiv:2603.10145v1
Branch: `exp/head-avoidance`
Model: d=12, D=768, V=32768
Baseline final BPB: 0.8603 (2079 steps, ~29 min)

## Summary

All attempts to bypass or mitigate the LM head gradient bottleneck at d=12
regressed or were neutral. The head bottleneck is likely not the binding
constraint at this model scale.

## Experiments

| # | Approach | Config | Final BPB | vs Baseline | Wall Delta |
|---|----------|--------|-----------|-------------|------------|
| 1 | Substitutive proxy (cosine sim) | K=3, start=0% | ~0.900 | +0.040 | 0% |
| 2 | Substitutive proxy (cosine sim) | K=3, start=75% | 0.8706 | +0.010 | 0% |
| 3 | Substitutive proxy (cosine sim) | K=10, start=75% | 0.8619 | +0.002 | 0% |
| 4 | Additive proxy (CE + 0.1*cosine) | K=3, start=75% | 0.8603 | +0.000 | +2% |
| 5 | Deep supervision (shared lm_head) | lambda=0.2, every step | killed early | +0.007 steady | +16% |

## What we learned

### Cosine similarity proxy is too weak (experiments 1-4)
- Proxy loss: `-cos(hidden, lm_head.weight[target])` — only pushes toward correct token
- No contrastive signal (softmax competition among all V tokens)
- Substitutive: each proxy step wastes a CE step; damage proportional to proxy frequency
- Additive: proxy gradient is noise — zero effect on BPB, just adds wall time

### Shared lm_head dual-objective hurts (experiment 5)
- Aux CE at layer N/2 using shared lm_head weights
- lm_head receives gradient from both layer-6 and layer-12 representations
- These distributions are very different; optimizing for both compromises the head
- Consistent +0.007 BPB regression from step 100 onward

### The bottleneck likely isn't binding at d=12/D=768
- Every gradient enhancement attempt either hurt or was neutral
- The model appears capacity-limited, not gradient-limited at this scale
- The paper's results may require larger models (larger V/D ratio) to manifest

## Untested ideas (for future reference if scaling up)

1. **Detached aux head**: `F.linear(aux_h, lm_head.weight.detach())` — real CE with
   softmax competition but head only trained from final layer. Would cleanly test
   whether lower layers benefit from direct CE gradient without corrupting the head.

2. **Separate aux head**: new Linear(D, V) with own weights. Clean gradient for both
   heads but adds 25M params.

3. **Contrastive proxy**: InfoNCE with in-batch negatives instead of cosine sim.
   Adds the competition signal cosine similarity lacks.

4. **Gradient amplification**: post-hoc scaling of backbone gradients based on
   head's singular values. No loss function changes.

5. **Intermediate layer shortcuts**: architectural changes (DenseNet-style cross-layer
   connections) rather than auxiliary losses.
