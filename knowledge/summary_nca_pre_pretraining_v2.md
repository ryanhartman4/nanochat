# Training Language Models via Neural Cellular Automata

**Paper:** Dan Lee, Seungwook Han, Akarsh Kumar, Pulkit Agrawal (2026). "Training Language Models via Neural Cellular Automata." arXiv:2603.10055 (ICML 2025)
**Code:** https://github.com/danihyunlee/nca-pre-pretraining

---

## Core Idea

Pre-pre-training LLMs on synthetic NCA trajectories (164M tokens) improves downstream language modeling by up to 6% and accelerates convergence by 1.6x. Surprisingly, this outperforms pre-pre-training on 1.6B tokens of natural language (C4) with 10x more compute. The key insight: NCA sequences force in-context rule inference (each sequence has a unique latent rule), which builds general-purpose attention circuits that transfer to language.

## Method

### NCA Data Generation
- **Grid:** 12x12 with periodic (toroidal) boundaries
- **States:** n=10 discrete states per cell (one-hot encoded)
- **Rule network:** Conv(4 filters, 3x3) → Conv(16 filters, 1x1) → ReLU → Conv(10 filters, 1x1)
- **Temperature:** τ = 1e-3 (near-deterministic sampling via softmax)
- **Tokenization:** 2x2 non-overlapping patches → bijective mapping to vocab of 10^4 = 10,000 tokens
- **Sequence length:** 1024 tokens
- **Gzip filtering:** Keep trajectories with compression ratio > 50% (reject trivially simple dynamics)
- **Total:** 164M tokens from 16,000 unique rules

### NCA Pre-Pre-Training
- **Model:** 1.6B params, 24 layers, 32 heads, 2048 hidden dim (Llama architecture)
- **Optimizer:** Adam, LR 1e-4, batch size 16, 100 epochs
- **Warmup:** 10% of total steps
- **No weight decay** during NCA phase

### Transfer Protocol (CRITICAL)
- **Reinit ONLY embeddings** (input embedding + output head) for new language vocab
- **Keep ALL transformer weights:** attention, MLP, layer norms
- All parameters unfrozen during language pre-training
- Language pre-training LR: 5e-4 (math/text) or 2e-4 (code)

## Key Findings

1. **5-6% perplexity improvement** on OpenWebText, OpenWebMath, CodeParrot at 1.6B scale. Effect robust across 4 seeds.

2. **1.4-1.6x faster convergence** to scratch baseline's final perplexity.

3. **Beats 1.6B tokens of C4** with only 164M NCA tokens — 10x less data, better results.

4. **Attention is the transfer mechanism.** Reinit ablation: removing attention weights causes the largest transfer degradation. MLP/LayerNorm transfer is domain-dependent (helps code, hurts web text).

5. **Optimal NCA complexity is domain-dependent.** Web text and math benefit from high complexity (50%+ gzip). Code benefits from intermediate complexity (30-40% gzip), matching code's own 32% gzip ratio.

6. **Smaller alphabet (n=2) scales better** than n=10 or n=15 — simpler dynamics produce more consistently transferable structure at larger token budgets. But n=10 is better at the 164M token budget used for headline results.

7. **Transfer persists and grows** throughout language pre-training — not just an initialization effect.

8. **Gains transfer to reasoning benchmarks:** GSM8K +14%, HumanEval +11%, BigBench-Lite +41% (at pass@4).

---

## Relevance to This Project — CRITICAL DIFFERENCES

### 1. Transfer Protocol: We reinit too much

| Component | Paper | Our implementation |
|-----------|-------|-------------------|
| Attention | **KEEP** | KEEP |
| MLP | **KEEP** | REINIT |
| LayerNorm | **KEEP** | REINIT |
| Embeddings | REINIT | REINIT |
| Value embeds | N/A | REINIT |
| Scalars (resid/x0 lambdas) | N/A | REINIT |

**This is the biggest discrepancy.** Our `transfer_nca_to_text()` in `base_train_nca.py` calls `model.init_weights()` which reinits EVERYTHING, then loads back only attention weights. The paper reinits ONLY embeddings.

The paper's ablation (Fig 5) shows MLP transfer helps on CodeParrot but hurts slightly on OpenWebText. Since nanochat trains on ClimbMix (web-heavy), reiniting MLPs may actually be correct. But we should test keeping MLPs too.

### 2. NCA Training Duration: We undertrain

| Setting | Paper | Our implementation |
|---------|-------|-------------------|
| Duration | **100 epochs** over 16K rules | **500 steps** |
| Rules | 16,000 train rules | ~100-200 (until token target hit) |
| Data regen | Fresh data each epoch | Fixed pre-generated dataset |
| Batch size | 16 (eff. 32 with grad accum 2) | 8 (d24 constraint) |

The paper trains for 100 epochs with data regeneration — the model sees many diverse NCA trajectories. We do 500 steps with a fixed dataset. The paper's NCA phase likely takes much longer than our ~5 minutes. At their model scale (1.6B), 100 epochs on 16K rules is substantial training.

### 3. Rule Architecture: Minor difference

| Setting | Paper | Ours |
|---------|-------|------|
| Conv1 | 4 filters, 3x3 | **alphabet_size** filters, 3x3 |
| Hidden | Conv(16, 1x1) → ReLU → Conv(10, 1x1) | Conv(16, 1x1) → ReLU... wait, ours is different |

Paper: `Conv(n_states→4, 3x3) → Conv(4→16, 1x1) → ReLU → Conv(16→n_states, 1x1)`
Ours: `Conv(alphabet_size→16, 3x3) → ReLU → Conv(16→alphabet_size, 1x1)`

Paper uses 4-channel bottleneck in the first conv, we use 16 hidden throughout. Paper has a 3-layer network, ours is 2-layer. This may affect the complexity of generated dynamics.

### 4. Temperature

Paper uses τ=1e-3, we use τ=1e-3. ✓ Match.

### 5. Sequence Length

Paper uses 1024, we use 2048. Our sequences are 2x longer, which means more temporal context per sample. This should be fine or better.

### Recommended Fixes (Priority Order)

1. **Increase NCA training duration.** 500 steps is almost certainly too few. The paper uses 100 epochs. Try 2000-5000 steps with the sweep to find the sweet spot for nanochat's d12/d24 scale.

2. **Test keeping MLP weights.** Modify `transfer_nca_to_text()` to only reinit embeddings, value_embeds, and scalars — keep attention AND MLP AND layernorm. The paper's default is to keep everything except embeddings.

3. **Increase rule diversity.** Use more unique rules (our `trajectories_per_rule=128` means we generate many sequences from few rules). The paper uses 16,000 unique rules. Consider generating fewer trajectories per rule but from many more rules.

4. **Match the rule architecture** to the paper's 3-layer design with 4-channel bottleneck.

5. **Consider data regeneration** — fresh NCA data each epoch prevents memorization of specific trajectories.
