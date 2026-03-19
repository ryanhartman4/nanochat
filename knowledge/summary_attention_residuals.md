# Attention Residuals (AttnRes)

**Paper:** MoonshotAI (2026). "Attention Residuals." arXiv:2603.15031
**Code:** https://github.com/MoonshotAI/Attention-Residuals
**Affiliation:** Moonshot AI (Kimi)

---

## Core Idea

Standard residual connections accumulate all layer outputs with fixed unit weights: h_l = Σ f_i(h_i). This causes hidden-state magnitudes to grow as O(L) with depth, progressively diluting each layer's contribution (the "PreNorm dilution" problem). AttnRes replaces this with **softmax attention over depth**: each layer has a single learned pseudo-query w_l ∈ R^d, and the layer's input is a softmax-weighted combination of all previous layer outputs, with weights computed via dot-product attention. This is exactly the same linear→softmax transition that transformed sequence modeling (RNNs → Transformers), but applied over the depth dimension.

**Block AttnRes** is the practical variant: layers are partitioned into N blocks (~8), with standard residuals within blocks and softmax attention across block-level representations. This reduces overhead from O(Ld) to O(Nd) while preserving most gains.

## Method

### Full AttnRes
- **Query:** Learnable vector w_l ∈ R^d per layer (initialized to zero → uniform weights at start)
- **Keys/Values:** Previous layer outputs v_i = f_i(h_i), with RMSNorm on keys to prevent magnitude domination
- **Attention:** Standard softmax: α_{i→l} = softmax(w_l · RMSNorm(v_i))
- **Layer input:** h_l = Σ α_{i→l} · v_i
- **Cost:** O(L²d) compute, O(Ld) memory — manageable since L << T
- **Extra params:** Just one d-dimensional vector per layer (negligible)

### Block AttnRes
- Partition L layers into N blocks of S=L/N layers each
- **Intra-block:** Standard residual summation (b_n = Σ f_j(h_j) for j in block n)
- **Inter-block:** Softmax attention over N block representations + token embedding
- N≈8 recovers most gains; loss degrades gracefully as block size increases
- Memory: O(Nd) instead of O(Ld)
- Inference latency overhead: <2%

### Key Design Choices (from ablation)
- **Input-independent query** (learned w_l) preferred over input-dependent projection (saves d×d params per layer)
- **Softmax > sigmoid** (competitive normalization forces sharper selection)
- **Single-head > multi-head** for depth attention (optimal depth mixture is uniform across channels)
- **RMSNorm on keys is critical** — prevents layers with larger outputs from dominating
- **Zero init mandatory** — starts as uniform averaging, prevents training instability

## Key Findings

1. **1.25x compute advantage:** Block AttnRes matches baseline loss at 1.25× less compute across scaling law sweep (194M-528M active params)
2. **Validated at 48B scale (3B active):** Kimi Linear architecture, 1.4T tokens, Block AttnRes with 9 blocks. Improves on ALL downstream benchmarks
3. **Strongest on reasoning tasks:** GPQA-Diamond +7.5, Minerva Math +3.6, HumanEval +3.1, MMLU +1.1, TriviaQA +1.9
4. **Mitigates PreNorm dilution:** Output magnitudes stay bounded (periodic reset at block boundaries), gradient distribution becomes more uniform across depth
5. **Scaling curves:** Baseline L = 1.891 × C^{-0.057}, Block AttnRes L = 1.870 × C^{-0.058} — consistent gap across entire compute range
6. **Shifts optimal architecture toward deeper/narrower:** Under fixed compute budget, AttnRes optimal at d_model/L_b ≈ 45 vs baseline optimal at ≈ 60. AttnRes exploits depth more effectively.
7. **Block size N≈8 is sufficient:** S=2,4,8 all land near same loss; larger blocks degrade toward baseline
8. **Beats DenseFormer (fixed weights, no gain) and mHC (1.747 vs AttnRes 1.737)** — input-dependent softmax selection is key
9. **Sliding window (last 8 layers) worse than Block AttnRes** — selectively accessing distant layers matters more than attending to many nearby ones
10. **Learned patterns show preserved locality + selective skip connections** — layers mostly attend to predecessors but develop cross-block shortcuts

---

## Relevance to This Project

### Applicability to NEXUS-Sprint: **HIGH**

AttnRes and DCA (arXiv:2502.06785) attack the same problem from similar angles. Here's how AttnRes compares and why it matters for the speedrun:

**Why it's a strong candidate:**
- nanochat d24-d26 = 24-26 layers. The scaling law experiments show consistent gains at 12-17 blocks (24-34 layers) — exactly our range.
- **Minimal implementation complexity:** One learned vector per layer + softmax over previous outputs. No new projection matrices, no kernel changes.
- **Block AttnRes with N≈8** is the practical variant. For d24 that's blocks of 3 layers each — very manageable.
- The 1.25x compute advantage directly translates to the speedrun: if baseline takes 99 min, AttnRes should take ~79 min.
- **Zero-init design** means it's safe — starts equivalent to standard residuals, gradually learns to be selective.
- **Inference overhead <2%** — no impact on chat_cli/chat_web usability.

**AttnRes vs DCA for the speedrun:**
| Aspect | AttnRes | DCA |
|--------|---------|-----|
| Convergence speedup | 1.25x (scaling law fit) | 3x (wall-clock to match PPL) |
| Implementation | 1 vector per layer + softmax | 3 GRN instances per block (Q/K/V) |
| Kernel changes | None | None |
| Extra params | ~Ld (negligible) | ~3×4×L (negligible) |
| Validated scale | 48B / 1.4T tokens | 449M / 131B tokens |
| Architecture mods | Residual connection only | Attention input composition |
| torch.compile | Should work (simple ops) | Should work (simple ops) |
| **Combinability** | **Could combine with DCA** | **Could combine with AttnRes** |

**Key insight: These are complementary.** AttnRes modifies how layer outputs aggregate (the residual stream). DCA modifies how Q/K/V are composed from historical layers. They operate on different parts of the forward pass and could stack.

**What would need to change in nanochat:**
1. `nanochat/gpt.py`: Add a pseudo-query parameter per transformer block (nn.Parameter, d-dimensional, zero-init). Before each block, compute softmax attention over stored block representations to form the input.
2. For Block AttnRes (N=8): accumulate within-block sums, store block-level representations in a list, attend across blocks at block boundaries.
3. `nanochat/engine.py`: Inference needs the block representation cache — just N d-dimensional vectors per token position. Tiny overhead.
4. RMSNorm on keys — nanochat already has RMSNorm infrastructure (used in attention).

**Risk assessment:**
- VERY LOW risk: zero-init means it starts identical to current model, no instability
- No kernel changes, pure PyTorch, torch.compile compatible
- Single d-dimensional vector per layer — memory overhead is unmeasurable

**Estimated integration effort:** 1 day for Block AttnRes. Simpler than DCA (fewer moving parts).

**Recommendation:** Implement AttnRes first (simpler, 1 day), then DCA on top. If AttnRes alone gives 1.25x, we'd hit ~79 min. If DCA stacks to give even partial additional gains, sub-1hr is achievable.
