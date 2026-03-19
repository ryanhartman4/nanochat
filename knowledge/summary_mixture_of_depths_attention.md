# Mixture-of-Depths Attention (MoDA)

**Paper:** Zhu, Fang, Liao et al. (2026). "Mixture-of-Depths Attention." arXiv:2603.15619
**Code:** https://github.com/hustvl/MoDA
**Affiliation:** ByteDance Seed + Huazhong University

---

## Core Idea

MoDA addresses **signal degradation** in deep transformers: as models get deeper, informative features from shallow layers get diluted by repeated residual additions. MoDA lets each attention head jointly attend to both the **sequence KV** (standard causal attention) and **depth KV** (KV pairs from the same token position in all preceding layers), unified under a single softmax. This creates cross-layer shortcuts without the O(L²D²) cost of dense connections.

Think of it as: every layer can look back at what all previous layers computed at each token position, and dynamically decide which historical layer's representation is most useful right now.

## Method

### Architecture
- **Depth KV stream:** Each layer produces KV pairs that are appended to a growing depth cache. Layer l's attention can read KV from layers 0..l-1 at each token position.
- **Unified softmax:** Sequence attention scores and depth attention scores are combined in one softmax — no separate attention operation needed.
- **FFN KV projections:** Optionally, FFN layers also project their input to depth KV (this is the best variant — row 4 in their ablation).
- **No extra query projections needed:** Reuses the existing Q from sequence attention for depth attention.

### Complexity
| Mechanism | Parameters | Prefill FLOPs |
|-----------|-----------|---------------|
| Depth Dense | O(L²D²) | O(TL²D²) |
| Depth Attention | O(LD²) | O(TL²D) |
| **MoDA** | **O(LD²/G)** | **O(TL²D)** |

MoDA is the most parameter-efficient, with same FLOPs as depth attention but fewer params due to GQA reuse.

### Hardware-Efficient Implementation
- Custom Triton kernel fuses sequence and depth attention in one pass with shared online-softmax states
- **Chunk-aware depth KV layout:** Groups queries by chunk, each chunk only accesses its local depth span (not the full T×L)
- **Group-aware indexing:** G adjacent query rows share the same base-time → reuse same depth KV blocks
- Achieves **97.3% of FlashAttention-2 efficiency** at 64K sequence length
- Extra time overhead: **2.73% at T=65536**, 8.59% at T=16384 (with G=8, L=64)

## Key Findings

1. **+2.11% average downstream improvement at 1.5B** (400B tokens, OLMo2 recipe): 62.28 → 64.39 across 10 tasks (PIQA, HellaSwag, WinoGrande, OpenBookQA, BoolQ, SciQ, ARC-E, ARC-C, COPA, MMLU)
2. **+1.76% at 700M** with same recipe: 57.11 → 58.87
3. **Validation PPL improves across all 10 domains** at both scales (average PPL: 15.61→15.46 at 700M, 13.67→13.47 at 1.5B)
4. **Negligible compute overhead:** Only 3.7% extra FLOPs for the recommended variant (Seq KV + Depth KV + FFN KV Proj)
5. **Post-norm + MoDA works better than pre-norm + MoDA**, especially in deeper models (48-layer: postnorm improvement 0.0409 vs prenorm 0.0041)
6. **Reusing attention KV as depth KV is free** (0 extra params, 0.12% extra FLOPs) and still provides +1.17% downstream gain
7. **FFN depth KV matters:** Adding FFN-side KV projection is the best accuracy-efficiency tradeoff
8. **Reduces attention sink behavior:** Attention mass redistributes from fixed sink positions toward informative sequence/depth slots
9. **Works at both 24 and 48 layers** — consistent gains regardless of depth

---

## Relevance to This Project

### Applicability to NEXUS-Sprint: **HIGH**

This is one of the most promising ideas for the speedrun. Here's why:

**Why it could help:**
- nanochat's d24-d26 models are exactly the depth regime where signal degradation matters. 24-26 layers of repeated residual additions dilute early-layer features.
- The paper shows gains even at 24 layers (3.4740 → 3.4338 val loss with full MoDA), and nanochat operates at d24-d26.
- The overhead is tiny: 3.7% extra FLOPs for +2% downstream improvement. For the speedrun, this means either (a) same wall-clock with better CORE score, or (b) fewer training steps needed to hit CORE threshold.
- **The zero-param variant** (just reusing attention KV as depth KV, no FFN projections) adds 0.12% FLOPs and still provides +1.17% downstream gain. This is essentially free.
- nanochat uses GQA — MoDA is designed for GQA and gets more efficient with larger G.
- Triton kernel exists and is compatible with FA2. nanochat uses FA3 but the same principles apply.

**What would need to change in nanochat:**
1. `nanochat/gpt.py`: Modify the attention block to maintain a depth KV cache across layers. Each layer appends its KV to the depth stream and reads from it.
2. `nanochat/gpt.py`: Add optional FFN KV projection (small Linear(D, D/G) per FFN layer).
3. Need a custom Triton kernel or adapt the MoDA kernel for FA3. Alternatively, start with a naive PyTorch implementation and only optimize if it shows gains.
4. `nanochat/engine.py`: Inference KV cache needs to include depth KV. At T=2048, the depth stream adds L×d per-token overhead — small at d24.
5. Consider switching to **post-norm** (paper shows MoDA + postnorm > MoDA + prenorm). nanochat currently uses... (need to check gpt.py).

**Risk assessment:**
- LOW risk for the zero-param variant (just depth KV reuse)
- MEDIUM risk for full MoDA with FFN projections (adds parameters, need to tune)
- The Triton kernel is open-source but targets FA2 — may need adaptation for nanochat's FA3 setup
- `torch.compile` compatibility needs testing

**Key question for the speedrun:**
The paper measures quality improvement (perplexity, downstream accuracy). For the speedrun, we need **convergence speed** — does MoDA reach the CORE threshold in fewer steps? The +2% downstream improvement strongly suggests yes, but the paper doesn't report step-to-quality curves. We'd need to test this empirically on d12.

**Estimated integration effort:** 2-3 days for naive PyTorch implementation, 4-5 days with Triton kernel.
