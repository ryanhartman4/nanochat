# Lost in Backpropagation: The LM Head is a Gradient Bottleneck

**Paper:** Nathan Godey, Yoav Artzi (2026). "Lost in Backpropagation: The LM Head is a Gradient Bottleneck." arXiv:2603.10145 (ICML 2026 submission)

---

## Core Idea

The LM head (the final linear projection D→V followed by softmax) is not just an expressivity bottleneck — it's a fundamental **optimization bottleneck**. When D << V, backpropagating V-dimensional gradients through a rank-D linear layer compresses the gradient, destroying 95-99% of the gradient norm. The surviving gradient is misaligned with the true logit gradient, redirecting signal energy from informative top components into noise on the tail coefficients.

This means the vast majority of model parameters (everything below the LM head) only ever see a lossy, low-rank projection of the supervision signal — regardless of architecture, optimizer, or training setup.

## Method

### Theoretical Framework

The authors formalize the LM objective in matrix form: loss = -⟨N, log σ(HW^T)⟩_F / T, where:
- H ∈ R^{C×D}: context representations (from the backbone)
- W ∈ R^{V×D}: LM head weight matrix
- N: context-token count matrix

Key theoretical results:

1. **Expressivity isn't the full story (Prop 3):** Even with D=2, the model can match any top-1 prediction arbitrarily closely. So the softmax bottleneck can't be characterized as just an expressivity issue.

2. **Gradient update is rank-constrained (Eq 5):** The actual logit update Δ has rank ≤ 2D, but the optimal logit gradient (P - Ñ) has rank ≈ V-1 in practice. The Eckart-Young-Mirsky theorem guarantees an unavoidable residual.

3. **SGD doesn't help (Prop 5):** Near convergence, even mini-batch gradients maintain high rank. The bottleneck persists regardless of batch sampling.

4. **Alternative LM heads don't help (Eq 9):** Any function f(H) = L that maps through D-dimensional hidden states has a rank-D Jacobian. The compression is inherent to the D→V dimensional mismatch, not to the specific form of the output layer.

### Experimental Validation

**SpamLang (synthetic):** A trivial language where each sequence repeats one token. The model has perfect expressivity for any D≥2 (Prop 3), but as V grows relative to D, the gradient bottleneck makes this trivial pattern unlearnable. At V=131072 with D=576, no learning rate works.

**2B parameter LLMs:** 8 models sharing the same Llama3 backbone (D_model=4096, 6 layers), but with low-rank LM heads W=AB where A∈R^{V×D}, B∈R^{D×D_model}, controlling the bottleneck strength. Results:
- D=4096 reaches D=32's final loss level within 700M tokens — a **16x convergence speedup**
- Even D=2048 vs D=4096 shows a +0.55 average benchmark gap after 11B tokens
- The effect is consistent across all downstream tasks

**Gradient analysis on Pythia/Llama/Qwen:** The logit gradient empirically reaches near-full rank (≈V-1) for batch sizes >10K tokens. After projection through W^T, 95-99% of the gradient norm is destroyed. The surviving gradient has only 0.1-0.2 cosine similarity with the original.

## Key Findings

1. **95-99% of gradient norm is destroyed** at the LM head for all tested model families (GPT-2, Pythia, Llama3, Qwen3-Base), following a consistent trend as D/V increases.

2. **The gradient compression is destructive, not just lossy:** It transfers energy from the informative top singular value components to the tail as random noise. The backbone receives a corrupted training signal.

3. **16x convergence gap** between D=32 and D=4096 for the same 2B backbone, isolating the bottleneck from architecture effects.

4. **The bottleneck worsens near convergence:** As model predictions approach the data distribution, the prediction error matrix (P-Ñ) becomes increasingly high-rank, making the compression more severe exactly when fine-grained learning matters most.

5. **No existing alternative LM head design fixes this** from a first-order optimization perspective. The rank-D Jacobian constraint applies to any function mapping D-dimensional hidden states to V-dimensional logits.

6. **The authors explicitly leave the solution as future work:** "promising directions for innovations that better preserve gradient flow, whether through pre-conditioning, optimization techniques, or well-suited softmax alternatives."

---

## Relevance to This Project

### The nanochat gradient bottleneck numbers

For nanochat d24 with the 24K tokenizer:
- D = 1536 (n_embd for d24 at aspect_ratio=64)
- V = 24576
- D/V = 0.0625

From the paper's Figure 4, at D/V ≈ 0.06, approximately **97% of gradient norm is destroyed**. This is severe.

With the original 32K tokenizer:
- V = 32768
- D/V = 0.047
- Even worse — approximately **98% gradient norm destroyed**

So our A.2.1 tokenizer reduction (32K → 24K) does help the D/V ratio (0.047 → 0.063), but the improvement is from 98% destroyed to 97% destroyed. That's directionally correct but not a game-changer.

### Why "just shrink vocab" is incomplete

The paper reveals that the gradient bottleneck is fundamentally a rank mismatch problem. Shrinking V helps the D/V ratio, but:
1. **The effect is logarithmic, not linear.** Going from 32K→24K is a 25% vocab reduction but only moves D/V from 0.047 to 0.063.
2. **The real lever is D, not V.** The paper shows D=4096 vs D=2048 gives massive gains. Increasing D would help much more than decreasing V.
3. **But nanochat can't increase D** without changing the model size (D is coupled to depth via the aspect ratio). So vocab reduction is the accessible lever, even if it's weak.

### What the paper suggests we should actually do

The paper explicitly identifies this as an open problem and suggests three directions:

1. **Pre-conditioning:** Modify the optimization to compensate for the gradient compression. This could mean different learning rates for the backbone vs head, or gradient scaling based on the projection loss.

2. **Optimization techniques:** The Muon optimizer used by nanochat already provides some relief — it uses Newton-style updates that are less affected by gradient direction noise. But the 95-99% norm destruction still applies to the gradient signal Muon receives.

3. **New LM head designs:** The paper shows that current alternatives (MoS, sigsoftmax, etc.) don't fix the first-order optimization issue. But they suggest that designs which "provide less lossy compression through J_f" could help. This could mean:
   - **Factored heads:** D → D_intermediate → V with D_intermediate > D (breaks the rank constraint)
   - **Multi-head prediction:** Multiple D→V_small projections instead of one D→V
   - **Hierarchical softmax:** Decompose V into a tree, reducing the per-projection dimensionality

### Why head avoidance is theoretically well-motivated

Our `exp/head-avoidance` branch bypasses the LM head on proxy steps, computing cosine similarity between hidden states and `lm_head.weight[targets]` instead. The paper's Eq. 9 (Jacobian analysis) shows this is the most direct response to the bottleneck:

1. **Eq. 9 proves the bottleneck is universal for ANY head design.** For any f(H) = L, the Jacobian J_f has rank ≤ D. No alternative softmax, factored head, or MoS will fix the first-order optimization issue. The only way to fully bypass it is to skip the head entirely — which is exactly what proxy steps do.

2. **The proxy gradient is full-rank in D.** When we compute `cosine_similarity(hidden, lm_head.weight[y])`, the gradient flows directly into the D-dimensional hidden space without passing through the V×D bottleneck. Every direction in D-space receives useful signal, not just the D directions that survive W^T projection.

3. **The bottleneck worsens near convergence (Prop 5 + Corollary 1).** As model predictions approach the data distribution, the prediction error matrix (P-Ñ) becomes increasingly high-rank, making the compression more severe exactly when fine-grained learning matters most. This suggests head avoidance might be MORE valuable later in training — which challenges our current `--head-avoidance-anneal` design that reduces proxy frequency over time.

4. **Gradient direction quality, not just norm.** The appendix (§C, Fig. 7) shows the cosine similarity between projected and original gradient is only 0.1-0.2. The backbone doesn't just receive a weaker signal — it receives a nearly orthogonal one. Our proxy loss provides a direct, high-quality directional signal toward the correct decoder weight vectors.

### Potential issues with the current head avoidance design

1. **Annealing direction may be backwards.** The `--head-avoidance-anneal` flag starts aggressive (every-2) and tapers to every-K. But Prop 5 says the bottleneck gets worse near convergence. We might want to INCREASE proxy frequency later in training, not decrease it.

2. **Proxy loss doesn't train the head.** On proxy steps, lm_head receives zero gradients (we zero-fill None grads). This means the head's weight matrix W — which defines the decoder space our proxy targets — only updates on CE steps. If the head falls behind the backbone's representations, the proxy targets become stale. Need to ensure enough CE steps to keep W aligned.

3. **The paper's D=2048 vs D=4096 gap (+0.55 avg score at 11B tokens)** suggests the bottleneck matters even at high D. For nanochat d24 (D=1536, D/V=0.047), we're in severe bottleneck territory. Head avoidance could be disproportionately valuable here compared to larger models.

### Concrete implications for the speedrun

1. **The 24K tokenizer helps but is a weak lever.** D/V goes from 0.047 to 0.063. The gradient bottleneck goes from ~98% to ~97% destroyed. Real, but not transformative.

2. **Head avoidance is the most theoretically grounded speedrun lever.** Unlike vocab reduction (marginal D/V improvement) or factored heads (still rank-constrained per Eq. 9), proxy steps completely sidestep the bottleneck on those steps. This is the paper's implied "nuclear option."

3. **NCA and head avoidance attack orthogonal bottlenecks.** NCA bootstraps attention induction circuits (architecture/initialization). Head avoidance improves gradient flow (optimization). They should compound — NCA gives better starting attention patterns, and head avoidance ensures the backbone receives uncorrupted gradient signal to build on them.

4. **The SpamLang experiment is a cautionary tale for small models.** At d12 (D=768, V=24576), D/V = 0.031. The paper's Figure 4 puts this at ~98.5% gradient destruction. This may partly explain why our d12 CORE score (0.149) was low — the gradient bottleneck is crippling at that scale, independent of NCA.

5. **Open question: what K is optimal?** The paper doesn't prescribe a fix, so we're in uncharted territory. The d12 H200 results (BPB 0.887 with K=3) need the matched baseline to know if head avoidance helps. If it does, sweeping K and testing reverse-annealing (increasing proxy frequency over training) could yield further gains.
