# Speculative Ideas Assessment: Recursive Prompts + Attention Layer Shortcuts

**Sources:** No specific papers — these are speculative architectural ideas from brainstorming.

---

## Idea #5: Recursive Prompts (Re-inject Original Prompt Midway)

### Concept
Instead of the input embedding only appearing at layer 0, re-inject the original prompt representation at an intermediate layer (e.g., layer L/2). This creates a "reminder" of the original input partway through the network, potentially helping with instruction following and preventing the model from drifting away from the input context in deeper layers.

### Prior Art
- **Prefix Tuning** (Li & Liang, 2021): Prepends trainable tokens to every attention layer's KV cache. Similar in spirit but uses learned soft prompts, not the original input.
- **Deep Prompt Tuning** (VPT-Deep, Jia et al., 2022): Injects trainable embeddings at every transformer layer. Used in vision transformers.
- **LLaMA-Adapter** (Zhang et al., 2023): Injects learnable prompts at top L layers with zero-init gating.
- **Universal Transformers** (Dehghani et al., 2018): Shares weights across depth with position-dependent embeddings — conceptually similar to re-injection.

### Analysis for the Speedrun

**Why it probably won't help:**

1. **CORE evaluates base model quality, not instruction following.** Re-injecting the input prompt helps with staying on-task during generation, but CORE's 22 tasks are few-shot evaluations where the model processes a prompt once and predicts. The model doesn't "drift" during CORE eval — it's a single forward pass.

2. **The residual stream already preserves input.** Standard residual connections mean the input embedding is already summed into every layer's hidden state (h_l = h_0 + Σ f_i). Re-injection would double-count the input at the injection point. This is essentially what AttnRes already does more elegantly — with learned weights instead of a fixed re-injection.

3. **Pre-training quality ≠ instruction following quality.** Recursive prompts address a generation-time problem. Pre-training optimizes next-token prediction, where the model always has the full context in the residual stream.

4. **AttnRes subsumes this idea.** AttnRes (idea #3) lets every layer selectively attend to the input embedding with learned softmax weights. This is a strictly more general and principled version of "re-inject the input at layer L/2."

**Verdict: Skip.** AttnRes already does this better. Not worth independent implementation.

---

## Idea #9: Attention Layer Shortcuts / Looped Attention 3x

### Concept
Instead of a 24-layer deep stack, use fewer unique layers with skip connections across them, or loop a smaller attention block 3x to achieve effective depth without the parameter/compute cost of 24 unique layers.

### Prior Art
- **DenseNet / DenseFormer** (Pagliardini et al., 2024): Dense cross-layer connections with learned scalar weights.
- **Hyper-Connections** (Zhu et al., 2025): Multi-stream residual recurrences with mixing matrices.
- **Looped Transformers** (multiple papers, see task #5 summary): Weight-shared layers looped N times.
- **AttnRes** (arXiv:2603.15031): Softmax attention over depth — selective skip connections.
- **DCA** (arXiv:2502.06785): Cross-layer Q/K/V composition from historical layer outputs.
- **MoDA** (arXiv:2603.15619): Depth KV in attention — cross-layer information retrieval.

### Analysis for the Speedrun

**This is a cluster of ideas, not a single technique.** Breaking it down:

1. **Skip connections across attention layers:** This IS AttnRes/DCA/MoDA. We've already assessed these as HIGH priority. No additional idea here — just implement the papers.

2. **Looped attention 3x instead of deeper layers:** Same as idea #4 (Looping Layers). Assessed as LOW-MEDIUM for speedrun — saves parameters but not FLOPs/wall-clock.

3. **Hybrid: fewer unique layers + cross-layer attention:** Could combine looping (8 unique layers × 3 loops = 24 effective depth) with AttnRes (selective access across loops). But this compounds complexity: you need both looping infrastructure AND cross-layer attention. For the sprint, better to do AttnRes/DCA on the existing 24-layer stack.

**Verdict: Subsumed by ideas #2, #3, #4.** The specific architectural proposals here are already covered by MoDA, DCA, AttnRes, and Looping Layers. No independent value to implement — focus on the concrete papers instead.

---

## Summary

| Idea | Status | Rationale |
|------|--------|-----------|
| #5 Recursive Prompts | **Skip** | Subsumed by AttnRes (idea #3). AttnRes selectively attends to input with learned weights — strictly better than fixed re-injection. |
| #9 Attention Shortcuts / Looped Attention | **Skip** | Subsumed by ideas #2 (MoDA), #3 (DCA/AttnRes), #4 (Looping). No independent implementation needed. |
