# Learning to Reason without External Rewards (Intuitor / RLIF)

**Paper:** Zhao, Kang, Feng, Levine, Song (2025). "Learning to Reason without External Rewards." arXiv:2505.19590 (ICLR 2026)
**Code:** https://github.com/sunblaze-ucb/Intuitor
**Affiliation:** UC Berkeley + Yale

---

## Core Idea

Intuitor proposes **Reinforcement Learning from Internal Feedback (RLIF)** — using a model's own confidence ("self-certainty") as the sole reward signal, with zero external supervision. Self-certainty is the average KL divergence between a uniform distribution and the model's next-token prediction distribution. Higher self-certainty = the model is more "sure" of its outputs. This replaces gold-standard answers in GRPO (Group Relative Policy Optimization), enabling fully unsupervised RL training.

The key insight: LLMs exhibit lower confidence on difficult/incorrect outputs. By rewarding higher self-certainty via GRPO, the model learns to produce more detailed reasoning steps and more confident answers — without ever seeing correct solutions.

## Method

### Self-Certainty Reward
- Self-certainty(o|q) = (1/|o|) Σ KL(Uniform || p_model(·|q, o_{<i}))
- Higher = more confident (model's distribution is further from uniform)
- Mode-seeking (KL with model as second arg), less prone to length bias than entropy
- Replaces external reward in GRPO's advantage computation: A_i = (u_i - mean) / std

### Training Pipeline
- Standard GRPO but with self-certainty scores instead of gold-answer matching
- Sample G=7 candidate solutions per query, score each with self-certainty
- Policy gradient update favoring high-confidence outputs
- KL penalty β=0.005 against reference policy to prevent collapse
- Trained on MATH dataset (7,500 problems), no gold answers used

## Key Findings

1. **Matches GRPO on in-domain math:** Qwen2.5-3B + Intuitor achieves GSM8K 79.2% and MATH500 61.2% vs GRPO's 82.6% and 63.6% — close but slightly below
2. **Better out-of-domain generalization:** On LiveCodeBench, Intuitor achieves 15.3% (65% relative improvement from base) vs GRPO's 8.5% (no improvement). On CRUXEval, Intuitor gets 41.6% (+76%) vs GRPO's 34.1% (+44%)
3. **Works on tiny models:** Qwen2.5-1.5B base model goes from 0% on LiveCodeBench to 9.9% — learns coherent reasoning chains from scratch
4. **Emergent structured reasoning:** Model develops step-by-step reasoning without being explicitly taught to do so
5. **AlpacaEval improvement:** 3.72 → 7.10 (better than GRPO's 6.91), indicating improved instruction following
6. **Domain-agnostic:** No need for domain-specific verifiers — just needs prompts

---

## Relevance to This Project

### Applicability to NEXUS-Sprint: **LOW**

**Why it doesn't help the speedrun:**

1. **This is a post-training technique (RL after pre-training).** The speedrun measures wall-clock time to reach CORE ≥ 0.2565 during *pre-training*. Intuitor/RLIF applies after pre-training is done — it's in the same category as nanochat's `chat_rl.py` stage, which runs after SFT.

2. **CORE evaluation doesn't benefit from RL.** CORE measures base model quality (22 downstream tasks, few-shot). It's evaluated before any SFT or RL stage. Self-certainty RL won't affect the pre-training metric.

3. **The speedrun leaderboard stops at CORE threshold.** The RL stage in nanochat's pipeline (`chat_rl.py`) runs after the speedrun timer has already stopped.

4. **Model scale mismatch.** Tested on 1.5B-3B instruction-tuned models. nanochat's base models at d24 (~1.2B params) have no instruction-following ability — they're raw base models. The self-certainty signal might not be meaningful before SFT.

### Applicability to NEXUS-Full: **MEDIUM-HIGH**

For NEXUS-Full, Intuitor is relevant as a **post-training RL stage replacement**:
- No need for gold solutions in math/code — could enable RL on arbitrary domains
- The self-certainty reward could replace or complement the planned MGPO/GRPO RL stage
- 65% relative improvement on code generation from math-only training is impressive generalization
- Could reduce the RL data curation cost significantly

### For nanochat chat_rl.py (outside speedrun):
- Could replace the current RL reward mechanism with self-certainty
- No verifier infrastructure needed — simpler pipeline
- But nanochat's RL stage is already optional and outside the speedrun scope

**Estimated integration effort:** 1-2 days to swap reward function in chat_rl.py. Not relevant for speedrun.
