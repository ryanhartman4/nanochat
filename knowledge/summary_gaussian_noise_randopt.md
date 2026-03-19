# Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights (RandOpt)

**Paper:** Gan, Isola (2026). "Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights." arXiv:2603.12228
**Code:** https://github.com/sunrainyg/RandOpt
**Affiliation:** MIT CSAIL

---

## Core Idea

After sufficient pretraining, the neighborhood around a model's weights is densely populated with task-specific "expert" solutions — random Gaussian perturbations of the pretrained weights often improve performance on specific tasks. The paper calls this the **"thicket regime"**: large, well-pretrained models have so many good solutions nearby that random guessing becomes viable as a post-training method.

**RandOpt** exploits this: sample N random weight perturbations, evaluate each on target data, select top K, and ensemble predictions via majority vote. Despite its simplicity, RandOpt is competitive with PPO, GRPO, and evolutionary strategies (ES) on reasoning tasks — and runs in O(1) sequential steps (fully parallel).

## Method

### RandOpt Algorithm
1. Start with pretrained weights θ
2. Sample N perturbations: θ_i = θ + σ·ε_i where ε_i ~ N(0, I)
3. Evaluate each θ_i on target training data
4. Select top K perturbations by score
5. Ensemble predictions via majority vote at inference time
6. σ ≈ 0.005 is the typical noise scale

### Key Phenomena
- **Solution density** scales with model size: larger models have more task-improving perturbations nearby
- **Solution diversity** also scales: perturbations are specialists, not generalists — improving one task while hurting others
- **Three regimes:** (1) "Needle in haystack" (small/untrained models — no good solutions nearby), (2) "Thicket" (large pretrained models — dense with experts), (3) "Plateau" (already optimal on target task)
- Spectral discordance D increases with scale, confirming specialists get more diverse with size

### Results
- Tested on Qwen2.5 family (0.5B to 32B), OLMo-3-7B
- Tasks: Countdown, GSM8K, MATH-500, OlympiadBench, MBPP, ROCStories, USPTO
- RandOpt with 5000 random guesses + top-K ensemble is competitive with GRPO and ES
- O(1) training time (all perturbations evaluated in parallel) vs O(T) for gradient-based methods
- Inference cost is K× higher due to ensembling (can be reduced via distillation)

## Key Findings

1. **Density scales with model size:** At 32B, ~15-20% of random perturbations improve GSM8K accuracy. At 0.5B, this drops to <1%.
2. **Perturbations are specialists:** Individual perturbations improve specific tasks while degrading others. PCA clustering reveals distinct expert groups.
3. **RandOpt is competitive at large scale:** On OLMo-3-7B Countdown task, RandOpt matches GRPO converged accuracy with 5000 random guesses.
4. **Thicket regime emerges from diverse pretraining:** In toy experiments, mixed-signal pretraining creates thickets; single-signal pretraining creates plateaus.
5. **The finding is about the loss landscape, not the method:** RandOpt is a probe showing that post-training becomes "easy" once you have strong pretrained representations.

---

## Relevance to This Project

### Applicability to NEXUS-Sprint: **VERY LOW**

**Why it doesn't help the speedrun:**

1. **This is a post-training technique.** RandOpt applies AFTER pretraining is complete. The speedrun measures wall-clock to CORE threshold during pretraining. Random weight perturbation + ensemble is irrelevant to the training speed question.

2. **CORE is a base model metric.** CORE evaluates the pretrained model directly (few-shot inference). RandOpt creates ensembles of perturbed models — you'd need to run CORE on K different models and aggregate, which isn't how the leaderboard works.

3. **Scale mismatch:** The thicket phenomenon requires "large, well-pretrained" models (7B+). nanochat's d24 is ~1.2B — squarely in the "needle in haystack" regime where random perturbations rarely improve performance.

4. **Not a training acceleration technique.** The paper explicitly states RandOpt is a "probe" for understanding the loss landscape, not a training method. It doesn't make pretraining faster.

### Potential Tangential Use
- **During hyperparameter search:** Instead of grid search for LR/batch size, random perturbation of model weights could explore the landscape. But this is speculative and not what the paper proposes.
- **For NEXUS-Full post-training:** At 2B+ scale, RandOpt could be an interesting alternative to standard RL. But that's outside the sprint scope.

**Verdict: Skip for the speedrun entirely.** The paper is fascinating science about the geometry of pretrained weight spaces, but provides no actionable technique for accelerating pretraining convergence.

**Estimated integration effort:** N/A for speedrun. 1-2 days for post-training experiments if desired.
