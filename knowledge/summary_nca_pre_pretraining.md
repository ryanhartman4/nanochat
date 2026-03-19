# NCA Pre-Pre-Training for Language Models

**Paper:** Han, Lee, Kumar, Agrawal (2026). "Training Language Models via Neural Cellular Automata." arXiv:2603.10055
**Code:** https://github.com/danihyunlee/nca-pre-pretraining

---

## Core Idea

Pre-pre-train a transformer on synthetic data from neural cellular automata (NCA) before standard language pre-training. NCA trajectories have rich spatiotemporal structure resembling natural language (Zipfian token distributions, long-range dependencies) but contain no semantic content — forcing the model to learn pure in-context rule inference.

**Key result:** 164M NCA tokens improves downstream language modeling by up to 6% and accelerates convergence by 1.6x. Outperforms 1.6B tokens of natural language (C4) as pre-pre-training.

---

## NCA Data Generation

### Grid Setup
- **Grid:** 12x12 cells, periodic boundaries
- **Alphabet:** 10 states per cell (one-hot encoded)
- **Transition rule:** Random neural network (3x3 conv → MLP with hidden size 16 → ReLU → 10 logits per cell)
- **Stochasticity:** τ = 10^-3 in softmax sampling

### Tokenization
- **2x2 non-overlapping patches** → each patch maps bijectively to a token
- Vocab size = 10^4 = 10,000 patch tokens (for n=10 alphabet)
- Grids serialized row-major with `<grid>` / `</grid>` delimiters between timesteps
- Sequences up to 1024 tokens

### Complexity Filtering
- Run gzip on serialized trajectories
- **Compression ratio** r = compressed_bytes / raw_bytes × 100
- Retain NCAs with r > 50% (default for web text/math)
- **Domain-specific tuning:** code benefits from 30-40% band, web text/math from 50%+

### Generation Scale
- 16,000 unique NCA rules per epoch
- 500 trajectory simulations per rule
- Rules resampled every epoch for diversity
- Total: 164M tokens (sufficient for transfer)

---

## Transfer Protocol

### What the paper does:
- Transfer **all weights** (attention, MLP, LayerNorm) from NCA phase
- **Reinitialize only the embedding layers** for the new text vocabulary
- All parameters are updated during subsequent pre-training

### Ablation findings (Figure 5 — critical for nanochat):
- **Attention layers are the most transferable** — reinitializing attention causes the largest degradation
- **MLP transfer is domain-dependent** — helps for code, hurts for web text (MLP encodes NCA-specific statistics)
- **LayerNorm transfer is mostly neutral**
- This suggests: **reinitializing everything except attention** (our plan) is well-supported by the paper

---

## Key Findings for nanochat

### 1. Alphabet size tradeoffs
- n=2 scales most favorably (continues improving where n=10,15 plateau)
- Smaller alphabet → simpler dynamics → more consistent transferable structure
- **Implication for nanochat:** Our plan to use 256 grid patch tokens needs reconsideration. The paper uses 10,000 tokens (10^4 from 2x2 patches on 10-color grid). With n=2 and 2x2 patches, that's only 2^4 = 16 tokens. We should test multiple alphabet sizes.

### 2. Complexity matching to target domain
- Higher complexity NCA → better for web text and math
- Lower complexity NCA → better for code
- nanochat trains on ClimbMix (general web text) → use 50%+ gzip band

### 3. Scale of NCA training
- Only 164M tokens needed (minutes of compute)
- Diminishing returns beyond intermediate budgets for larger alphabets
- **Perfect for the speedrun:** negligible wall-clock cost, significant convergence speedup

### 4. Model scale effects
- Transfer gains decrease with model scale (8.6% at 400M → 5.7% at 1.6B)
- nanochat d24 is ~350M active params → expect gains in the 6-9% range
- Even at their largest (1.6B), convergence speedup was 1.4-1.6x

---

## Adapting for nanochat

### Differences from the paper's setup

| Aspect | Paper | nanochat adaptation |
|--------|-------|---------------------|
| Framework | JAX/Flax | PyTorch |
| Model | 1.6B Llama (24L, 32H, 2048D) | d12 (~124M) or d24 (~350M) |
| NCA generation | On-the-fly | Pre-generated dataset |
| Tokenizer | tiktoken | RustBPE |
| NCA vocab | 10,000 tokens (n=10, patch=2) | TBD — should experiment |
| Transfer | Reinit embeddings only | Reinit everything except attention |
| Training | Separate scripts | Inline in base_train.py |
| Optimizer | AdamW | Muon + AdamW |

### Implementation plan for nanochat

1. **NCA data generator** (new script: `scripts/nca_generate.py`)
   - 12x12 grid, periodic boundaries
   - Random neural network transition rules (conv + MLP)
   - Serialize with patch tokenization
   - Filter by gzip complexity (50%+ band)
   - Save as parquet or torch tensors

2. **NCA pre-pre-training stage** (modify `scripts/base_train.py`)
   - After model init, before main training loop
   - Create temporary embedding + LM head sized to NCA vocab
   - Train for N steps on NCA data (parameterized, start at ~164M tokens)
   - After NCA phase: keep attention weights, reinitialize everything else
   - Re-create standard embeddings and LM head for text vocabulary

3. **Key decisions still needed:**
   - NCA vocab size: 16 tokens (n=2) vs 10,000 (n=10) vs our planned 256
   - Number of NCA training steps / tokens
   - Whether Muon optimizer works for NCA or needs different hyperparameters
   - Gzip filtering threshold

### Open question: NCA vocab size

Our spec says 256 grid patch tokens, but the paper's findings suggest:
- n=2 (16 tokens with 2x2 patches) scales best
- n=10 (10,000 tokens) was the default
- n=15 (50,625 tokens) was tested but plateaus faster

256 tokens would correspond to ~n=4 (4^4 = 256). This is actually a reasonable middle ground, but we should A/B test against n=2 (16 tokens) since the paper found it scales better.

---

## References within codebase

- `nanochat/gpt.py` — Model architecture. NCA stage needs temporary embedding/head layers.
- `scripts/base_train.py` — Training loop. NCA stage inserts between model init (line 151) and main loop (line 415).
- `nanochat/tokenizer.py` — Text tokenizer. NCA uses its own patch tokenization (separate from this).
- `nanochat/dataloader.py` — Data loading. NCA needs its own loader (simpler — just load pre-generated tensors).
