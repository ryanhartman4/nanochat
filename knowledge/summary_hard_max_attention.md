# Hard-Max Attention / "Can LLMs Be Computers?" — Summary

**Source:** [Percepta blog](https://www.percepta.ai/blog/can-llms-be-computers) (Tzamos et al., March 2026)
**No arxiv paper found** — blog post is the primary publication.

## Core Thesis

Percepta compiled a **WebAssembly (WASM) interpreter directly into transformer weights**, enabling 100% deterministic execution of arbitrary C programs within an autoregressive transformer. Not a metaphor — the weights literally implement an interpreter.

## Key Technical Details

### 2D Attention Heads
- **Architecture:** 7 layers, d_model=36, 18 attention heads → **2 dimensions per head**
- Each historical token's Key vector is 2D; the Query is a direction on a 2D plane
- Finding the best-matching Key becomes a **convex hull extreme value query** (farthest point along Query direction on the convex hull)
- This reduces attention lookup from **O(n) to O(log n)** per step

### HullKVCache
- Dynamically maintains the convex hull of historical Keys during token generation
- Each attention query operates only on the hull, not the full sequence
- This is the mechanism behind "exponentially faster inference"

### Hard-Max vs Softmax
- Standard softmax: weighted average over all keys (O(n))
- Hard-max: selects the single best-matching key (enabled by 2D convex hull → O(log n))
- Deterministic selection rather than probabilistic mixing

### WebAssembly Execution
- C/C++ code compiled to WASM bytecode → tokenized into model input
- Attention + MLP layers collectively implement the interpreter's logic
- Model simulates execution step-by-step, emitting state transitions as tokens
- **Weights are compiled, not learned** — this is the key controversy

## Results
- Solved world's hardest Sudoku in under 3 minutes
- Multi-digit addition with zero errors
- **30,000+ tokens/sec on ordinary CPU** (thanks to O(log n) attention)
- 100% deterministic accuracy on all tested programs

## Controversy
- Weights are **compiled from the interpreter, not trained via gradient descent**
- Critics argue this is "writing a very unusual computer program" rather than training AI
- The real question: can this be extended to hybrid models that learn *when* to compute deterministically vs probabilistically?

## Relevance to NEXUS-Sprint

### Applicability: LOW for speedrun, MEDIUM-HIGH for NEXUS-Full
- The 2D attention / convex hull KV cache is genuinely interesting for inference speed
- But the core result (compiled weights) doesn't help with *training* speed — our bottleneck
- The O(log n) attention only helps at long context (n >> 2048); at nanochat's T=2048, FA3's optimized O(n²) is already fast enough
- **Potential angle:** Could the 2D head restriction be used as a regularizer during training? Forces attention to learn simpler, more structured patterns. Speculative.

### For the speedrun specifically:
- d_model=36 is toy-scale; unclear how 2D heads behave at d=768-1536
- Compiled weights bypass training entirely — not applicable
- HullKVCache could help inference (chat_cli/chat_web) but not the training speed target
