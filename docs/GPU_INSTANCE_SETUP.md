# GPU Instance Setup Guide

Step-by-step bootstrap for running nanochat experiments on a fresh GPU instance.
GPU Claude is loaded directly onto the instance — no SSH step needed.

---

## 1. System Dependencies

```bash
# Triton needs python3-dev for compilation
apt-get update && apt-get install -y python3-dev
```

Check CUDA is available:
```bash
nvidia-smi
```

Expect: H100/H200 80GB+, CUDA 12.x+, driver 550+.

## 2. Clone the Repo

```bash
cd /root
git clone https://github.com/ryanhartman4/nanochat.git
cd nanochat
```

If already cloned from a prior session:
```bash
cd /root/nanochat
git fetch origin
```

## 3. Checkout the Experiment Branch

Check the handoff doc for which branch to use. Example:
```bash
git checkout exp/attention_improvements
```

## 4. Install Python Dependencies

```bash
uv sync --extra gpu
```

This installs PyTorch with CUDA 12.8, Flash Attention, wandb, and all other deps into `.venv/`.

**Important:** Always use `uv run python` (not bare `python`) for one-off commands, or activate the venv first:
```bash
source .venv/bin/activate
```

## 5. Verify the Setup

### Quick smoke test (d4, ~20 steps, ~1 min):
```bash
OMP_NUM_THREADS=4 uv run python -m scripts.base_train \
    --depth=4 --run=dummy --num-iterations=20
```

Should complete without errors. Check for:
- `Flash Attention 3` detected (not SDPA fallback)
- No OOM errors
- Loss decreasing

### Check FA3 detection:
```bash
uv run python -c "from nanochat.flash_attention import flash_attn; print(flash_attn)"
```

## 6. Disk Space Workaround

Some instances have small overlay filesystems (16GB). Training data + torch.compile cache can exceed this.

If disk is tight:
```bash
# Route nanochat cache to RAM-backed tmpfs
mkdir -p /dev/shm/nanochat_cache
ln -sf /dev/shm/nanochat_cache ~/.cache/nanochat

# Route torch compile cache off overlay
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/torchinductor
```

## 7. Run the Experiment

Check the handoff doc (`docs/HANDOFF_*.md`) for the specific run command. General patterns:

### Single GPU d12 (quick experiment, ~30 min):
```bash
OMP_NUM_THREADS=4 uv run python -m scripts.base_train \
    --depth=12 \
    --run="d12-experiment-name" \
    --model-tag="d12-experiment-name" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
```

### 8xGPU d24 (leaderboard-scale, ~90 min):
```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --depth=24 \
    --run="d24-experiment-name" \
    --model-tag="d24-experiment-name" \
    --fp8
```

**Note:** `torchrun` commands must be on a single line (shell splits multi-line into separate commands).

### Known d24 constraints:
- `--device-batch-size=16` (32 OOMs at d24 + FP8, ~77.7GB/GPU)
- NCA batch size 8 if NCA is enabled (32 OOMs)

## 8. Monitor Training

### wandb
Runs log to the `nanochat` wandb project. Key metrics:
- `val_bpb` — validation bits-per-byte (lower is better)
- `core_metric` — DCLM CORE score (target: ≥ 0.256525)
- `train/tok_per_sec` — throughput
- `train/mfu` — model FLOPS utilization

### GPU monitoring
```bash
watch -n 2 nvidia-smi
```

See `dev/GPU_MONITORING.md` for more detailed monitoring tools.

## 9. Common Issues

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: triton` | `apt-get install python3-dev` then `uv sync --extra gpu` |
| OOM at batch size 32 | Use `--device-batch-size=16` |
| `torch.compile` cache fills disk | Set `TORCHINDUCTOR_CACHE_DIR=/dev/shm/torchinductor` |
| `TORCH_DISABLE_ADDR2LINE=1` | Add to suppress noisy C++ stack trace warnings |
| `torchrun --run` flag conflict | Drop the `--run` flag (it conflicts with torchrun's `--run-path`) or use `python -m torch.distributed.run` instead |
| wandb login needed | `uv run wandb login` with API key |

## 10. After the Run

Report back to the local session with:
1. Final BPB and/or CORE score
2. Comparison to baseline
3. Any wandb screenshots or notable observations
4. Wall-clock time
5. Any issues encountered
