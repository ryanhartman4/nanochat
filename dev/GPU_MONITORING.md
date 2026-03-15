# GPU Training Run Monitoring Guide

Reference for Claude instances launched on GPU nodes to monitor nanochat training runs.

## What Works (lightweight, use these)

- **`nvidia-smi`** — shows all GPUs, utilization, power, temp, VRAM, and PIDs in one call. Run every few minutes. Key signals:
  - GPU util dropping from 100% to ~30-50% = eval phase or end of training
  - GPU util at 0%, power at idle (~76W), 0 VRAM = run finished
  - Temp 60-75°C is normal under load on H100s; >83°C means throttling

- **`ps -p <PID> -o pid,stat,%cpu,%mem,etime --no-headers`** — check if the training process is alive and how long it's been running. For multi-GPU runs (`torchrun --nproc_per_node=8`), there will be 8 Python child processes; check the parent or use `ps aux | grep base_train`

- **`ls -la <wandb_run_dir>/run-*.wandb`** — check file size growth to confirm metrics are being logged. Bigger jumps (~64KB) often correlate with eval checkpoints

- **Report files** — check `~/.cache/nanochat/report/` for markdown summaries written at the end of each stage (tokenizer training, base model training). These contain final BPB, CORE metric, MFU, training time, etc.

- **Checkpoint files** — check `~/.cache/nanochat/base_checkpoints/<model>/` for saved `.pt` files to confirm training completed

## What Doesn't Work (avoid these)

- **`strace -p <PID>`** — invasive, attaches to the training process, can interfere with performance. Don't use it to try to capture stdout

- **Parsing wandb protobuf files with Python** — requires the project venv, uses CPU/memory, and the protobuf format is fragile. Not worth it for monitoring

- **Reading `/proc/<PID>/fd/1`** — stdout goes to a terminal (e.g. `/dev/pts/0`), can't be read from another process

- **wandb Python API for offline runs** — doesn't work for offline/in-progress runs

## What the User Will Tell You

The training output (step number, loss, BPB, ETA) prints to the terminal the user is watching. Rely on them to share specific numbers. Your job is to track system health (GPU util, temp, power, process alive) and provide context on what the numbers mean.

## Typical Run Timelines

### d12 on single H100

- 0-3 min: NCA pre-pre-training (500 steps)
- 3-35 min: main training (~2,079 steps), loss drops from ~10 to ~3-4, BPB evals every 250 steps
- 35-44 min: final evals (val BPB, CORE metric, model sampling, checkpoint save)

### d12 on 8xH100

- 0-1 min: NCA pre-pre-training (500 steps, faster with DDP)
- 1-5 min: main training (~2,079 steps)
- 5-10 min: final evals

### d24 on 8xH100 (leaderboard target)

- 0-1 min: NCA pre-pre-training
- 1-60 min: main training (baseline ~1.65hr, target sub-1.0hr with NCA + 24K tokenizer)
- 60-70 min: final evals

## NCA-Specific Monitoring

During the NCA pre-pre-training phase:
- NCA loss should start near `ln(vocab)` — e.g. `ln(256) = 5.55` for n=4
- Loss should decrease over 500 steps (typically drops to ~4.5)
- "NCA transfer: preserving attention weights..." confirms transfer completed
- "NCA pre-pre-training complete" confirms the phase is done
- After transfer, the first validation BPB should be lower than a cold-start baseline (~3.15 vs ~3.25-3.35 for d12)
