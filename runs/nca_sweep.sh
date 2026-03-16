#!/bin/bash
# NCA step count sweep — runs 3 sequential d12 experiments on a single GPU.
# Each run takes ~15 min, total ~45 min.
#
# Tests NCA steps 500/750/1000 at n=10, lr=1e-4 (paper defaults).
# Compare BPB curves against baseline (no NCA) to find optimal step count.
#
# Usage: bash runs/nca_sweep.sh
# Baseline already collected separately (no NCA, d12, same hardware).

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Activate venv
source .venv/bin/activate

# Generate NCA data once (reused across all runs)
# Need enough tokens for the longest run (1000 steps * batch_size * seq_len)
echo "=== Generating NCA data ==="
NCA_DATA_DIR="$NANOCHAT_BASE_DIR/nca_sweep_data"
if [ ! -d "$NCA_DATA_DIR" ]; then
    OMP_NUM_THREADS=8 python -m scripts.nca_generate \
        --num-tokens 164000000 --seq-len 2048 \
        --alphabet-size 10 --device cuda --output "$NCA_DATA_DIR"
else
    echo "NCA data already exists at $NCA_DATA_DIR, skipping generation"
fi

# Download data shards if needed (d12 needs ~8 shards minimum)
python -m nanochat.dataset -n 8

# Train tokenizer if needed
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer.json" ]; then
    python -m scripts.tok_train
fi

# Run sweep
for NCA_STEPS in 500 750 1000; do
    echo ""
    echo "=========================================="
    echo "=== NCA sweep: steps=$NCA_STEPS ==="
    echo "=========================================="

    # Clear BPB log so each run gets a clean CSV
    rm -f "$NANOCHAT_BASE_DIR/bpb_log.csv"

    python -m scripts.base_train \
        --depth=12 \
        --nca-steps=$NCA_STEPS \
        --nca-data="$NCA_DATA_DIR" \
        --nca-lr=1e-4 \
        --nca-alphabet-size=10 \
        --eval-every=100 \
        --core-metric-every=10000 \
        --sample-every=-1 \
        --save-every=-1 \
        --run=dummy

    # Save BPB log with descriptive name
    cp "$NANOCHAT_BASE_DIR/bpb_log.csv" "$NANOCHAT_BASE_DIR/bpb_nca${NCA_STEPS}.csv"
    echo "=== Saved: bpb_nca${NCA_STEPS}.csv ==="
done

echo ""
echo "=========================================="
echo "=== Sweep complete ==="
echo "=========================================="
echo "Results saved to:"
echo "  $NANOCHAT_BASE_DIR/bpb_nca500.csv"
echo "  $NANOCHAT_BASE_DIR/bpb_nca750.csv"
echo "  $NANOCHAT_BASE_DIR/bpb_nca1000.csv"
echo "Compare against baseline: $NANOCHAT_BASE_DIR/bpb_log.csv (from baseline run)"
