#!/bin/bash
# NCA redesign sweep — tests delimiter tokens + ICL masking on single H200.
#
# 4 sequential d12 experiments (~2 hours total):
#   0. Baseline (no NCA)                         ~15 min
#   1. 2000 rules × 50 epochs  (3,100 steps)     ~25 min
#   2. 2000 rules × 100 epochs (6,200 steps)     ~35 min
#   3. 4000 rules × 50 epochs  (6,250 steps)     ~35 min
#
# Configs 2 vs 3 isolate rules-vs-epochs at the same step count.
# Single GPU gives effective batch 32 (close to paper's 16).
#
# Usage: bash runs/nca_sweep.sh
# Results: $NANOCHAT_BASE_DIR/sweep_results/bpb_*.csv

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
SWEEP_DIR="$NANOCHAT_BASE_DIR/sweep_results"
mkdir -p "$NANOCHAT_BASE_DIR" "$SWEEP_DIR"

# Activate venv
source .venv/bin/activate

# --- Prerequisites (run once) ---

# Download data shards (d12 needs ~8 minimum)
python -m nanochat.dataset -n 8

# Train tokenizer if needed
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer.json" ]; then
    python -m scripts.tok_train
fi

# --- Generate NCA datasets for each config ---
# Each config needs its own dataset (different num_rules).
# Generation is fast on GPU (<1 min each).

declare -A NCA_CONFIGS
NCA_CONFIGS=(
    ["nca-2k-50ep"]="--num-rules 2000 --num-epochs 50"
    ["nca-2k-100ep"]="--num-rules 2000 --num-epochs 100"
    ["nca-4k-50ep"]="--num-rules 4000 --num-epochs 50"
)

for NAME in "${!NCA_CONFIGS[@]}"; do
    DATA_DIR="$NANOCHAT_BASE_DIR/sweep_${NAME}"
    if [ -f "$DATA_DIR/nca_data.pt" ]; then
        echo "=== NCA data for $NAME already exists, skipping ==="
    else
        echo "=== Generating NCA data: $NAME ==="
        OMP_NUM_THREADS=8 python -m scripts.nca_generate \
            ${NCA_CONFIGS[$NAME]} \
            --seq-len 2048 --alphabet-size 10 --device cuda \
            --output "$DATA_DIR"
    fi
done

# --- Common training args ---
# d12 on single GPU, eval every 100 steps for fine-grained BPB curves
COMMON_ARGS="--depth=12 --eval-every=100 --core-metric-every=10000 --sample-every=-1 --save-every=-1 --run=dummy"

# --- Run 0: Baseline (no NCA) ---
echo ""
echo "=========================================="
echo "=== [0/3] Baseline (no NCA) ==="
echo "=========================================="
rm -f "$NANOCHAT_BASE_DIR/bpb_log.csv"

python -m scripts.base_train $COMMON_ARGS

cp "$NANOCHAT_BASE_DIR/bpb_log.csv" "$SWEEP_DIR/bpb_baseline.csv"
echo "=== Saved: bpb_baseline.csv ==="

# --- NCA Runs ---
RUN_NUM=1
TOTAL_RUNS=3
for NAME in nca-2k-50ep nca-2k-100ep nca-4k-50ep; do
    echo ""
    echo "=========================================="
    echo "=== [$RUN_NUM/$TOTAL_RUNS] $NAME ==="
    echo "=========================================="
    rm -f "$NANOCHAT_BASE_DIR/bpb_log.csv"

    python -m scripts.base_train $COMMON_ARGS \
        --nca-data="$NANOCHAT_BASE_DIR/sweep_${NAME}" \
        --nca-lr=1e-4 \
        --nca-alphabet-size=10 \
        --nca-batch-size=32

    cp "$NANOCHAT_BASE_DIR/bpb_log.csv" "$SWEEP_DIR/bpb_${NAME}.csv"
    echo "=== Saved: bpb_${NAME}.csv ==="
    RUN_NUM=$((RUN_NUM + 1))
done

# --- Summary ---
echo ""
echo "=========================================="
echo "=== Sweep complete ==="
echo "=========================================="
echo "Results in $SWEEP_DIR:"
echo "  CSV format: step,bpb,min_bpb,wall_time_min"
echo ""
for f in "$SWEEP_DIR"/bpb_*.csv; do
    LAST_LINE=$(tail -1 "$f")
    LAST_BPB=$(echo "$LAST_LINE" | cut -d',' -f2)
    LAST_MIN_BPB=$(echo "$LAST_LINE" | cut -d',' -f3)
    WALL_MIN=$(echo "$LAST_LINE" | cut -d',' -f4)
    echo "  $(basename $f): min_BPB=$LAST_MIN_BPB, wall=${WALL_MIN}min"
done
echo ""
echo "Lower BPB = better. Look for the NCA config whose BPB curve"
echo "reaches baseline's final BPB fastest (convergence speedup)."
