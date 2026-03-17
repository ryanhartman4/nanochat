#!/bin/bash
# NCA scalar-fix sweep — tests whether preserving learnable scalars during
# transfer fixes the convergence penalty observed in prior sweeps.
#
# 5 sequential d12 experiments:
#   0. Baseline (no NCA)                         ~15 min
#   1. 2000 rules × 50 epochs  (3,100 steps)     ~25 min
#   2. 4000 rules × 50 epochs  (6,250 steps)     ~35 min
#   3. 6000 rules × 50 epochs  (9,375 steps)     ~50 min
#   4. 8000 rules × 50 epochs  (12,500 steps)    ~60 min
#
# All use attn-only transfer (best from prior sweep) + scalar preservation.
# Single GPU, nca_batch_size=32 (effective batch 32, close to paper's 16).
#
# Prior results (WITHOUT scalar fix):
#   Baseline: 0.8538, best NCA: 0.8545 (+0.0007, 4K attn-only)
#
# Usage: bash runs/nca_sweep_scalars.sh
# Total: ~3 hours
# Results: $NANOCHAT_BASE_DIR/sweep_scalars/bpb_*.csv

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
SWEEP_DIR="$NANOCHAT_BASE_DIR/sweep_scalars"
mkdir -p "$NANOCHAT_BASE_DIR" "$SWEEP_DIR"

# Activate venv
source .venv/bin/activate

# --- Prerequisites ---

python -m nanochat.dataset -n 8

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer.json" ]; then
    python -m scripts.tok_train
fi

# --- Generate NCA datasets ---
# 50 epochs across all configs (prior sweep showed more epochs hurts).

RULE_COUNTS=(2000 4000 6000 8000)

for RULES in "${RULE_COUNTS[@]}"; do
    DATA_DIR="$NANOCHAT_BASE_DIR/sweep_nca-${RULES}-50ep"
    if [ -f "$DATA_DIR/nca_data.pt" ]; then
        echo "=== NCA data for ${RULES} rules already exists, skipping ==="
    else
        echo "=== Generating NCA data: ${RULES} rules × 50 epochs ==="
        OMP_NUM_THREADS=8 python -m scripts.nca_generate \
            --num-rules $RULES --num-epochs 50 \
            --seq-len 2048 --alphabet-size 10 --device cuda \
            --output "$DATA_DIR"
    fi
done

# --- Common training args ---
COMMON_ARGS="--depth=12 --eval-every=100 --core-metric-every=10000 --sample-every=-1 --save-every=-1 --run=dummy"

# --- Run 0: Baseline ---
echo ""
echo "=========================================="
echo "=== [0/4] Baseline (no NCA) ==="
echo "=========================================="
rm -f "$NANOCHAT_BASE_DIR/bpb_log.csv"

python -m scripts.base_train $COMMON_ARGS

cp "$NANOCHAT_BASE_DIR/bpb_log.csv" "$SWEEP_DIR/bpb_baseline.csv"
echo "=== Saved: bpb_baseline.csv ==="

# --- NCA Runs ---
RUN_NUM=1
TOTAL_RUNS=${#RULE_COUNTS[@]}

for RULES in "${RULE_COUNTS[@]}"; do
    NAME="nca-${RULES}-50ep"
    echo ""
    echo "=========================================="
    echo "=== [$RUN_NUM/$TOTAL_RUNS] $NAME (attn-only + scalars) ==="
    echo "=========================================="
    rm -f "$NANOCHAT_BASE_DIR/bpb_log.csv"

    python -m scripts.base_train $COMMON_ARGS \
        --nca-data="$NANOCHAT_BASE_DIR/sweep_${NAME}" \
        --nca-lr=1e-4 \
        --nca-alphabet-size=10 \
        --nca-batch-size=32 \
        --nca-transfer-mode=attn-only

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
echo "  Prior best (without scalar fix): 4K attn-only = 0.8545 (+0.0007 vs baseline 0.8538)"
echo ""
for f in "$SWEEP_DIR"/bpb_*.csv; do
    LAST_LINE=$(tail -1 "$f")
    LAST_MIN_BPB=$(echo "$LAST_LINE" | cut -d',' -f3)
    WALL_MIN=$(echo "$LAST_LINE" | cut -d',' -f4)
    echo "  $(basename $f): min_BPB=$LAST_MIN_BPB, wall=${WALL_MIN}min"
done
echo ""
echo "Key question: does any config BEAT baseline (min_BPB < 0.8538)?"
echo "If yes: the scalar fix resolved the convergence penalty."
echo "If no at 8K: NCA may not transfer to nanochat at d12 scale — try d24."
