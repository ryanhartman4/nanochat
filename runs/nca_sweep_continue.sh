#!/bin/bash
# NCA sweep continuation — runs the 2 remaining configs that didn't complete.
#   2/3: nca-2k-100ep (2000 rules × 100 epochs, ~6,200 steps)  ~35 min
#   3/3: nca-4k-50ep  (4000 rules × 50 epochs,  ~6,250 steps)  ~35 min
#
# NCA data already generated. Existing results (baseline, nca-2k-50ep) untouched.
#
# Usage: bash runs/nca_sweep_continue.sh

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
SWEEP_DIR="$NANOCHAT_BASE_DIR/sweep_results"
mkdir -p "$SWEEP_DIR"

COMMON_ARGS="--depth=12 --eval-every=100 --core-metric-every=10000 --sample-every=-1 --save-every=-1 --run=dummy"

# --- Run 2/3: nca-2k-100ep ---
echo ""
echo "=========================================="
echo "=== [2/3] nca-2k-100ep ==="
echo "=========================================="
rm -f "$NANOCHAT_BASE_DIR/bpb_log.csv"

uv run python -m scripts.base_train $COMMON_ARGS \
    --nca-data="$NANOCHAT_BASE_DIR/sweep_nca-2k-100ep" \
    --nca-lr=1e-4 \
    --nca-alphabet-size=10 \
    --nca-batch-size=32

cp "$NANOCHAT_BASE_DIR/bpb_log.csv" "$SWEEP_DIR/bpb_nca-2k-100ep.csv"
echo "=== Saved: bpb_nca-2k-100ep.csv ==="

# --- Run 3/3: nca-4k-50ep ---
echo ""
echo "=========================================="
echo "=== [3/3] nca-4k-50ep ==="
echo "=========================================="
rm -f "$NANOCHAT_BASE_DIR/bpb_log.csv"

uv run python -m scripts.base_train $COMMON_ARGS \
    --nca-data="$NANOCHAT_BASE_DIR/sweep_nca-4k-50ep" \
    --nca-lr=1e-4 \
    --nca-alphabet-size=10 \
    --nca-batch-size=32

cp "$NANOCHAT_BASE_DIR/bpb_log.csv" "$SWEEP_DIR/bpb_nca-4k-50ep.csv"
echo "=== Saved: bpb_nca-4k-50ep.csv ==="

# --- Summary ---
echo ""
echo "=========================================="
echo "=== Sweep complete ==="
echo "=========================================="
echo "Results in $SWEEP_DIR:"
echo ""
for f in "$SWEEP_DIR"/bpb_*.csv; do
    LAST_LINE=$(tail -1 "$f")
    LAST_BPB=$(echo "$LAST_LINE" | cut -d',' -f2)
    LAST_MIN_BPB=$(echo "$LAST_LINE" | cut -d',' -f3)
    WALL_MIN=$(echo "$LAST_LINE" | cut -d',' -f4)
    echo "  $(basename $f): min_BPB=$LAST_MIN_BPB, wall=${WALL_MIN}min"
done
