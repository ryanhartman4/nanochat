#!/bin/bash
# NCA attn-only transfer diagnostic — re-runs 2K and 4K configs with MLP reinit.
#
# Compares against existing full-transfer results:
#   2K×50ep full: 0.8553   4K×50ep full: 0.8564   Baseline: 0.8538
#
# Uses pre-generated NCA data from the original sweep.
#
# Usage: bash runs/nca_sweep_attn_only.sh

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
SWEEP_DIR="$NANOCHAT_BASE_DIR/sweep_results"
mkdir -p "$SWEEP_DIR"

COMMON_ARGS="--depth=12 --eval-every=100 --core-metric-every=10000 --sample-every=-1 --save-every=-1 --run=dummy"

# --- Run 1/2: nca-2k-50ep attn-only ---
echo ""
echo "=========================================="
echo "=== [1/2] nca-2k-50ep (attn-only transfer) ==="
echo "=========================================="
rm -f "$NANOCHAT_BASE_DIR/bpb_log.csv"

uv run python -m scripts.base_train $COMMON_ARGS \
    --nca-data="$NANOCHAT_BASE_DIR/sweep_nca-2k-50ep" \
    --nca-lr=1e-4 \
    --nca-alphabet-size=10 \
    --nca-batch-size=32 \
    --nca-transfer-mode=attn-only

cp "$NANOCHAT_BASE_DIR/bpb_log.csv" "$SWEEP_DIR/bpb_nca-2k-50ep-attn-only.csv"
echo "=== Saved: bpb_nca-2k-50ep-attn-only.csv ==="

# --- Run 2/2: nca-4k-50ep attn-only ---
echo ""
echo "=========================================="
echo "=== [2/2] nca-4k-50ep (attn-only transfer) ==="
echo "=========================================="
rm -f "$NANOCHAT_BASE_DIR/bpb_log.csv"

uv run python -m scripts.base_train $COMMON_ARGS \
    --nca-data="$NANOCHAT_BASE_DIR/sweep_nca-4k-50ep" \
    --nca-lr=1e-4 \
    --nca-alphabet-size=10 \
    --nca-batch-size=32 \
    --nca-transfer-mode=attn-only

cp "$NANOCHAT_BASE_DIR/bpb_log.csv" "$SWEEP_DIR/bpb_nca-4k-50ep-attn-only.csv"
echo "=== Saved: bpb_nca-4k-50ep-attn-only.csv ==="

# --- Summary ---
echo ""
echo "=========================================="
echo "=== Attn-only diagnostic complete ==="
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
