#!/bin/bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/torchinductor
export OMP_NUM_THREADS=4
RESULTS_DIR=/root/nanochat/results
mkdir -p "$RESULTS_DIR"

COMMON_ARGS="--depth=12 --eval-every=100 --core-metric-every=999999 --sample-every=-1 --save-every=-1"

echo "=== Run 1: Baseline (no XSA) ==="
uv run python -m scripts.base_train \
    $COMMON_ARGS \
    --run="dummy" --model-tag="d12-baseline" \
    2>&1 | tee "$RESULTS_DIR/d12-baseline.log"

echo ""
echo "=== Run 2: XSA all layers ==="
uv run python -m scripts.base_train \
    $COMMON_ARGS \
    --run="dummy" --model-tag="d12-xsa-all" \
    --xsa-start-layer=0 \
    2>&1 | tee "$RESULTS_DIR/d12-xsa-all.log"

echo ""
echo "=== Run 3: XSA last 4 layers ==="
uv run python -m scripts.base_train \
    $COMMON_ARGS \
    --run="dummy" --model-tag="d12-xsa-last4" \
    --xsa-start-layer=8 \
    2>&1 | tee "$RESULTS_DIR/d12-xsa-last4.log"

echo ""
echo "=== All runs complete. Logs in $RESULTS_DIR ==="
