#!/bin/bash
ps aux | grep -E 'base_train|nca_generate' | grep -v grep
echo "---GPU---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo "---RESULTS---"
for f in ~/.cache/nanochat/sweep_scalars/bpb_*.csv; do
    test -f "$f" && echo "$(basename $f): $(tail -1 $f)"
done
echo "---CURRENT---"
tail -3 ~/.cache/nanochat/bpb_log.csv 2>/dev/null
echo "done"
