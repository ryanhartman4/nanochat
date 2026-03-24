#!/bin/bash
# Monitor XSA experiment progress. Designed to be read by Claude on a cron loop.
# Outputs a concise status report with BPB trajectory and baseline comparison.

RESULTS_DIR=/root/nanochat/results
RUNS=("d12-baseline" "d12-xsa-all" "d12-xsa-last4")
XSA_LABELS=("Baseline (no XSA)" "XSA all layers (start=0)" "XSA last 4 layers (start=8)")

# Prior baseline from earlier d12 experiments (for comparison)
PRIOR_BASELINE_BPB=0.854

echo "========================================"
echo "XSA Experiment Monitor — $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "========================================"
echo "Prior d12 baseline BPB (from earlier experiments): $PRIOR_BASELINE_BPB"
echo ""

# Check if the main script is still running
if pgrep -f "run_xsa_experiments.sh" > /dev/null 2>&1; then
    echo "STATUS: Experiments script is RUNNING"
elif pgrep -f "scripts.base_train" > /dev/null 2>&1; then
    echo "STATUS: A training process is RUNNING (launched outside wrapper script)"
else
    echo "STATUS: No training process detected"
fi
echo ""

# GPU status
if command -v nvidia-smi &> /dev/null; then
    echo "--- GPU Status ---"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' '{printf "GPU: %s | Util: %s%% | Mem: %s/%s MiB | Temp: %s°C\n", $1, $2, $3, $4, $5}'
    echo ""
fi

# Collect BPB trajectories per run (for the comparison table)
declare -A BPB_DATA  # key: "run:step" -> bpb value

for i in "${!RUNS[@]}"; do
    run="${RUNS[$i]}"
    label="${XSA_LABELS[$i]}"
    log="$RESULTS_DIR/${run}.log"

    echo "--- Run $((i+1)): $label ---"

    if [ ! -f "$log" ]; then
        echo "  Not started yet (no log file)"
        echo ""
        continue
    fi

    # Extract key info
    last_step_line=$(grep -E "^step " "$log" | tail -1)
    val_lines=$(grep -E "^Step [0-9]+ \| Validation bpb:" "$log")
    min_bpb=$(grep "Minimum validation bpb:" "$log" | tail -1)
    peak_mem=$(grep "Peak memory usage:" "$log" | tail -1)
    wall_time=$(grep "Main training time:" "$log" | tail -1)
    tok_sec=$(grep -E "^step " "$log" | tail -1 | grep -oP 'tok/sec: [\d,]+' | sed 's/tok\/sec: //')
    total_iters=$(echo "$last_step_line" | grep -oP '/\d+' | head -1 | tr -d '/')

    # Store BPB values for comparison table
    if [ -n "$val_lines" ]; then
        while read -r line; do
            step=$(echo "$line" | grep -oP 'Step \d+' | grep -oP '\d+')
            bpb=$(echo "$line" | grep -oP 'bpb: [\d.]+' | grep -oP '[\d.]+')
            BPB_DATA["${run}:${step}"]="$bpb"
        done <<< "$val_lines"
    fi

    if [ -n "$min_bpb" ]; then
        # Run is complete
        final_bpb=$(echo "$min_bpb" | grep -oP '[\d.]+$')
        delta=$(awk "BEGIN {printf \"%.6f\", $final_bpb - $PRIOR_BASELINE_BPB}")
        echo "  COMPLETE"
        echo "  $min_bpb  (delta vs prior baseline: $delta)"
        echo "  $wall_time"
        [ -n "$peak_mem" ] && echo "  $peak_mem"
        [ -n "$tok_sec" ] && echo "  Last tok/sec: $tok_sec"
    elif [ -n "$last_step_line" ]; then
        # Run is in progress
        pct=$(echo "$last_step_line" | grep -oP '\([0-9.]+%\)')
        eta=$(echo "$last_step_line" | grep -oP 'eta: \S+')
        loss=$(echo "$last_step_line" | grep -oP 'loss: [\d.]+')
        cur_step=$(echo "$last_step_line" | grep -oP '^step \d+' | grep -oP '\d+')
        echo "  IN PROGRESS — step $cur_step/$total_iters $pct"
        [ -n "$loss" ] && echo "  Current $loss"
        [ -n "$tok_sec" ] && echo "  tok/sec: $tok_sec"
        [ -n "$eta" ] && echo "  $eta"
        # Show latest BPB
        if [ -n "$val_lines" ]; then
            latest_val=$(echo "$val_lines" | tail -1)
            latest_bpb=$(echo "$latest_val" | grep -oP 'bpb: [\d.]+' | grep -oP '[\d.]+')
            latest_step=$(echo "$latest_val" | grep -oP 'Step \d+' | grep -oP '\d+')
            delta=$(awk "BEGIN {printf \"%.6f\", $latest_bpb - $PRIOR_BASELINE_BPB}")
            echo "  Latest BPB at step $latest_step: $latest_bpb (delta vs prior baseline: $delta)"
        fi
    else
        echo "  STARTING (no training steps logged yet)"
    fi
    echo ""
done

# ============================================================
# BPB comparison table across all runs at every 100 steps
# ============================================================

# Collect all steps that have BPB data from any run
ALL_STEPS=()
for key in "${!BPB_DATA[@]}"; do
    step="${key#*:}"
    ALL_STEPS+=("$step")
done

if [ ${#ALL_STEPS[@]} -gt 0 ]; then
    # Sort and deduplicate steps
    SORTED_STEPS=($(printf '%s\n' "${ALL_STEPS[@]}" | sort -n | uniq))

    echo "========================================"
    echo "BPB TRAJECTORY COMPARISON (every 100 steps)"
    echo "Prior d12 baseline: $PRIOR_BASELINE_BPB"
    echo "========================================"
    printf "%-8s | %-12s %-10s | %-12s %-10s | %-12s %-10s\n" \
        "Step" "Baseline" "Δ prior" "XSA-all" "Δ prior" "XSA-last4" "Δ prior"
    printf "%-8s-+-%-12s-%-10s-+-%-12s-%-10s-+-%-12s-%-10s\n" \
        "--------" "------------" "----------" "------------" "----------" "------------" "----------"

    for step in "${SORTED_STEPS[@]}"; do
        base_bpb="${BPB_DATA[d12-baseline:${step}]:-}"
        xsa_all_bpb="${BPB_DATA[d12-xsa-all:${step}]:-}"
        xsa_last4_bpb="${BPB_DATA[d12-xsa-last4:${step}]:-}"

        # Compute deltas vs prior baseline
        if [ -n "$base_bpb" ]; then
            base_delta=$(awk "BEGIN {printf \"%+.6f\", $base_bpb - $PRIOR_BASELINE_BPB}")
        else
            base_delta="-"
            base_bpb="-"
        fi
        if [ -n "$xsa_all_bpb" ]; then
            xsa_all_delta=$(awk "BEGIN {printf \"%+.6f\", $xsa_all_bpb - $PRIOR_BASELINE_BPB}")
        else
            xsa_all_delta="-"
            xsa_all_bpb="-"
        fi
        if [ -n "$xsa_last4_bpb" ]; then
            xsa_last4_delta=$(awk "BEGIN {printf \"%+.6f\", $xsa_last4_bpb - $PRIOR_BASELINE_BPB}")
        else
            xsa_last4_delta="-"
            xsa_last4_bpb="-"
        fi

        printf "%-8s | %-12s %-10s | %-12s %-10s | %-12s %-10s\n" \
            "$step" "$base_bpb" "$base_delta" "$xsa_all_bpb" "$xsa_all_delta" "$xsa_last4_bpb" "$xsa_last4_delta"
    done
    echo ""

    # Also show delta between current runs (XSA vs this run's baseline)
    # if baseline run has data
    baseline_final="${BPB_DATA[d12-baseline:${SORTED_STEPS[-1]}]:-}"
    if [ -n "$baseline_final" ]; then
        echo "--- Deltas vs THIS run's baseline (at latest shared step) ---"
        for i in 1 2; do
            run="${RUNS[$i]}"
            label="${XSA_LABELS[$i]}"
            # Find the latest step that both baseline and this run share
            for ((j=${#SORTED_STEPS[@]}-1; j>=0; j--)); do
                s="${SORTED_STEPS[$j]}"
                b="${BPB_DATA[d12-baseline:${s}]:-}"
                x="${BPB_DATA[${run}:${s}]:-}"
                if [ -n "$b" ] && [ -n "$x" ]; then
                    delta=$(awk "BEGIN {printf \"%+.6f\", $x - $b}")
                    echo "  $label at step $s: $x vs baseline $b => delta: $delta"
                    break
                fi
            done
        done
        echo ""
    fi
fi

# Final summary
complete_count=0
for run in "${RUNS[@]}"; do
    grep -q "Minimum validation bpb:" "$RESULTS_DIR/${run}.log" 2>/dev/null && ((complete_count++)) || true
done

if [ "$complete_count" -eq 3 ]; then
    echo "========================================"
    echo "*** ALL 3 RUNS COMPLETE ***"
    echo "Ready to write results/2026-03-23-d12-xsa.md and commit."
    echo "========================================"
fi
