#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~3 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 150 shards are needed for GPT-2 capability pretraining, add 20 for padding.
# The maximum total number of shards available in the entire dataset is 6542.
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

# Kick off NCA generation in background (runs parallel with tokenizer training)
# Use --device cuda while GPUs are idle during tokenizer training
# Override OMP_NUM_THREADS for this command only (global =1 cripples CPU conv2d)
# Alphabet size 10 matches the paper's default (Han et al. 2026). We initially tested n=2 and n=4
# but n=10 is what produced the paper's headline results (1.6x speedup, 6% improvement).
OMP_NUM_THREADS=8 python -m scripts.nca_generate --num-tokens 164000000 --seq-len 2048 \
    --alphabet-size 10 --device cuda --output $NANOCHAT_BASE_DIR/nca_data &
NCA_GEN_PID=$!

# train the tokenizer with default 32K vocab
# NOTE: 24K was tested (2026-03-16 d24 run) but reverted. At D-24 scale, the marginal
# gradient flow improvement (D/V: 0.047 → 0.063) did not compensate for reduced content
# per sequence — each 2048-token window covers less text with fewer BPE merges, hurting
# CORE tasks that need broad knowledge coverage. See results/2026-03-16-d24-8xh100-nca-24k.md
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for NCA generation and dataset download to complete..."
wait $NCA_GEN_PID
wait $DATASET_DOWNLOAD_PID

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
# NOTE: --nca-batch-size=8 required at D-24 (default 32 OOMs at ~77.7GB/GPU)
# NOTE: --device-batch-size=16 is the max that fits D-24 + FP8 (32 OOMs, needs 6GB more than available)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --nca-steps=500 --nca-data=$NANOCHAT_BASE_DIR/nca_data --nca-lr=1e-4 --nca-alphabet-size=10 --nca-batch-size=8 --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
