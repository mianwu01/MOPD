#!/bin/bash
# Stage 0 smoke test wrapper for train_opd.sh.
# Tiny data slice, short training, vllm rollout. Goal: end-to-end pipeline check.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/ubuntu/models/Qwen3-1.7B-Base}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/home/ubuntu/models/Qwen3-4B}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B-Base}"

# Use pre-sliced smoke data
export OUTPUT_DIR="${REPO_ROOT}/data/grpo_processed_smoke"
export SKIP_PREPARE=1

# Short training
export TOTAL_EPOCHS=2          # 2 train steps with 256 rows / bs 256
export TRAIN_BATCH_SIZE=256
export PPO_MINI_BATCH_SIZE=64
export PPO_MICRO_BATCH_SIZE_PER_GPU=2
export TEST_FREQ=2             # eval after step 2
export SAVE_FREQ=999           # don't save mid-test
export VAL_BEFORE_TRAIN=False
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=2048   # bound rollout time
export ROLLOUT_N=1
export GPU_MEMORY_UTIL=0.6
export TP_SIZE=1
export ROLLOUT_NAME=vllm       # sglang not installed
export RUN_ID=smoke

bash "${SCRIPT_DIR}/train_opd.sh"
