#!/bin/bash
# Stage 1 B1: GRPO baseline on Qwen3-1.7B-Base + DAPO-Math.
# Pure RL, no teacher. Anchor for "is multi-teacher OPD better than RL alone?"
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/ubuntu/models/Qwen3-1.7B-Base}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B-Base}"

export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"   # 3 epoch ≈ 3-4 h on 8×A100; override to 15 for full
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.6}"
export TP_SIZE="${TP_SIZE:-1}"
export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
export TEST_FREQ="${TEST_FREQ:-20}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export RUN_ID="${RUN_ID:-Stage1-B1}"

bash "${SCRIPT_DIR}/train_grpo_native.sh"
