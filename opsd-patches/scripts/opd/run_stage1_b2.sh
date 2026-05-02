#!/bin/bash
# Stage 1 B2: single-teacher OPD on Qwen3-1.7B-Base student + Qwen3-4B teacher,
# DAPO-Math. This is the direct anchor for Stage 2 multi-teacher OPD.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/ubuntu/models/Qwen3-1.7B-Base}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/home/ubuntu/models/Qwen3-4B}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B-Base}"

# Reuse the full grpo_processed data prep (already done in Stage 0)
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/grpo_processed}"
export SKIP_PREPARE="${SKIP_PREPARE:-1}"

export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"   # 3 epoch ≈ 3-4 h; override to 15 for full
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export ROLLOUT_N="${ROLLOUT_N:-1}"
export GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.6}"
export TP_SIZE="${TP_SIZE:-1}"
export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
export TEST_FREQ="${TEST_FREQ:-20}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export RUN_ID="${RUN_ID:-Stage1-B2}"

bash "${SCRIPT_DIR}/train_opd.sh"
