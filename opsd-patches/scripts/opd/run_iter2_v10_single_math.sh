#!/bin/bash
# Iter2 v10: SINGLE-TEACHER math distillation alignment test.
# Goal: see if 1.5B student can approach Qwen2.5-Math-7B-Instruct on MATH-500
# under pure math-only distillation. If even this fails, multi-teacher framework
# stands on shaky ground.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=== Pre-flight: math teacher health check ==="
PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/opsd/bin/python /home/ubuntu/MOPD/teacher_health_check.py \
    /home/ubuntu/models/qwen2.5/teacher-math-math \
    || { echo "🚨 teacher health check FAILED"; exit 1; }
rm -f /dev/shm/psm_* /dev/shm/mp-* 2>/dev/null
rm -rf /tmp/ray/session_2026-05-04_0* 2>/dev/null
echo "=== pre-flight passed ==="

# wandb
export WANDB_PROJECT=OPSD-multi-teacher-baseline
export WANDB_ENTITY=meanwork
export WANDB_RUN_GROUP=v10-single-teacher-baseline
export WANDB_TAGS=opd,single-teacher,math-only,alignment-test

export MODEL_PATH=/home/ubuntu/models/qwen2.5/student-1.5b-instruct
export MODEL_NAME=Qwen2.5-1.5B-Instruct

# SINGLE teacher
export PRIMARY_TEACHER_PATH=/home/ubuntu/models/qwen2.5/teacher-math-math
export DEFAULT_TEACHER_NAME=math
export DEFAULT_REWARD_FN=math_dapo

# Single-teacher routing — only math; same TEACHERS_HYDRA format but 1 entry
export TEACHERS_HYDRA='[{name:math,path:/home/ubuntu/models/qwen2.5/teacher-math-math,data_sources:[math_dapo,math,aime24,aime25]}]'

RFN=/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards
export REWARD_FNS_HYDRA="{math_dapo:{path:${RFN}/math_reward.py,name:compute_score},math:{path:${RFN}/math_reward.py,name:compute_score},aime24:{path:${RFN}/math_reward.py,name:compute_score},aime25:{path:${RFN}/math_reward.py,name:compute_score}}"

# math-only training data
export TRAIN_FILE="${REPO_ROOT}/data/5domain_mixed/train_math_only.parquet"
# only math val sets
export VAL_FILES="['${REPO_ROOT}/data/5domain_mixed/val_math500.parquet','${REPO_ROOT}/data/5domain_mixed/val_aime24.parquet','${REPO_ROOT}/data/5domain_mixed/val_aime25.parquet']"

# Stratified sampler not needed (only 1 source). Use random shuffling.
# Larger bs since only 1 teacher (no FSDP swap overhead)
export TRAIN_BATCH_SIZE=256
export PPO_MINI_BATCH_SIZE=64
export PPO_MICRO_BATCH_SIZE_PER_GPU=2
export TOTAL_EPOCHS=4               # 4K rows × 4 = 16K, allow more passes
export TOTAL_TRAINING_STEPS=80      # but cap at 80 for direct compare
export LEARNING_RATE=1e-6           # match v5b for clean comparison
export TEST_FREQ=20
export SAVE_FREQ=20
export VAL_BEFORE_TRAIN=True
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=4096
export ROLLOUT_N=1
export VAL_N=4
export GPU_MEMORY_UTIL=0.5
export TP_SIZE=1
export ROLLOUT_NAME=vllm
export OPD_LOSS_TYPE=reverse_kl
export OPD_CHUNK_SIZE=256
export RUN_ID=Stage2-Iter2-v10-single-teacher-math-lr1e-6
export GPUS_PER_NODE=8

bash "${SCRIPT_DIR}/train_opd_mt.sh"
