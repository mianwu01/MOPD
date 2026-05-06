#!/bin/bash
# v10f: base Qwen2.5-7B-Instruct as math teacher.
# Strict clone of v10b (reverse_kl + lr=1e-6 + 1 epoch); only teacher path differs.
# Hypothesis: base 7B teacher (0.7185 on MATH-500) ≈ FutureMa-merged (0.7200) →
# distillation lift should also be ~+3pt, confirming "lift comes from distribution
# similarity to student", not from teacher-specific math gain.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export WANDB_PROJECT=OPSD-multi-teacher-baseline
export WANDB_ENTITY=meanwork
export WANDB_RUN_GROUP=v10f-base7b-teacher
export WANDB_TAGS=opd,single-teacher,base-7b-instruct,distribution-similarity-test

export MODEL_PATH=/home/ubuntu/models/qwen2.5/student-1.5b-instruct
export MODEL_NAME=Qwen2.5-1.5B-Instruct
export PRIMARY_TEACHER_PATH=/home/ubuntu/models/qwen2.5/vanilla-7b-instruct
export TEACHERS_HYDRA='[{name:math,path:/home/ubuntu/models/qwen2.5/vanilla-7b-instruct,data_sources:[math_dapo,math,aime24,aime25]}]'
export REWARD_FNS_HYDRA='{math_dapo:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},math:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},aime24:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},aime25:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score}}'
export DEFAULT_REWARD_FN=math_dapo
export DEFAULT_TEACHER_NAME=math

export TRAIN_FILE="${REPO_ROOT}/data/control_singleteacher_math/train.parquet"
export VAL_FILES="['${REPO_ROOT}/data/iter1_mixed/val_math500.parquet','${REPO_ROOT}/data/iter1_mixed/val_aime24.parquet','${REPO_ROOT}/data/iter1_mixed/val_aime25.parquet']"

export TRAIN_BATCH_SIZE=256
export PPO_MINI_BATCH_SIZE=64
export PPO_MICRO_BATCH_SIZE_PER_GPU=2
export TOTAL_EPOCHS=1
export LEARNING_RATE=1e-6
export TEST_FREQ=20
export SAVE_FREQ=20
export VAL_BEFORE_TRAIN=False
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=4096
export ROLLOUT_N=1
export VAL_N=4
export GPU_MEMORY_UTIL=0.5
export TP_SIZE=1
export ROLLOUT_NAME=vllm
export OPD_LOSS_TYPE=reverse_kl
export OPD_CHUNK_SIZE=256
export RUN_ID=v10f-base7B-teacher
export GPUS_PER_NODE=8

bash "${SCRIPT_DIR}/train_opd_mt.sh"
