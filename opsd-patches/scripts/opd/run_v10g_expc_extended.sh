#!/bin/bash
# v10g: extended ExpC — same teacher (FutureMa-merged), same recipe (reverse_kl,
# lr=1e-6, bs=256, ROLLOUT_N=1), only training duration extended:
# 1 epoch (39 steps) -> ~4 epochs (TOTAL_TRAINING_STEPS=160).
# Goal: does ExpC's monotonic rise continue past step 39, or hit a real ceiling?
# Eval every 20 steps; save every 40.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export WANDB_PROJECT=OPSD-multi-teacher-baseline
export WANDB_ENTITY=meanwork
export WANDB_RUN_GROUP=v10g-expc-extended
export WANDB_TAGS=opd,single-teacher,futurema,extended-160-steps,ceiling-test

export MODEL_PATH=/home/ubuntu/models/qwen2.5/student-1.5b-instruct
export MODEL_NAME=Qwen2.5-1.5B-Instruct
export PRIMARY_TEACHER_PATH=/home/ubuntu/models/qwen2.5/teacher-math-futurema-merged
export TEACHERS_HYDRA='[{name:math,path:/home/ubuntu/models/qwen2.5/teacher-math-futurema-merged,data_sources:[math_dapo,math,aime24,aime25]}]'
export REWARD_FNS_HYDRA='{math_dapo:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},math:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},aime24:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},aime25:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score}}'
export DEFAULT_REWARD_FN=math_dapo
export DEFAULT_TEACHER_NAME=math

export TRAIN_FILE="${REPO_ROOT}/data/control_singleteacher_math/train.parquet"
export VAL_FILES="['${REPO_ROOT}/data/iter1_mixed/val_math500.parquet','${REPO_ROOT}/data/iter1_mixed/val_aime24.parquet','${REPO_ROOT}/data/iter1_mixed/val_aime25.parquet']"

export TRAIN_BATCH_SIZE=256
export PPO_MINI_BATCH_SIZE=64
export PPO_MICRO_BATCH_SIZE_PER_GPU=2
export TOTAL_EPOCHS=10                  # bound by TOTAL_TRAINING_STEPS
export TOTAL_TRAINING_STEPS=160         # ← was 39 in ExpC
export LEARNING_RATE=1e-6
export TEST_FREQ=20                     # eval at 20, 40, 60, 80, 100, 120, 140, 160
export SAVE_FREQ=20                     # save every val (preemption-resilient; 8× 19GB ≈ 152GB)
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
export RUN_ID=v10g-ExpC-extended-160
export GPUS_PER_NODE=8

bash "${SCRIPT_DIR}/train_opd_mt.sh"
