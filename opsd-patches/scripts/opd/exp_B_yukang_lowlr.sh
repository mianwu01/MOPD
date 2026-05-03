#!/bin/bash
# Control B: single-teacher math OPD (Yukang only), same other config as Iter1.
# Goal: tell whether Iter1 collapse is caused by multi-teacher imbalance or by
# vanilla OPD on math-yukang alone (which Control A revealed is far KL=37 from student).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MODEL_PATH=/home/ubuntu/models/qwen2.5/student-1.5b-instruct
export MODEL_NAME=Qwen2.5-1.5B-Instruct
export PRIMARY_TEACHER_PATH=/home/ubuntu/models/qwen2.5/teacher-math-yukang
# Single teacher list — multi-teacher routing path still active but only one bucket.
export TEACHERS_HYDRA='[{name:math,path:/home/ubuntu/models/qwen2.5/teacher-math-yukang,data_sources:[math_dapo,math,aime24,aime25]}]'
export REWARD_FNS_HYDRA='{math_dapo:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},math:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},aime24:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},aime25:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score}}'
export DEFAULT_REWARD_FN=math_dapo
export DEFAULT_TEACHER_NAME=math

# Math-only train data (filtered from iter1_mixed, exact same prompts the math-half of Iter1 used).
export TRAIN_FILE="${REPO_ROOT}/data/control_singleteacher_math/train.parquet"
# Math-only val files (no medqa; comparison anchor is iter1's math500/aime24/aime25 numbers).
export VAL_FILES="['${REPO_ROOT}/data/iter1_mixed/val_math500.parquet','${REPO_ROOT}/data/iter1_mixed/val_aime24.parquet','${REPO_ROOT}/data/iter1_mixed/val_aime25.parquet']"

# IDENTICAL to Iter1 except: single teacher, no medqa data, no medqa reward, no medqa val.
export TRAIN_BATCH_SIZE=256
export PPO_MINI_BATCH_SIZE=64
export PPO_MICRO_BATCH_SIZE_PER_GPU=2
export TOTAL_EPOCHS=1
export LEARNING_RATE=1e-7
export TEST_FREQ=20      # eval at step 20 and 40 (= ~end of 1 epoch on 10178 samples)
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
export RUN_ID=ExpB-yukang-lr1e-7
export GPUS_PER_NODE=8

bash "${SCRIPT_DIR}/train_opd_mt.sh"
