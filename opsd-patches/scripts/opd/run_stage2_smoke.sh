#!/bin/bash
# Stage 2 multi-teacher SMOKE: verify pipeline alive, NOT a real training run.
# - 64 train rows (32 math + 32 medqa), bs=64, ubs=2 → ~1 step
# - val_before_train=False, test_freq=999 (skip eval)
# - max_response=2048 (fast)
# Goal: vocab assert passes, both teachers build, bucket counts logged, no OOM.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Build smoke train slice if missing
SMOKE_DIR="${REPO_ROOT}/data/iter1_mixed_smoke"
mkdir -p "$SMOKE_DIR"
if [ ! -f "$SMOKE_DIR/train.parquet" ]; then
    PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/opsd/bin/python <<EOF
import pandas as pd
df = pd.read_parquet("${REPO_ROOT}/data/iter1_mixed/train.parquet")
math = df[df.data_source == 'math_dapo'].head(32)
med = df[df.data_source == 'medqa'].head(32)
out = pd.concat([math, med], ignore_index=True).sample(frac=1, random_state=0).reset_index(drop=True)
out.to_parquet("${SMOKE_DIR}/train.parquet")
print(f"smoke train: {len(out)} rows ({(out.data_source=='math_dapo').sum()} math + {(out.data_source=='medqa').sum()} medqa)")
EOF
    cp "${REPO_ROOT}/data/iter1_mixed/val_math500.parquet" "${SMOKE_DIR}/"
    cp "${REPO_ROOT}/data/iter1_mixed/val_medqa.parquet" "${SMOKE_DIR}/"
fi

export MODEL_PATH=/home/ubuntu/models/qwen2.5/student-1.5b-instruct
export MODEL_NAME=Qwen2.5-1.5B-Instruct
export PRIMARY_TEACHER_PATH=/home/ubuntu/models/qwen2.5/teacher-math-yukang
export TEACHERS_HYDRA='[{name:math,path:/home/ubuntu/models/qwen2.5/teacher-math-yukang,data_sources:[math_dapo,math,aime24,aime25]},{name:medical,path:/home/ubuntu/models/qwen2.5/teacher-medical-umls,data_sources:[medqa]}]'
export REWARD_FNS_HYDRA='{math_dapo:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},math:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},aime24:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},aime25:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},medqa:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/medical_reward.py,name:compute_score}}'
export DEFAULT_REWARD_FN=math_dapo

export TRAIN_FILE="${SMOKE_DIR}/train.parquet"
export VAL_FILES="['${SMOKE_DIR}/val_math500.parquet','${SMOKE_DIR}/val_medqa.parquet']"

export TRAIN_BATCH_SIZE=64
export PPO_MINI_BATCH_SIZE=16
export PPO_MICRO_BATCH_SIZE_PER_GPU=2
export TOTAL_EPOCHS=1
export TOTAL_TRAINING_STEPS=1
export TEST_FREQ=999
export SAVE_FREQ=999
export VAL_BEFORE_TRAIN=False
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=2048
export ROLLOUT_N=1
export VAL_N=1
export GPU_MEMORY_UTIL=0.5
export TP_SIZE=1
export ROLLOUT_NAME=vllm
export RUN_ID=Stage2-Smoke
export GPUS_PER_NODE=8

bash "${SCRIPT_DIR}/train_opd_mt.sh"
