#!/bin/bash
# Iter1 FRESH smoke: math-math + medical-huatuo, 20 steps.
# Direct replay of old Iter1 config (which collapsed due to broken yukang)
# with verified-healthy new teachers. Should NOT collapse.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- pre-flight: teacher health check (catches yukang-style collapsed ckpts) ---
echo "=== Pre-flight: teacher health check ==="
PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/opsd/bin/python /home/ubuntu/MOPD/teacher_health_check.py \
    /home/ubuntu/models/qwen2.5/teacher-math-math \
    /home/ubuntu/models/qwen2.5/teacher-medical-huatuo \
    || { echo "🚨 teacher health check FAILED — aborting"; exit 1; }
echo "=== Pre-flight passed; starting training ==="

export MODEL_PATH=/home/ubuntu/models/qwen2.5/student-1.5b-instruct
export MODEL_NAME=Qwen2.5-1.5B-Instruct
export PRIMARY_TEACHER_PATH=/home/ubuntu/models/qwen2.5/teacher-math-math
export TEACHERS_HYDRA='[{name:math,path:/home/ubuntu/models/qwen2.5/teacher-math-math,data_sources:[math_dapo,math,aime24,aime25]},{name:medical,path:/home/ubuntu/models/qwen2.5/teacher-medical-huatuo,data_sources:[medqa]}]'
export REWARD_FNS_HYDRA='{math_dapo:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},math:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},aime24:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},aime25:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},medqa:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/medical_reward.py,name:compute_score}}'
export DEFAULT_REWARD_FN=math_dapo
export DEFAULT_TEACHER_NAME=math

export TRAIN_FILE="${REPO_ROOT}/data/iter1_mixed/train.parquet"
export VAL_FILES="['${REPO_ROOT}/data/iter1_mixed/val_math500.parquet','${REPO_ROOT}/data/iter1_mixed/val_aime24.parquet','${REPO_ROOT}/data/iter1_mixed/val_aime25.parquet','${REPO_ROOT}/data/iter1_mixed/val_medqa.parquet']"

# Same hyperparams as old Iter1 for direct comparison; only smoke length changed.
export TRAIN_BATCH_SIZE=256
export PPO_MINI_BATCH_SIZE=64
export PPO_MICRO_BATCH_SIZE_PER_GPU=2
export TOTAL_EPOCHS=1
export TOTAL_TRAINING_STEPS=20      # SMOKE: stop after 20 steps
export LEARNING_RATE=1e-6
export TEST_FREQ=10
export SAVE_FREQ=20
export VAL_BEFORE_TRAIN=True        # fresh baseline (different teacher pair)
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=4096
export ROLLOUT_N=1
export VAL_N=4
export GPU_MEMORY_UTIL=0.5
export TP_SIZE=1
export ROLLOUT_NAME=vllm
export OPD_LOSS_TYPE=reverse_kl
export OPD_CHUNK_SIZE=256
export RUN_ID=Stage2-Iter1-mathmed-NEW-smoke20
export GPUS_PER_NODE=8

bash "${SCRIPT_DIR}/train_opd_mt.sh"
