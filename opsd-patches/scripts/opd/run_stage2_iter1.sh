#!/bin/bash
# Stage 2 Iter1 formal run: Qwen2.5-1.5B-Instruct student, math (Yukang) +
# medical (UMLS-SFT) teachers, 1 epoch on iter1_mixed (~20k samples mixed 1:1).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MODEL_PATH=/home/ubuntu/models/qwen2.5/student-1.5b-instruct
export MODEL_NAME=Qwen2.5-1.5B-Instruct
export PRIMARY_TEACHER_PATH=/home/ubuntu/models/qwen2.5/teacher-math-yukang
export TEACHERS_HYDRA='[{name:math,path:/home/ubuntu/models/qwen2.5/teacher-math-yukang,data_sources:[math_dapo,math,aime24,aime25]},{name:medical,path:/home/ubuntu/models/qwen2.5/teacher-medical-umls,data_sources:[medqa]}]'
export REWARD_FNS_HYDRA='{math_dapo:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},math:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},aime24:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},aime25:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/math_reward.py,name:compute_score},medqa:{path:/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards/medical_reward.py,name:compute_score}}'
export DEFAULT_REWARD_FN=math_dapo
export DEFAULT_TEACHER_NAME=math

export TRAIN_FILE="${REPO_ROOT}/data/iter1_mixed/train.parquet"
export VAL_FILES="['${REPO_ROOT}/data/iter1_mixed/val_math500.parquet','${REPO_ROOT}/data/iter1_mixed/val_aime24.parquet','${REPO_ROOT}/data/iter1_mixed/val_aime25.parquet','${REPO_ROOT}/data/iter1_mixed/val_medqa.parquet']"

# Training config (8 H100):
# - bs=256 prompts (bigger to keep per-batch math/medqa fluctuation < ~3 std).
#   With balanced 1:1 dataset, std≈8 per teacher → trainer's per-teacher
#   truncation-to-(n_dp multiple) drops ≤ ~6% rows. At bs=128 it'd be ~25%.
# - mini_bs=64 (4 grad steps/iter), ubs=2
# - max_response=4096 (real reasoning room), val_n=4 (cheap), n_rollouts=1
# - 1 epoch on 20356 mixed = ~80 steps
# - val_before_train=False (already have baseline from prior run, save 10 min)
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
export RUN_ID=Stage2-Iter1-mathmed
export GPUS_PER_NODE=8

bash "${SCRIPT_DIR}/train_opd_mt.sh"
