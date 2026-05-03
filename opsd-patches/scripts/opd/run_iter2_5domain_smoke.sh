#!/bin/bash
# Iter2 SMOKE: 5-domain × 5-teacher OPD on Qwen2.5-1.5B-Instruct.
# 20 steps, balanced 1:1:1:1:1 mixed parquet. First time multi-teacher >2 teachers.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- pre-flight: 5-teacher health check ---
echo "=== Pre-flight: 5-teacher health check ==="
PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/opsd/bin/python /home/ubuntu/MOPD/teacher_health_check.py \
    /home/ubuntu/models/qwen2.5/teacher-math-math \
    /home/ubuntu/models/qwen2.5/teacher-medical-huatuo \
    /home/ubuntu/models/qwen2.5/teacher-code-coder \
    /home/ubuntu/models/qwen2.5/teacher-tool-light \
    /home/ubuntu/models/qwen2.5/teacher-search-r1v3 \
    || { echo "🚨 teacher health check FAILED"; exit 1; }
echo "=== Pre-flight passed; cleaning stale shm + ray sessions ==="
rm -f /dev/shm/psm_* /dev/shm/mp-* 2>/dev/null
rm -rf /tmp/ray/session_2026-04-* /tmp/ray/session_2026-05-02_* /tmp/ray/session_2026-05-03_0* 2>/dev/null
echo "=== starting training ==="

export MODEL_PATH=/home/ubuntu/models/qwen2.5/student-1.5b-instruct
export MODEL_NAME=Qwen2.5-1.5B-Instruct

# Primary teacher = math-math (default fallback for unrouted rows)
export PRIMARY_TEACHER_PATH=/home/ubuntu/models/qwen2.5/teacher-math-math
export DEFAULT_TEACHER_NAME=math
export DEFAULT_REWARD_FN=math_dapo

# 5 teachers: name → (path, list of data_sources to route)
export TEACHERS_HYDRA='[{name:math,path:/home/ubuntu/models/qwen2.5/teacher-math-math,data_sources:[math_dapo,math,aime24,aime25]},{name:medical,path:/home/ubuntu/models/qwen2.5/teacher-medical-huatuo,data_sources:[medical_o1,medqa]},{name:code,path:/home/ubuntu/models/qwen2.5/teacher-code-coder,data_sources:[code_kodcode]},{name:tool,path:/home/ubuntu/models/qwen2.5/teacher-tool-light,data_sources:[tool_star]},{name:search,path:/home/ubuntu/models/qwen2.5/teacher-search-r1v3,data_sources:[search_nqhq]}]'

# Per-domain reward fn dispatch
RFN=/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards
export REWARD_FNS_HYDRA="{math_dapo:{path:${RFN}/math_reward.py,name:compute_score},math:{path:${RFN}/math_reward.py,name:compute_score},aime24:{path:${RFN}/math_reward.py,name:compute_score},aime25:{path:${RFN}/math_reward.py,name:compute_score},medqa:{path:${RFN}/medical_reward.py,name:compute_score},medical_o1:{path:${RFN}/medical_o1_reward.py,name:compute_score},code_kodcode:{path:${RFN}/code_reward.py,name:compute_score},tool_star:{path:${RFN}/tool_reward.py,name:compute_score},search_nqhq:{path:${RFN}/search_reward.py,name:compute_score}}"

export TRAIN_FILE="${REPO_ROOT}/data/5domain_mixed/train.parquet"
export VAL_FILES="['${REPO_ROOT}/data/5domain_mixed/val_math500.parquet','${REPO_ROOT}/data/5domain_mixed/val_aime24.parquet','${REPO_ROOT}/data/5domain_mixed/val_aime25.parquet','${REPO_ROOT}/data/5domain_mixed/val_medqa.parquet']"

# Same hyperparams as Iter1 fresh smoke for direct comparison
export TRAIN_BATCH_SIZE=256
export PPO_MINI_BATCH_SIZE=64
export PPO_MICRO_BATCH_SIZE_PER_GPU=2
export TOTAL_EPOCHS=1
export TOTAL_TRAINING_STEPS=20
export LEARNING_RATE=1e-6
export TEST_FREQ=10
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
export RUN_ID=Stage2-Iter2-5domain-smoke20
export GPUS_PER_NODE=8

bash "${SCRIPT_DIR}/train_opd_mt.sh"
