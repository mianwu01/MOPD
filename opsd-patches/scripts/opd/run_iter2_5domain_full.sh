#!/bin/bash
# Iter2 PAPER-GRADE: 5-domain × 5-teacher OPD with REAL reward functions.
# - code_reward: subprocess Python sandbox runs unit tests (KodCode + HumanEval+)
# - tool_reward: <answer> tag EM/F1/numeric matching (Tool-Star held-out)
# - search_reward: SQuAD-style EM/F1 over multi-target GT (7 SearchR1 benchmarks)
# - math_reward: existing boxed match
# - medical_reward: existing MCQ letter match (used on MedQA val)
#
# Train: 80 steps (vs smoke 20). Val every 20 steps. Save every 20.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- pre-flight: 5-teacher health check + clean stale resources ---
echo "=== Pre-flight: 5-teacher health check ==="
PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/opsd/bin/python /home/ubuntu/MOPD/teacher_health_check.py \
    /home/ubuntu/models/qwen2.5/teacher-math-math \
    /home/ubuntu/models/qwen2.5/teacher-medical-huatuo \
    /home/ubuntu/models/qwen2.5/teacher-code-coder \
    /home/ubuntu/models/qwen2.5/teacher-tool-light \
    /home/ubuntu/models/qwen2.5/teacher-search-r1v3 \
    || { echo "🚨 teacher health check FAILED"; exit 1; }
echo "=== cleaning stale shm + ray sessions ==="
rm -f /dev/shm/psm_* /dev/shm/mp-* 2>/dev/null
rm -rf /tmp/ray/session_2026-04-* /tmp/ray/session_2026-05-02_* /tmp/ray/session_2026-05-03_0* 2>/dev/null
echo "=== pre-flight passed ==="

# wandb config — meanwork team, OPSD project
export WANDB_PROJECT=OPSD-multi-teacher-baseline
export WANDB_ENTITY=meanwork
export WANDB_RUN_GROUP=iter2-5domain
export WANDB_TAGS=opd-mt,5-teacher,iter2,paper-grade

export MODEL_PATH=/home/ubuntu/models/qwen2.5/student-1.5b-instruct
export MODEL_NAME=Qwen2.5-1.5B-Instruct

# Primary teacher = math-math (default fallback for unrouted rows)
export PRIMARY_TEACHER_PATH=/home/ubuntu/models/qwen2.5/teacher-math-math
export DEFAULT_TEACHER_NAME=math
export DEFAULT_REWARD_FN=math_dapo

# 5 teachers — also map val data_sources (code_humaneval, search_nqhq, tool_star) to right teacher
export TEACHERS_HYDRA='[{name:math,path:/home/ubuntu/models/qwen2.5/teacher-math-math,data_sources:[math_dapo,math,aime24,aime25]},{name:medical,path:/home/ubuntu/models/qwen2.5/teacher-medical-huatuo,data_sources:[medical_o1,medqa]},{name:code,path:/home/ubuntu/models/qwen2.5/teacher-code-coder,data_sources:[code_kodcode,code_humaneval]},{name:tool,path:/home/ubuntu/models/qwen2.5/teacher-tool-light,data_sources:[tool_star]},{name:search,path:/home/ubuntu/models/qwen2.5/teacher-search-r1v3,data_sources:[search_nqhq]}]'

# Per-domain reward fn dispatch — REAL rewards only
RFN=/home/ubuntu/OPSD_OnPolicyDistillation/src/rewards
export REWARD_FNS_HYDRA="{math_dapo:{path:${RFN}/math_reward.py,name:compute_score},math:{path:${RFN}/math_reward.py,name:compute_score},aime24:{path:${RFN}/math_reward.py,name:compute_score},aime25:{path:${RFN}/math_reward.py,name:compute_score},medqa:{path:${RFN}/medical_reward.py,name:compute_score},medical_o1:{path:${RFN}/medical_o1_reward.py,name:compute_score},code_kodcode:{path:${RFN}/code_reward.py,name:compute_score},code_humaneval:{path:${RFN}/code_reward.py,name:compute_score},tool_star:{path:${RFN}/tool_reward.py,name:compute_score},search_nqhq:{path:${RFN}/search_reward.py,name:compute_score}}"

export TRAIN_FILE="${REPO_ROOT}/data/5domain_mixed/train.parquet"
# 7 val sets covering all 5 domains
export VAL_FILES="['${REPO_ROOT}/data/5domain_mixed/val_math500.parquet','${REPO_ROOT}/data/5domain_mixed/val_aime24.parquet','${REPO_ROOT}/data/5domain_mixed/val_aime25.parquet','${REPO_ROOT}/data/5domain_mixed/val_medqa.parquet','${REPO_ROOT}/data/5domain_mixed/val_humaneval.parquet','${REPO_ROOT}/data/5domain_mixed/val_tool.parquet','${REPO_ROOT}/data/5domain_mixed/val_search.parquet']"

export TRAIN_BATCH_SIZE=320
export PPO_MINI_BATCH_SIZE=80   # 320/80 = 4 grad steps per iter (matched to old 256/64)
export PPO_MICRO_BATCH_SIZE_PER_GPU=2

# Stratified sampler: per-batch teacher counts. sum must = TRAIN_BATCH_SIZE
export STRAT_SAMPLER_HYDRA='{math_dapo:64,medical_o1:64,code_kodcode:64,tool_star:64,search_nqhq:64}'
export TOTAL_EPOCHS=2
export TOTAL_TRAINING_STEPS=80
export LEARNING_RATE=1e-6
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
export RUN_ID=Stage2-Iter2-5domain-full80-v5-stratified
export GPUS_PER_NODE=8

bash "${SCRIPT_DIR}/train_opd_mt.sh"
