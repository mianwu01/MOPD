#!/usr/bin/env bash
# Single-teacher OPD branch (Phase-1 / Gate-A' / bake-off input).
#
# Env knobs (defaults in brackets):
#   DOMAIN        routing key, used for naming            [math]
#   TEACHER       teacher model path or HF id             [Qwen/Qwen2.5-Math-7B-Instruct]
#   STUDENT       student model path or HF id             [Qwen/Qwen2.5-1.5B-Instruct]
#   DATA          KDFlow JSONL (see build_kdflow_data.py) [required]
#   SAVE_ROOT     output root                             [./output/branches]
#   GPUS          gpus on this node                       [8]
#   LR            learning rate                           [2e-6]
#   SAVE_STEPS    branch snapshot cadence (K)             [20]
#   MAX_LEN       max prompt+response length              [4096]  # <= math teacher window
#   GEN_MAX_LEN   rollout max new tokens                  [2048]
#   OFT_BLOCK     OFT block size; 0 = full-param          [0]
#   RUN_TAG       suffix for save dir / wandb             [fp]
#
# Examples:
#   DOMAIN=math   DATA=data/kdflow/math.jsonl    bash sweep/train_branch.sh
#   DOMAIN=math   OFT_BLOCK=64 RUN_TAG=oft DATA=data/kdflow/math.jsonl bash sweep/train_branch.sh
#   DOMAIN=medical TEACHER=FreedomIntelligence/HuatuoGPT-o1-7B DATA=data/kdflow/medical.jsonl bash sweep/train_branch.sh
set -euo pipefail

DOMAIN=${DOMAIN:-math}
TEACHER=${TEACHER:-Qwen/Qwen2.5-Math-7B-Instruct}
STUDENT=${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}
DATA=${DATA:?set DATA=<kdflow jsonl>}
SAVE_ROOT=${SAVE_ROOT:-./output/branches}
GPUS=${GPUS:-8}
LR=${LR:-2e-6}
SAVE_STEPS=${SAVE_STEPS:-20}
MAX_LEN=${MAX_LEN:-4096}
GEN_MAX_LEN=${GEN_MAX_LEN:-2048}
OFT_BLOCK=${OFT_BLOCK:-0}
RUN_TAG=${RUN_TAG:-fp}

RUN_NAME="branch_${DOMAIN}_${RUN_TAG}_lr${LR}"
SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"

OPTS=""
OPTS+=" --num_nodes 1 --num_gpus_per_node ${GPUS} --backend fsdp2"
OPTS+=" --train_batch_size 128 --micro_train_batch_size 8"
OPTS+=" --learning_rate ${LR} --lr_warmup_ratio 0.05 --num_epochs 1"
OPTS+=" --save_path ${SAVE_PATH} --save_steps ${SAVE_STEPS}"
OPTS+=" --bf16 True --gradient_checkpointing True --enable_sleep True"

OPTS+=" --student_name_or_path ${STUDENT}"
OPTS+=" --teacher_name_or_path ${TEACHER}"
if [ "${OFT_BLOCK}" != "0" ]; then
  OPTS+=" --oft_block_size ${OFT_BLOCK}"
fi

OPTS+=" --rollout_batch_size 256 --rollout_tp_size 1"
OPTS+=" --rollout_mem_fraction_static 0.6 --n_samples_per_prompt 1"
OPTS+=" --generate_max_len ${GEN_MAX_LEN}"

OPTS+=" --train_dataset_path ${DATA}"
OPTS+=" --max_len ${MAX_LEN} --input_key messages --apply_chat_template True"
OPTS+=" --preprocess_num_workers 16"

# reverse-KL, pure distillation (isolate the aggregation variable; reward off)
OPTS+=" --kd_ratio 1.0 --kd_loss_fn rkl --kd_algorithm vanilla_kd"
OPTS+=" --teacher_tp_size 1 --teacher_dp_size ${GPUS}"
OPTS+=" --teacher_mem_fraction_static 0.4"

OPTS+=" --logging_steps 1 --use_wandb ${USE_WANDB:-False}"
OPTS+=" --wandb_project MOPD-sweep --wandb_run_name ${RUN_NAME}"

python -m kdflow.cli.train_kd_on_policy ${OPTS}
