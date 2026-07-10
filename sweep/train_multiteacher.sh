#!/usr/bin/env bash
# Multi-teacher OPD baseline (the opponent): naive or per-teacher-weighted
# Euclidean sum, with the math teacher's 4k context cap enforced.
#
# Env knobs:
#   STUDENT        [Qwen/Qwen2.5-1.5B-Instruct]
#   DATA           mixed KDFlow JSONL with teacher_routing_key   [required]
#   TEACHERS_JSON  multi_teacher_config                          [sweep/configs/teachers_5domain.json]
#   WEIGHTS_JSON   per-teacher loss weights ('' = uniform)       ['']
#   MAXLEN_JSON    per-teacher context caps                      [sweep/configs/teacher_max_len.json]
#   SAVE_ROOT      [./output/multiteacher]   GPUS [8]   LR [2e-6]
#   SAVE_STEPS     [20]   RUN_TAG [uniform]
set -euo pipefail

STUDENT=${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}
DATA=${DATA:?set DATA=<mixed kdflow jsonl>}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TEACHERS_JSON=${TEACHERS_JSON:-${SCRIPT_DIR}/configs/teachers_5domain.json}
WEIGHTS_JSON=${WEIGHTS_JSON:-}
MAXLEN_JSON=${MAXLEN_JSON:-${SCRIPT_DIR}/configs/teacher_max_len.json}
SAVE_ROOT=${SAVE_ROOT:-./output/multiteacher}
GPUS=${GPUS:-8}
LR=${LR:-2e-6}
SAVE_STEPS=${SAVE_STEPS:-20}
RUN_TAG=${RUN_TAG:-uniform}

RUN_NAME="mt_${RUN_TAG}_lr${LR}"
SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"

OPTS=""
OPTS+=" --num_nodes 1 --num_gpus_per_node ${GPUS} --backend fsdp2"
OPTS+=" --train_batch_size 128 --micro_train_batch_size 8"
OPTS+=" --learning_rate ${LR} --lr_warmup_ratio 0.05 --num_epochs 1"
OPTS+=" --save_path ${SAVE_PATH} --save_steps ${SAVE_STEPS}"
OPTS+=" --bf16 True --gradient_checkpointing True --enable_sleep True"

OPTS+=" --student_name_or_path ${STUDENT}"

OPTS+=" --rollout_batch_size 256 --rollout_tp_size 1"
OPTS+=" --rollout_mem_fraction_static 0.6 --n_samples_per_prompt 1"
OPTS+=" --generate_max_len ${GEN_MAX_LEN:-2048}"

OPTS+=" --train_dataset_path ${DATA}"
OPTS+=" --max_len ${MAX_LEN:-4096} --input_key messages --apply_chat_template True"
OPTS+=" --preprocess_num_workers 16"
OPTS+=" --teacher_routing_key teacher_routing_key"

OPTS+=" --kd_ratio 1.0 --kd_loss_fn rkl --kd_algorithm vanilla_kd"
OPTS+=" --multi_teacher_config ${TEACHERS_JSON}"
OPTS+=" --teacher_max_len ${MAXLEN_JSON}"
if [ -n "${WEIGHTS_JSON}" ]; then
  OPTS+=" --teacher_loss_weights ${WEIGHTS_JSON}"
fi
OPTS+=" --teacher_tp_size 1 --teacher_dp_size ${GPUS}"
OPTS+=" --teacher_mem_fraction_static 0.4"

OPTS+=" --logging_steps 1 --use_wandb ${USE_WANDB:-False}"
OPTS+=" --wandb_project MOPD-sweep --wandb_run_name ${RUN_NAME}"

python -m kdflow.cli.train_kd_on_policy ${OPTS}
