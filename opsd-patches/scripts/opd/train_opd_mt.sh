#!/bin/bash
# Stage 2 multi-teacher OPD launcher.
# Reads env vars then calls `python -m opd.main_opd` with Hydra overrides for
# the new opd.teachers / opd.reward_fns config blocks (added in src/opd/).
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

# Force opsd conda env (which python resolves to base by default).
PYBIN=${PYBIN:-/home/ubuntu/miniconda3/envs/opsd/bin/python}
export PATH=/home/ubuntu/miniconda3/envs/opsd/bin:$PATH

ulimit -n 65535
export PYTHONPATH="${SRC_ROOT}:$PYTHONPATH"
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 29500-39999 -n 1)}

# ---- required env vars ----
MODEL_PATH=${MODEL_PATH:?MODEL_PATH (student) required}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_PATH")}
# Hydra-formatted strings, see iter1 wrapper for examples:
TEACHERS_HYDRA=${TEACHERS_HYDRA:?TEACHERS_HYDRA hydra-list string required}
REWARD_FNS_HYDRA=${REWARD_FNS_HYDRA:?REWARD_FNS_HYDRA hydra-dict string required}
PRIMARY_TEACHER_PATH=${PRIMARY_TEACHER_PATH:?PRIMARY_TEACHER_PATH (== teachers[0].path) required}
DEFAULT_REWARD_FN=${DEFAULT_REWARD_FN:?DEFAULT_REWARD_FN (key in reward_fns) required}
DEFAULT_TEACHER_NAME=${DEFAULT_TEACHER_NAME:-math}

TRAIN_FILE=${TRAIN_FILE:?TRAIN_FILE required}
VAL_FILES=${VAL_FILES:?VAL_FILES (hydra list) required}

# ---- training hyperparams ----
train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
learning_rate=${LEARNING_RATE:-1e-6}
total_epochs=${TOTAL_EPOCHS:-1}
total_training_steps=${TOTAL_TRAINING_STEPS:-null}
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-50}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
rollout_n=${ROLLOUT_N:-1}
val_n=${VAL_N:-8}
tp_size=${TP_SIZE:-1}
gpu_memory_util=${GPU_MEMORY_UTIL:-0.5}
rollout_name=${ROLLOUT_NAME:-vllm}

opd_loss_type=${OPD_LOSS_TYPE:-reverse_kl}
opd_chunk_size=${OPD_CHUNK_SIZE:-256}
opd_max_length=${OPD_MAX_LENGTH:-16384}

temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
val_temperature=${VAL_TEMPERATURE:-0.7}
val_top_p=${VAL_TOP_P:-0.8}
val_top_k=${VAL_TOP_K:-20}

val_before_train=${VAL_BEFORE_TRAIN:-True}

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}

RUN_ID=${RUN_ID:-Stage2-Iter1}
EXP_NAME=${EXP_NAME:-${MODEL_NAME}-${RUN_ID}-mt-${opd_loss_type}-lr${learning_rate}-bs${train_batch_size}}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${REPO_ROOT}/outputs"}
output_dir="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "$output_dir"

# Background GPU monitor
GPU_MONITOR_LOG="${output_dir}/gpu_memory_monitor.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv -l 5 > "${GPU_MONITOR_LOG}" 2>&1 &
GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID 2>/dev/null" EXIT

echo "=== Stage 2 Multi-Teacher OPD Configuration ==="
echo "MODEL_PATH (student): $MODEL_PATH"
echo "PRIMARY_TEACHER_PATH: $PRIMARY_TEACHER_PATH"
echo "TEACHERS_HYDRA: $TEACHERS_HYDRA"
echo "REWARD_FNS_HYDRA: $REWARD_FNS_HYDRA"
echo "DEFAULT_REWARD_FN: $DEFAULT_REWARD_FN"
echo "TRAIN_FILE: $TRAIN_FILE"
echo "VAL_FILES: $VAL_FILES"
echo "n_gpus: $GPUS_PER_NODE  bs=$train_batch_size  mini_bs=$ppo_mini_batch_size  ubs=$ppo_micro_batch_size_per_gpu"
echo "epochs=$total_epochs  total_steps=$total_training_steps  test_freq=$test_freq  save_freq=$save_freq"
echo "EXP_NAME: $EXP_NAME"
echo "output_dir: $output_dir"
echo "==============================================="

# Build optional total_training_steps override
EXTRA_ARGS=()
if [ "$total_training_steps" != "null" ]; then
    EXTRA_ARGS+=("trainer.total_training_steps=$total_training_steps")
fi

"$PYBIN" -m opd.main_opd \
    --config-path "${SRC_ROOT}/opd/config" \
    --config-name opd_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files="$VAL_FILES" \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.train_batch_size=$train_batch_size \
    ${STRAT_SAMPLER_HYDRA:++data.sampler.class_path=pkg://opd/stratified_sampler} \
    ${STRAT_SAMPLER_HYDRA:++data.sampler.class_name=StratifiedDataSourceSampler} \
    ${STRAT_SAMPLER_HYDRA:++data.sampler.per_source=$STRAT_SAMPLER_HYDRA} \
    ${STRAT_SAMPLER_HYDRA:+data.dataloader_num_workers=0} \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$learning_rate \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.model.path=$PRIMARY_TEACHER_PATH \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_size \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_util \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=$val_n \
    opd.loss_type=${opd_loss_type} \
    opd.chunk_size=${opd_chunk_size} \
    opd.max_length=${opd_max_length} \
    ${OPD_REWARD_BETA:+opd.reward_beta=$OPD_REWARD_BETA} \
    +actor_rollout_ref.opd_mt.teachers="$TEACHERS_HYDRA" \
    +actor_rollout_ref.opd_mt.default_teacher="$DEFAULT_TEACHER_NAME" \
    +opd.reward_fns="$REWARD_FNS_HYDRA" \
    +opd.default_reward_fn=$DEFAULT_REWARD_FN \
    reward.custom_reward_function.path="${SRC_ROOT}/rewards/math_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${WANDB_PROJECT:-OPSD-multi-teacher-baseline} \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.default_local_dir=$output_dir \
    +trainer.validation_data_dir=$output_dir \
    trainer.val_before_train=${val_before_train} \
    trainer.log_val_generations=10 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Multi-teacher OPD training completed ==="
echo "Output: $output_dir"
