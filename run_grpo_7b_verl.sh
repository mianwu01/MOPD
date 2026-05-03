#!/bin/bash
# verl-based GRPO training for Qwen2.5-7B-Instruct on OpenR1-Math-220k subset.
# Replaces broken open-r1/TRL path. Uses opsd env + verl 0.7.1 (proven on this stack).
set -e

OPSD_REPO=/home/ubuntu/OPSD_OnPolicyDistillation
SRC_ROOT="${OPSD_REPO}/src"
DATA_DIR="${OPSD_REPO}/data/openr1_math_grpo"
PYBIN=/home/ubuntu/miniconda3/envs/opsd/bin/python

ulimit -n 65535
export PYTHONPATH="${SRC_ROOT}:$PYTHONPATH"
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 29500-39999 -n 1)}

# wandb (logged in as nothern, has meanwork team access)
export WANDB_PROJECT=Qwen2.5-7B-OpenR1-GRPO
export WANDB_ENTITY=meanwork
export WANDB_TAGS=qwen25-7b,verl,openr1-math,grpo

# Model: pre-cached
MODEL_PATH=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
MODEL_NAME=Qwen2.5-7B-Instruct

# Train hyperparams (8× H100 80GB)
TRAIN_BATCH_SIZE=128       # prompts/step
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE_PER_GPU=2
LEARNING_RATE=1e-6
TOTAL_EPOCHS=1
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=2048
ROLLOUT_N=8                 # k samples per prompt for GRPO advantage
TP_SIZE=1
GPU_MEMORY_UTIL=0.5
KL_LOSS_COEF=0.001
ENTROPY_COEFF=0
ROLLOUT_NAME=vllm           # opsd env has vllm 0.19, no sglang

TEMPERATURE=1.0
TOP_P=1.0
TOP_K=-1
VAL_TEMPERATURE=0.7
VAL_TOP_P=0.8
VAL_TOP_K=20
VAL_N=4

GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

EXP_NAME=Qwen2.5-7B-OpenR1-GRPO-bs${TRAIN_BATCH_SIZE}-lr${LEARNING_RATE}-n${ROLLOUT_N}
OUTPUT_DIR=/home/ubuntu/openr1_outputs/${EXP_NAME}
mkdir -p "$OUTPUT_DIR"

TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_MATH500="${DATA_DIR}/val_math500.parquet"
VAL_AIME24="${DATA_DIR}/val_aime24.parquet"
VAL_AIME25="${DATA_DIR}/val_aime25.parquet"

# Background GPU monitor
GPU_LOG="${OUTPUT_DIR}/gpu_memory_monitor.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv -l 5 > "${GPU_LOG}" 2>&1 &
GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID 2>/dev/null" EXIT

set -x
exec "$PYBIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$TRAIN_FILE \
    data.val_files="['$VAL_MATH500','$VAL_AIME24','$VAL_AIME25']" \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.top_k=$TOP_K \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.top_k=$VAL_TOP_K \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    reward.custom_reward_function.path="${SRC_ROOT}/rewards/math_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=Qwen2.5-7B-OpenR1-GRPO \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.validation_data_dir=$OUTPUT_DIR \
    trainer.val_before_train=True \
    trainer.log_val_generations=10 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=$TOTAL_EPOCHS
