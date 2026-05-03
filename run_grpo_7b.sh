#!/bin/bash
# Launch open-r1 GRPO training for Qwen2.5-7B-Instruct on OpenR1-Math-220k.
# 8× H100, vLLM colocate, full-shard ZeRO-3, ~6-10h for max_steps=500.
set -e

OPENR1_DIR=/home/ubuntu/open-r1
LOG_DIR=/home/ubuntu/MOPD/logs
OUT_DIR=/home/ubuntu/openr1_outputs/Qwen2.5-7B-Open-R1-GRPO

mkdir -p "$LOG_DIR" "$OUT_DIR"

export PATH=$OPENR1_DIR/openr1/bin:$PATH
export PYTHONPATH=$OPENR1_DIR/src:$PYTHONPATH
export PYTHONNOUSERSITE=1
export ACCELERATE_LOG_LEVEL=info

# wandb cfg (already logged in via ~/.netrc as nothern; entity=meanwork is one of nothern's teams)
export WANDB_PROJECT=Qwen2.5-7B-OpenR1-GRPO
export WANDB_ENTITY=meanwork
export WANDB_RUN_GROUP=open-r1-grpo
export WANDB_TAGS=qwen25-7b,openr1-math-220k,grpo,teacher-training

cd "$OPENR1_DIR"

exec $OPENR1_DIR/openr1/bin/accelerate launch \
    --config_file recipes/accelerate_configs/fsdp.yaml \
    --num_processes 8 \
    src/open_r1/grpo.py \
    --config recipes/Qwen2.5-7B-Instruct/grpo/config_demo.yaml
