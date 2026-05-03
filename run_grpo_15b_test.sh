#!/bin/bash
# Diagnostic: run open-r1 ORIGINAL Qwen2.5-1.5B GRPO recipe with no modifications.
# Goal: prove the env can run open-r1 GRPO at all.
set -e

OPENR1_DIR=/home/ubuntu/open-r1
export PATH=$OPENR1_DIR/openr1/bin:$PATH
export PYTHONPATH=$OPENR1_DIR/src:$PYTHONPATH
export PYTHONNOUSERSITE=1
export ACCELERATE_LOG_LEVEL=info
export WANDB_MODE=offline   # don't pollute the meanwork project with diagnostic runs

cd $OPENR1_DIR
exec $OPENR1_DIR/openr1/bin/accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes 8 \
    src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml \
    --output_dir /home/ubuntu/openr1_outputs/diag_15b \
    --max_steps 3 \
    --report_to none \
    --push_to_hub false \
    --vllm_mode colocate
