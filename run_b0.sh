#!/bin/bash
# Launch B0 eval on GPU 2 with conservative memory budget while another session
# uses the rest of the box. If OOM, tune --gpu-memory-utilization down.
set -e

LOG=/home/ubuntu/MOPD/eval_b0.log
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export VLLM_LOGGING_LEVEL=WARNING

cd /home/ubuntu/MOPD
exec /home/ubuntu/miniconda3/envs/opsd/bin/python eval_b0.py \
    --gpu-memory-utilization 0.18 \
    --max-model-len 18432 \
    --max-tokens 16384 \
    --n 16 \
    > "$LOG" 2>&1
