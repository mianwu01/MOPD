#!/bin/bash
set -e
ALIGN=/home/ubuntu/MOPD/eval_alignment.py
LOGS=/home/ubuntu/MOPD/logs
MERGED=/home/ubuntu/MOPD/merged_ckpts
PYBIN=/home/ubuntu/miniconda3/envs/opsd/bin/python

BASE_15B=/home/ubuntu/models/qwen2.5/student-1.5b-instruct
FUTUREMA=/home/ubuntu/models/qwen2.5/teacher-math-futurema-merged
MATH7B=/home/ubuntu/models/qwen2.5/teacher-math-math

# 5 alignment evals — student × teacher
declare -a runs=(
  "baseline_vs_futurema|$BASE_15B|$FUTUREMA"
  "expc39_vs_futurema|$MERGED/expc_step39|$FUTUREMA"
  "v10g100_vs_futurema|$MERGED/v10g_step100|$FUTUREMA"
  "baseline_vs_math7b|$BASE_15B|$MATH7B"
  "v10b20_vs_math7b|$MERGED/v10b_step20|$MATH7B"
)

for run_spec in "${runs[@]}"; do
  IFS='|' read -r tag student teacher <<< "$run_spec"
  echo "=== START ${tag} at $(date +%H:%M:%S) ==="
  PYTHONNOUSERSITE=1 $PYBIN $ALIGN \
    --student "$student" \
    --teacher "$teacher" \
    --n_problems 100 \
    --student_gpu 6 \
    --teacher_gpu 7 \
    --out_json $LOGS/align_${tag}.json \
    > $LOGS/align_${tag}.log 2>&1
  echo "=== DONE ${tag} at $(date +%H:%M:%S) ==="
done

echo "ALL_ALIGN_DONE"
