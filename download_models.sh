#!/bin/bash
# Download Qwen2.5 student + 5 RL'd teacher checkpoints. ~80 GB total.
# Sequential to keep network/disk sane. Each call resumes if interrupted.
set -e

HF=/home/ubuntu/skypilot-runtime/bin/hf
DEST=/home/ubuntu/models/qwen2.5

mkdir -p "$DEST"

declare -A MODELS=(
    ["Qwen/Qwen2.5-1.5B-Instruct"]="student-1.5b-instruct"
    ["Yukang/Qwen2.5-7B-Open-R1-GRPO"]="teacher-math-yukang"
    ["PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3"]="teacher-search-searchr1"
    ["emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold"]="teacher-tool-toolrl"
    ["RLVR-SvS/SvS-Qwen-Code-7B"]="teacher-code-svs"
    ["prithivMLmods/Qwen-UMLS-7B-Instruct"]="teacher-medical-umls"
)

for repo in "${!MODELS[@]}"; do
    local_name="${MODELS[$repo]}"
    target_dir="${DEST}/${local_name}"
    echo "=== [$(date '+%H:%M:%S')] $repo  ->  $target_dir"
    if [ -f "${target_dir}/.download_complete" ]; then
        echo "  already complete, skipping"
        continue
    fi
    "$HF" download "$repo" --local-dir "$target_dir" --max-workers 8
    touch "${target_dir}/.download_complete"
    df -h /home/ubuntu | tail -1
done

echo "=== ALL DOWNLOADS DONE [$(date '+%H:%M:%S')] ==="
du -sh "${DEST}"/* 2>&1
