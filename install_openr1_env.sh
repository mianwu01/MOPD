#!/bin/bash
# Install open-r1 deps into conda env `openr1` (Python 3.11).
# Pinned per open-r1/setup.py and Makefile.
set -e
set -x

PIP=/home/ubuntu/miniconda3/envs/openr1/bin/pip
PY=/home/ubuntu/miniconda3/envs/openr1/bin/python

$PY -m pip install --upgrade pip
$PIP install --no-cache-dir setuptools

# 1. torch 2.6 cu124 (open-r1 pin) — from PyTorch official index
$PIP install --no-cache-dir "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124

# 2. vllm 0.8.5.post1 (open-r1 Makefile)
$PIP install --no-cache-dir "vllm==0.8.5.post1"

# 3. flash-attn (no build isolation to use installed torch)
$PIP install --no-cache-dir --no-build-isolation flash-attn

# 4. open-r1 + its deps via editable install
cd /home/ubuntu/open-r1
GIT_LFS_SKIP_SMUDGE=1 $PIP install --no-cache-dir -e ".[dev]"

# 5. Verify
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); print('devices:', torch.cuda.device_count())"
$PY -c "import vllm; print('vllm', vllm.__version__)"
$PY -c "import trl; print('trl', trl.__version__)"
$PY -c "import accelerate; print('accelerate', accelerate.__version__)"
$PY -c "import transformers; print('transformers', transformers.__version__)"
$PY -c "import deepspeed; print('deepspeed', deepspeed.__version__)"
$PY -c "import wandb; print('wandb', wandb.__version__)"
$PY -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
$PY -c "import open_r1; print('open_r1 importable')" || echo "open_r1 import failed (need PYTHONPATH=src)"

echo "=== openr1 env install complete ==="
