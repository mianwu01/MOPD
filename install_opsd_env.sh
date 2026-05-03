#!/bin/bash
# Install all OPSD env deps into the conda env `opsd`.
# Targets: CUDA 12.8 + H100 (Hopper SM 90).
# Pinned to memory-validated stack (ray 2.55+, vllm 0.19, verl 0.7.1).
set -e
set -x

CONDA_PIP=/home/ubuntu/miniconda3/envs/opsd/bin/pip
CONDA_PY=/home/ubuntu/miniconda3/envs/opsd/bin/python

# 1. torch 2.9.x with cu128 (matches CUDA 12.8). Keep simple - one pip call.
$CONDA_PIP install --no-cache-dir \
    "torch==2.9.1" "torchvision==0.24.1" "torchaudio==2.9.1" \
    --index-url https://download.pytorch.org/whl/cu128

# 2. Core ML libs
$CONDA_PIP install --no-cache-dir \
    "transformers==4.57.1" \
    "tensordict" \
    "hydra-core" \
    "datasets" \
    "hf_transfer" \
    "huggingface_hub[cli]" \
    "math_verify" \
    "accelerate" \
    "peft" \
    "wandb" \
    "tqdm" \
    "pyyaml" \
    "omegaconf"

# 3. Ray + vLLM + verl. verl 0.7.1 pulls correct verl-internal deps.
$CONDA_PIP install --no-cache-dir \
    "ray==2.55.1" \
    "vllm==0.19.0"

$CONDA_PIP install --no-cache-dir "verl==0.7.1"

# 4. Verify
$CONDA_PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); print('devices:', torch.cuda.device_count())"
$CONDA_PY -c "import verl; print('verl', verl.__version__)"
$CONDA_PY -c "import vllm; print('vllm', vllm.__version__)"
$CONDA_PY -c "import ray; print('ray', ray.__version__)"
$CONDA_PY -c "import transformers; print('transformers', transformers.__version__)"
$CONDA_PY -c "import tensordict; print('tensordict OK')"

echo "=== OPSD env install complete ==="
