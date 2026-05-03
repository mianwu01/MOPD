#!/bin/bash
# STRICT install per open-r1 Makefile (uv venv, official order, no version overrides)
set -e
set -x

cd /home/ubuntu/open-r1
export PATH=$HOME/.local/bin:$PATH

# 1. uv venv (per Makefile)
uv venv openr1 --python 3.11

# 2. Activate + install deps in Makefile order
source openr1/bin/activate
uv pip install --upgrade pip
uv pip install vllm==0.8.5.post1
uv pip install setuptools
uv pip install flash-attn --no-build-isolation
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"

# 3. Verify
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); print('devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import trl; print('trl', trl.__version__)"
python -c "import accelerate; print('accelerate', accelerate.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"
python -c "import wandb; print('wandb', wandb.__version__)"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
python -c "from flash_attn import flash_attn_func; print('flash_attn_func ok')"

echo "=== STRICT openr1 install complete ==="
