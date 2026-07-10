"""Checkpoint I/O: stream tensors from HF checkpoint dirs; save merged models.

Supports safetensors (single file or sharded with index) and pytorch .bin
fallback. Only one tensor is materialized per call, so 1.5B-scale merges run
in modest RAM.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Optional

import torch

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
    _HAS_SAFETENSORS = True
except ImportError:  # pragma: no cover
    _HAS_SAFETENSORS = False


class TensorSource:
    """Lazy name -> tensor access over a HF checkpoint directory."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._name_to_file: Dict[str, str] = {}
        self._bin_cache: Optional[Dict[str, torch.Tensor]] = None
        self._index()

    def _index(self):
        d = self.model_dir
        st_index = os.path.join(d, "model.safetensors.index.json")
        st_single = os.path.join(d, "model.safetensors")
        bin_single = os.path.join(d, "pytorch_model.bin")
        if os.path.exists(st_index):
            with open(st_index) as f:
                self._name_to_file = json.load(f)["weight_map"]
            self._mode = "safetensors"
        elif os.path.exists(st_single):
            if not _HAS_SAFETENSORS:
                raise RuntimeError("safetensors not installed")
            with safe_open(st_single, framework="pt") as f:
                self._name_to_file = {k: "model.safetensors" for k in f.keys()}
            self._mode = "safetensors"
        elif os.path.exists(bin_single):
            self._mode = "bin"
            self._bin_cache = torch.load(bin_single, map_location="cpu",
                                         weights_only=True)
            self._name_to_file = {k: "pytorch_model.bin" for k in self._bin_cache}
        else:
            raise FileNotFoundError(f"no model weights found under {d}")

    def keys(self) -> List[str]:
        return sorted(self._name_to_file)

    def get(self, name: str) -> torch.Tensor:
        if self._mode == "bin":
            return self._bin_cache[name]
        path = os.path.join(self.model_dir, self._name_to_file[name])
        with safe_open(path, framework="pt") as f:
            return f.get_tensor(name)

    __call__ = get


def common_names(sources: List[TensorSource]) -> List[str]:
    names = set(sources[0].keys())
    for s in sources[1:]:
        names &= set(s.keys())
    return sorted(names)


def save_merged_model(merged: Dict[str, torch.Tensor], base_dir: str,
                      out_dir: str, dtype: torch.dtype = torch.bfloat16):
    """Write merged weights as a HF checkpoint, copying config/tokenizer from base."""
    if not _HAS_SAFETENSORS:
        raise RuntimeError("safetensors required to save merged models")
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(base_dir):
        if fname.endswith((".json", ".txt", ".jinja")) and "index" not in fname:
            shutil.copy2(os.path.join(base_dir, fname), os.path.join(out_dir, fname))
    tensors = {k: v.to(dtype).contiguous() for k, v in merged.items()}
    save_file(tensors, os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})
