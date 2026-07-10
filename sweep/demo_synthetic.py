#!/usr/bin/env python3
"""No-GPU demo of the whole offline pipeline on tiny synthetic checkpoints.

Creates a fake 2-layer "model" (base + two domain branches whose updates
partially conflict), then runs the real bake-off and the real diagnostics on
it. Takes ~5 seconds on a laptop. Start here if you are new to the project.

  python3 sweep/demo_synthetic.py
"""

import json
import os
import shutil
import subprocess
import sys

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEMO = os.path.join(ROOT, "output", "demo_synthetic")


def write_ckpt(dirname, tensors):
    os.makedirs(dirname, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()},
              os.path.join(dirname, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(dirname, "config.json"), "w") as f:
        json.dump({"model_type": "demo"}, f)


def main():
    if os.path.exists(DEMO):
        shutil.rmtree(DEMO)
    torch.manual_seed(0)

    # A tiny "student": two weight matrices + a norm vector.
    base = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(24, 24),
        "model.layers.0.mlp.up_proj.weight": torch.randn(48, 24),
        "model.norm.weight": torch.ones(24),
    }

    # Two branches: a SHARED update direction plus a CONFLICTING one, so the
    # diagnostics have something real to detect (cos < 0 on the conflict part).
    shared = {k: 0.02 * torch.randn_like(v) for k, v in base.items()}
    conflict = {k: 0.02 * torch.randn_like(v) for k, v in base.items()}
    branch_math = {k: base[k] + shared[k] + conflict[k] for k in base}
    branch_med = {k: base[k] + shared[k] - 0.7 * conflict[k] for k in base}

    write_ckpt(f"{DEMO}/base", base)
    write_ckpt(f"{DEMO}/branch_math", branch_math)
    write_ckpt(f"{DEMO}/branch_medical", branch_med)
    print(f"synthetic checkpoints -> {DEMO}\n")

    py = sys.executable
    subprocess.run([py, os.path.join(HERE, "offline_merge_bakeoff.py"),
                    "--base", f"{DEMO}/base",
                    "--branch", f"math={DEMO}/branch_math",
                    "--branch", f"medical={DEMO}/branch_medical",
                    "--out_dir", f"{DEMO}/merged"], check=True)
    print()
    subprocess.run([py, os.path.join(HERE, "delta_diagnostics.py"),
                    "--base", f"{DEMO}/base",
                    "--branch", f"math={DEMO}/branch_math",
                    "--branch", f"medical={DEMO}/branch_medical",
                    "--out", f"{DEMO}/diagnostics.json"], check=True)

    print("\nWhat to look at:")
    print(f"  {DEMO}/merged/<operator>/       one merged checkpoint per operator")
    print(f"  {DEMO}/merged/v2_ord/merge_manifest.json")
    print(f"  {DEMO}/diagnostics.json         per-group cosines and rho")
    print("\nBecause the branches share one component and conflict on another,")
    print("pairwise cos lands between 0 and 1 and rho sits above the 1/sqrt(2)")
    print("null — rerun with `- 0.7 * conflict` changed to `+ 0.7` in this file")
    print("and watch cos and rho rise; make it `-1.0` and watch them fall.")


if __name__ == "__main__":
    main()
