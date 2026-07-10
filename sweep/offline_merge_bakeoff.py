#!/usr/bin/env python3
"""Offline merge bake-off: apply every merge operator to per-domain branches.

This is the one-day go/no-go for the geometry toolkit: if the geometric merges
(v2_ord and friends) cannot beat plain_avg / ta OFFLINE, they will not do so
in-loop, and Phase-4 engineering should not be built.

Usage:
  python3 sweep/offline_merge_bakeoff.py \
      --base /models/Qwen2.5-1.5B-Instruct \
      --branch math=/ckpt/branch_math/epoch_1_global_step_80 \
      --branch medical=/ckpt/branch_medical/epoch_1_global_step_80 \
      --ops plain_avg ta ties v2_ord v2_ord_noxc v2_dir_only v2_mag_only v3_polar \
      --out_dir /ckpt/merged

Each op writes a full HF checkpoint to {out_dir}/{op}/ (config + tokenizer
copied from base) plus a manifest.json recording inputs and settings.
PEFT (LoRA/OFT) branches must be materialized to full checkpoints first —
see materialize_peft.py.
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mopd_merge import OP_REGISTRY, merge_models
from mopd_merge.ckpt_io import TensorSource, common_names, save_merged_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("bakeoff")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base (pre-branch) student checkpoint dir")
    p.add_argument("--branch", action="append", required=True,
                   help="name=path of a per-domain branch checkpoint; repeatable")
    p.add_argument("--ops", nargs="+", default=sorted(OP_REGISTRY),
                   choices=sorted(OP_REGISTRY))
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-branch TA weights (same order as --branch); default uniform")
    p.add_argument("--ties_density", type=float, default=0.2)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    dtype = getattr(torch, args.dtype)

    branches = dict(spec.split("=", 1) for spec in args.branch)
    log.info("base: %s", args.base)
    log.info("branches: %s", branches)

    base_src = TensorSource(args.base)
    branch_srcs = {name: TensorSource(path) for name, path in branches.items()}
    names = common_names([base_src] + list(branch_srcs.values()))
    log.info("%d shared parameters", len(names))

    order = sorted(branch_srcs)
    gets = [branch_srcs[n].get for n in order]

    for op in args.ops:
        t0 = time.time()
        out = os.path.join(args.out_dir, op)
        log.info("merging with %s -> %s", op, out)
        merged = merge_models(base_src.get, gets, names, op,
                              weights=args.weights,
                              **({"density": args.ties_density} if op == "ties" else {}))
        save_merged_model(merged, args.base, out, dtype=dtype)
        with open(os.path.join(out, "merge_manifest.json"), "w") as f:
            json.dump({
                "op": op,
                "base": args.base,
                "branches": {n: branches[n] for n in order},
                "weights": args.weights or [1.0] * len(order),
                "ties_density": args.ties_density if op == "ties" else None,
                "elapsed_sec": round(time.time() - t0, 1),
            }, f, indent=2)
        log.info("done %s in %.1fs", op, time.time() - t0)


if __name__ == "__main__":
    main()
