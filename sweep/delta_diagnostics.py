#!/usr/bin/env python3
"""Delta-interference diagnostics: the corrected Gate-B instrument.

Measures, on accumulated K-step per-domain deltas (NOT per-step gradients):
  - pairwise cos(delta_a, delta_b) per parameter group and global
  - rho = ||sum delta|| / sum ||delta||, against the 1/sqrt(N) no-conflict null
  - optional split-half reliability per domain (two half-run/seed checkpoints)
    and the disattenuated cross-domain alignment A_ab = cos / sqrt(r_a r_b)

Interpretation (pre-registered):
  cos ~= 0 (within null) and rho ~= 1/sqrt(N)  -> no measurable conflict
  cos << 0 beyond null on some pair            -> genuine interference; the
                                                  geometry method has a target
  reliability r ~ 0                             -> K too small; deltas are noise,
                                                  increase K before concluding

Usage:
  python3 sweep/delta_diagnostics.py \
      --base /models/Qwen2.5-1.5B-Instruct \
      --branch math=/ckpt/branch_math/epoch_1_global_step_80 \
      --branch medical=/ckpt/branch_med/epoch_1_global_step_80 \
      [--half math=/ckpt/branch_math_halfA,/ckpt/branch_math_halfB] \
      --out diagnostics_report.json
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mopd_merge.ckpt_io import TensorSource, common_names
from mopd_merge.diagnostics import (
    delta_cosine_report, disattenuated_alignment, split_half_reliability,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--branch", action="append", required=True, help="name=path")
    p.add_argument("--half", action="append", default=[],
                   help="name=pathA,pathB split-half checkpoints for reliability")
    p.add_argument("--out", default="diagnostics_report.json")
    args = p.parse_args()

    branches = dict(spec.split("=", 1) for spec in args.branch)
    base_src = TensorSource(args.base)
    branch_srcs = {n: TensorSource(p_) for n, p_ in branches.items()}
    names = common_names([base_src] + list(branch_srcs.values()))

    report = delta_cosine_report(base_src.get,
                                 {n: s.get for n, s in branch_srcs.items()}, names)

    reliability = {}
    for spec in args.half:
        name, paths = spec.split("=", 1)
        pa, pb = paths.split(",")
        reliability[name] = split_half_reliability(
            base_src.get, TensorSource(pa).get, TensorSource(pb).get, names)
    report["split_half_reliability"] = reliability

    if reliability:
        disatt = {}
        for pair, cos in report["groups"]["_global"]["pairwise_cos"].items():
            a, b = pair.split("|")
            if a in reliability and b in reliability:
                disatt[pair] = disattenuated_alignment(cos, reliability[a], reliability[b])
        report["disattenuated_alignment"] = disatt

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    g = report["groups"]["_global"]
    n = len(report["domains"])
    print(f"\ndomains: {report['domains']}")
    print(f"global rho = {g['rho']:.4f}   (no-conflict null = 1/sqrt({n}) = {1/math.sqrt(n):.4f})")
    print("pairwise cos (global):")
    for pair, c in sorted(g["pairwise_cos"].items(), key=lambda kv: kv[1]):
        print(f"  {pair:30s} {c:+.4f}")
    if reliability:
        print("split-half reliability:", {k: round(v, 3) for k, v in reliability.items()})
    if report.get("disattenuated_alignment"):
        print("disattenuated alignment:",
              {k: round(v, 3) for k, v in report["disattenuated_alignment"].items()})
    print(f"\nfull per-group report -> {args.out}")


if __name__ == "__main__":
    main()
