#!/usr/bin/env python3
"""Paired per-problem comparison of two models' eval results.

The min-over-domains headline is unmeasurable with unpaired single runs
(HumanEval+ SE ~ 3.9pp); this implements the required paired analysis:
difference-in-differences on the identical problem set, with a bootstrap CI
and a paired permutation test.

Input: two JSONL files with one record per problem: {"id": ..., "correct": 0/1}
(mean@k: give "correct" as the fraction in [0,1]). Rows are matched on "id".

Usage:
  python3 sweep/paired_compare.py --a results_method.jsonl --b results_baseline.jsonl
"""

import argparse
import json
import random


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[str(r["id"])] = float(r["correct"])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="method results jsonl")
    p.add_argument("--b", required=True, help="baseline results jsonl")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ra, rb = load(args.a), load(args.b)
    ids = sorted(set(ra) & set(rb))
    if len(ids) < len(ra) or len(ids) < len(rb):
        print(f"warning: unmatched ids dropped (a={len(ra)}, b={len(rb)}, matched={len(ids)})")
    diffs = [ra[i] - rb[i] for i in ids]
    n = len(diffs)
    mean_a = sum(ra[i] for i in ids) / n
    mean_b = sum(rb[i] for i in ids) / n
    delta = mean_a - mean_b

    rng = random.Random(args.seed)
    boots = []
    for _ in range(args.boot):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    lo, hi = boots[int(0.025 * args.boot)], boots[int(0.975 * args.boot)]

    # paired sign-flip permutation test
    extreme = 0
    for _ in range(args.boot):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(s) >= abs(delta):
            extreme += 1
    pval = (extreme + 1) / (args.boot + 1)

    print(f"n matched problems : {n}")
    print(f"method mean        : {mean_a:.4f}")
    print(f"baseline mean      : {mean_b:.4f}")
    print(f"paired delta       : {delta:+.4f}  (95% CI [{lo:+.4f}, {hi:+.4f}])")
    print(f"permutation p      : {pval:.4f}")
    if lo <= 0 <= hi:
        print("verdict            : NOT significant — do not claim a win on this domain")
    else:
        print("verdict            : significant at 95%")


if __name__ == "__main__":
    main()
