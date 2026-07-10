"""Interference diagnostics over per-domain branch deltas.

Answers the Gate-B question on the right objects (accumulated K-step deltas,
not per-step gradients): do domains pull the shared student in conflicting
directions, and is any apparent conflict above the sampling-noise floor?

Outputs per parameter-group and global:
  cos(d_a, d_b)      pairwise cosine between domain deltas
  rho                ||sum_d delta_d|| / sum_d ||delta_d||  (with sqrt(N)
                     reported as the no-conflict null, NOT 1.0)
  reliability r_d    split-half cosine within a domain (two half-run deltas)
  A_ab               disattenuated alignment cos(a,b)/sqrt(r_a * r_b)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Dict, List, Sequence

import torch

# Coarse parameter grouping for Qwen2-style checkpoints.
_GROUPS = (
    ("embed", ("embed_tokens",)),
    ("lm_head", ("lm_head",)),
    ("attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
    ("mlp", ("gate_proj", "up_proj", "down_proj")),
    ("norm", ("norm", "ln_")),
)


def _group_of(name: str) -> str:
    low = name.lower()
    for gname, keys in _GROUPS:
        if any(k in low for k in keys):
            return gname
    return "other"


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm() + 1e-12
    return float((a * b).sum() / denom)


def delta_cosine_report(base_get: Callable[[str], torch.Tensor],
                        branch_gets: Dict[str, Callable[[str], torch.Tensor]],
                        names: List[str]) -> dict:
    """Pairwise cosine + rho per group and global, streaming over parameters.

    branch_gets: domain_name -> (param_name -> tensor).
    Accumulates dot products / norms incrementally so nothing large is retained.
    """
    domains = sorted(branch_gets)
    dots = defaultdict(lambda: defaultdict(float))   # group -> (a,b) -> dot
    sqnorms = defaultdict(lambda: defaultdict(float))  # group -> a -> ||d||^2

    for name in names:
        base = base_get(name).float().flatten()
        deltas = {d: branch_gets[d](name).float().flatten() - base for d in domains}
        for g in (_group_of(name), "_global"):
            for i, a in enumerate(domains):
                sqnorms[g][a] += float((deltas[a] ** 2).sum())
                for b in domains[i + 1:]:
                    dots[g][(a, b)] += float((deltas[a] * deltas[b]).sum())

    report = {"domains": domains, "groups": {}}
    for g in sorted(sqnorms):
        norms = {d: math.sqrt(sqnorms[g][d]) for d in domains}
        pair_cos = {
            f"{a}|{b}": dots[g][(a, b)] / (norms[a] * norms[b] + 1e-12)
            for i, a in enumerate(domains) for b in domains[i + 1:]
        }
        sum_sq = sum(sqnorms[g][d] for d in domains) + 2 * sum(dots[g].values())
        rho = math.sqrt(max(sum_sq, 0.0)) / (sum(norms.values()) + 1e-12)
        n = len(domains)
        report["groups"][g] = {
            "pairwise_cos": pair_cos,
            "rho": rho,
            "rho_null_orthogonal": 1.0 / math.sqrt(n),
            "norms": norms,
        }
    return report


def split_half_reliability(base_get: Callable[[str], torch.Tensor],
                           half_a_get: Callable[[str], torch.Tensor],
                           half_b_get: Callable[[str], torch.Tensor],
                           names: List[str]) -> float:
    """Within-domain split-half cosine: reliability r of a domain's delta.

    half_a / half_b are checkpoints of the SAME domain trained on disjoint
    data halves (or two seeds). r near 0 means the delta is noise-dominated
    at this K — increase K before interpreting cross-domain cosines.
    """
    dot, na, nb = 0.0, 0.0, 0.0
    for name in names:
        base = base_get(name).float().flatten()
        da = half_a_get(name).float().flatten() - base
        db = half_b_get(name).float().flatten() - base
        dot += float((da * db).sum())
        na += float((da ** 2).sum())
        nb += float((db ** 2).sum())
    return dot / (math.sqrt(na) * math.sqrt(nb) + 1e-12)


def disattenuated_alignment(cos_ab: float, r_a: float, r_b: float) -> float:
    """Spearman-corrected cross-domain alignment; only meaningful when both
    reliabilities are positive and non-trivial."""
    if r_a <= 0 or r_b <= 0:
        return float("nan")
    return cos_ab / math.sqrt(r_a * r_b)
