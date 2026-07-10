"""Merge operators over per-domain branch checkpoints.

Every operator has signature
    op(name, base_tensor, branch_tensors, weights, **kwargs) -> merged_tensor
where branch_tensors is a list of per-domain tensors for parameter `name` and
weights is a normalized list of per-domain coefficients (Task-Arithmetic
lambdas). `merge_models` streams over parameter names so peak memory is
O(N_domains * one tensor).

Operators
---------
plain_avg        FedAvg on deltas (the make-or-break control)
ta               Task Arithmetic: W0 + sum(lambda_d * delta_d)
ties             TIES: trim -> elect sign -> disjoint mean
v2_ord           ORD: so(d)-averaged Procrustes rotation (optional ×c) + TA/TIES residual
v2_dir_only      V2 ablation: rotation channel only (residuals dropped)
v2_mag_only      V2 ablation: residual channel only (rotation = identity)
v3_polar         polar(delta): extrinsic-mean direction + SPD-mean magnitude
                 (carries the det/far-from-identity landmine; guarded + logged)
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Sequence

import torch

from .decompose import ord_decompose, ord_reconstruct
from .linalg import cayley_exp, cayley_log, polar_decompose, polar_factor

log = logging.getLogger("mopd_merge")

# Parameters where geometric treatment is meaningless; always merged with TA.
_ALWAYS_LINEAR = ("bias", "norm", "ln_", "layernorm")


def _is_geometric(name: str, t: torch.Tensor) -> bool:
    if t.ndim != 2:
        return False
    lowered = name.lower()
    return not any(k in lowered for k in _ALWAYS_LINEAR)


def _normalize(weights: Sequence[float]) -> List[float]:
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("merge weights must sum to a positive value")
    return [float(w) / s for w in weights]


# --------------------------------------------------------------------------- #
# Linear-family operators
# --------------------------------------------------------------------------- #

def op_plain_avg(name, base, branches, weights, **kw):
    delta = sum((b.float() - base.float()) for b in branches) / len(branches)
    return (base.float() + delta).to(base.dtype)


def op_ta(name, base, branches, weights, *, scale: float = 1.0, **kw):
    weights = _normalize(weights)
    delta = sum(w * (b.float() - base.float()) for w, b in zip(weights, branches))
    return (base.float() + scale * delta).to(base.dtype)


def op_ties(name, base, branches, weights, *, density: float = 0.2,
            scale: float = 1.0, **kw):
    """TIES-merging (Yadav et al. 2023): trim, elect sign, disjoint mean."""
    weights = _normalize(weights)
    deltas = [w * (b.float() - base.float()) for w, b in zip(weights, branches)]
    trimmed = []
    for d in deltas:
        k = max(1, int(density * d.numel()))
        thresh = d.abs().flatten().kthvalue(d.numel() - k + 1).values
        trimmed.append(torch.where(d.abs() >= thresh, d, torch.zeros_like(d)))
    stacked = torch.stack(trimmed)
    elected = torch.sign(stacked.sum(dim=0))
    agree = (torch.sign(stacked) == elected.unsqueeze(0)) & (stacked != 0)
    contrib = torch.where(agree, stacked, torch.zeros_like(stacked)).sum(dim=0)
    count = agree.sum(dim=0).clamp(min=1)
    merged_delta = contrib / count
    return (base.float() + scale * merged_delta).to(base.dtype)


# --------------------------------------------------------------------------- #
# V2: Orthogonal-Residual Decoupling merge
# --------------------------------------------------------------------------- #

def op_v2_ord(name, base, branches, weights, *, xc: bool = True,
              residual: str = "ta", rotation: str = "sod",
              ties_density: float = 0.2, polar_method: str = "svd", **kw):
    """Direction channel: Procrustes rotations averaged in so(d) (optionally
    norm-restored by ×c). Magnitude channel: residuals merged by TA or TIES.

    rotation: "sod" (merge rotations) or "identity" (magnitude-only ablation)
    residual: "ta", "ties", or "none" (direction-only ablation)
    """
    weights = _normalize(weights)
    if not _is_geometric(name, base):
        return op_ta(name, base, branches, weights)

    side = None
    qs, es = [], []
    for b in branches:
        r, e, side_b, info = ord_decompose(b, base, side=side or "auto",
                                           method=polar_method)
        side = side or side_b
        q, ok = cayley_log(r)
        if not ok:
            log.warning("%s: rotation outside Cayley domain (det=%s) — "
                        "falling back to TA for this tensor", name, info["det"])
            return op_ta(name, base, branches, weights)
        qs.append(q)
        es.append(e)

    if rotation == "identity":
        r_merged = torch.eye(qs[0].shape[0], dtype=torch.float32)
    else:
        q_sum = sum(w * q for w, q in zip(weights, qs))
        if xc:
            num = sum(w * torch.linalg.matrix_norm(q) for w, q in zip(weights, qs))
            den = torch.linalg.matrix_norm(q_sum) + 1e-12
            q_sum = q_sum * (num / den)
        r_merged = cayley_exp(q_sum)

    if residual == "none":
        e_merged = torch.zeros_like(es[0].float())
    elif residual == "ta":
        e_merged = sum(w * e.float() for w, e in zip(weights, es))
    elif residual == "ties":
        zero = torch.zeros_like(base)
        e_merged = op_ties(name, zero, [e for e in es], weights,
                           density=ties_density).float()
    else:
        raise ValueError(f"bad residual mode: {residual}")

    return ord_reconstruct(r_merged, e_merged.to(base.dtype), base, side)


def op_v2_dir_only(name, base, branches, weights, **kw):
    return op_v2_ord(name, base, branches, weights, residual="none", **kw)


def op_v2_mag_only(name, base, branches, weights, **kw):
    return op_v2_ord(name, base, branches, weights, rotation="identity", **kw)


# --------------------------------------------------------------------------- #
# V3: polar decomposition of the delta itself (Muon-flavored)
# --------------------------------------------------------------------------- #

def op_v3_polar(name, base, branches, weights, *, polar_method: str = "svd", **kw):
    """delta_d = Q_d @ H_d; extrinsic mean of Q_d (re-projected to orthogonal),
    SPD (convex) mean of H_d. WARNING: Q_d of an update matrix is generically
    far from identity and may sit in the det=-1 component — diagnostics are
    logged; results from this operator carry that caveat by design.
    """
    weights = _normalize(weights)
    if not _is_geometric(name, base):
        return op_ta(name, base, branches, weights)

    qs, hs, sides = [], [], []
    for b in branches:
        d = b.float() - base.float()
        q, h, s = polar_decompose(d, method=polar_method)
        qs.append(q)
        hs.append(h.float())
        sides.append(s)
    if len(set(sides)) != 1:
        return op_ta(name, base, branches, weights)
    side = sides[0]

    q_mean, info = polar_factor(sum(w * q.float() for w, q in zip(weights, qs)),
                                method=polar_method)
    if info["det"] is not None and info["det"] < 0:
        log.warning("%s: v3 merged direction has det<0 — det=-1 component mix", name)
    h_mean = sum(w * h for w, h in zip(weights, hs))
    delta = q_mean.float() @ h_mean if side == "right" else h_mean @ q_mean.float()
    return (base.float() + delta).to(base.dtype)


# --------------------------------------------------------------------------- #
# Registry + streaming driver
# --------------------------------------------------------------------------- #

OP_REGISTRY: Dict[str, Callable] = {
    "plain_avg": op_plain_avg,
    "ta": op_ta,
    "ties": op_ties,
    "v2_ord": op_v2_ord,
    "v2_ord_noxc": lambda *a, **k: op_v2_ord(*a, xc=False, **k),
    "v2_dir_only": op_v2_dir_only,
    "v2_mag_only": op_v2_mag_only,
    "v3_polar": op_v3_polar,
}


def merge_models(base_get: Callable[[str], torch.Tensor],
                 branch_gets: List[Callable[[str], torch.Tensor]],
                 names: List[str],
                 op: str,
                 weights: Sequence[float] | None = None,
                 **op_kwargs) -> Dict[str, torch.Tensor]:
    """Stream parameter names through the chosen operator.

    base_get / branch_gets are callables name -> tensor (see ckpt_io.TensorSource)
    so only one parameter's tensors are resident at a time.
    """
    if op not in OP_REGISTRY:
        raise KeyError(f"unknown op {op}; choose from {sorted(OP_REGISTRY)}")
    fn = OP_REGISTRY[op]
    weights = list(weights) if weights is not None else [1.0] * len(branch_gets)
    if len(weights) != len(branch_gets):
        raise ValueError("weights length must match number of branches")
    merged: Dict[str, torch.Tensor] = {}
    for name in names:
        base = base_get(name)
        branches = [g(name) for g in branch_gets]
        merged[name] = fn(name, base, branches, weights, **op_kwargs)
    return merged
