"""Orthogonal-Residual Decoupling (ORD) of a finetuned weight against its base.

Given branch weight W_d and base W0 (same shape), find the nearest rotation
carrying W0 toward W_d, plus a residual:

    W_d = R_d @ W0 + E_d        (side = "left",  R is out x out)
    W_d = W0 @ R_d + E_d        (side = "right", R is in  x in )

R_d solves the orthogonal Procrustes problem on that side:
    left:  R_d = polar(W_d @ W0.T)
    right: R_d = polar(W0.T @ W_d)

side="auto" picks the smaller square dimension (e.g. hidden side for
lm_head / embeddings / MLP up-projections), which keeps R tractable.
For nearby checkpoints R_d is near identity with det=+1, i.e. inside the
Cayley domain required by so(d) averaging.
"""

from __future__ import annotations

import torch

from .linalg import polar_factor


def _pick_side(shape) -> str:
    out_dim, in_dim = shape
    return "left" if out_dim <= in_dim else "right"


def ord_decompose(w_d: torch.Tensor, w0: torch.Tensor, side: str = "auto",
                  method: str = "svd"):
    """Decompose w_d = R @ w0 + E (left) or w0 @ R + E (right).

    Returns (r, e, side, info) with info carrying det / dist_from_eye of R.
    """
    if w_d.shape != w0.shape or w_d.ndim != 2:
        raise ValueError("ord_decompose needs two matrices of identical shape")
    if side == "auto":
        side = _pick_side(w_d.shape)
    w_d32, w032 = w_d.float(), w0.float()
    if side == "left":
        r, info = polar_factor(w_d32 @ w032.T, method=method)
        e = w_d32 - r @ w032
    elif side == "right":
        r, info = polar_factor(w032.T @ w_d32, method=method)
        e = w_d32 - w032 @ r
    else:
        raise ValueError(f"side must be left/right/auto, got {side}")
    return r, e.to(w_d.dtype), side, info


def ord_reconstruct(r: torch.Tensor, e: torch.Tensor, w0: torch.Tensor,
                    side: str) -> torch.Tensor:
    """Inverse of ord_decompose: rebuild the weight from (R, E, W0)."""
    w032 = w0.float()
    if side == "left":
        w = r.float() @ w032 + e.float()
    elif side == "right":
        w = w032 @ r.float() + e.float()
    else:
        raise ValueError(f"bad side: {side}")
    return w.to(w0.dtype)
