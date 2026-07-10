"""Core linear-algebra primitives: polar factor, Newton-Schulz, Cayley maps.

Conventions
-----------
- polar factor of A (m x n): the nearest (semi-)orthogonal matrix Q = U @ Vh
  from the thin SVD A = U S Vh. For square A this is the classical "nearest
  orthogonal matrix" (orthogonal Procrustes with identity target). polar() is
  scale-invariant: polar(c*A) == polar(A) for c > 0.
- Cayley transform: R = (I - Q) @ inv(I + Q) for skew-symmetric Q gives
  R in SO(n) (no eigenvalue -1); inverse: Q = (I - R) @ inv(I + R).
  Near-identity rotations (the OFT / ORD regime) are always in the domain.
"""

from __future__ import annotations

import torch

# Muon-style quintic Newton-Schulz coefficients (Jordan et al.); each iteration
# applies p(X) = a*X + b*X@X^T@X + c*(X@X^T)^2@X, driving singular values to 1.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def _as_float(a: torch.Tensor) -> torch.Tensor:
    return a.float() if a.dtype not in (torch.float32, torch.float64) else a


def polar_factor(a: torch.Tensor, method: str = "svd", ns_iters: int = 8):
    """Nearest (semi-)orthogonal matrix to `a`, plus diagnostics.

    Returns (q, info) where info = {"det": float|None, "dist_from_eye": float|None}.
    det / dist_from_eye are only defined for square inputs (None otherwise).
    """
    if a.ndim != 2:
        raise ValueError(f"polar_factor expects a matrix, got shape {tuple(a.shape)}")
    a32 = _as_float(a)
    if method == "svd":
        u, _, vh = torch.linalg.svd(a32, full_matrices=False)
        q = u @ vh
    elif method == "ns":
        q = newton_schulz(a32, iters=ns_iters)
    else:
        raise ValueError(f"unknown polar method: {method}")

    info = {"det": None, "dist_from_eye": None}
    if q.shape[0] == q.shape[1]:
        info["det"] = float(torch.linalg.det(q))
        eye = torch.eye(q.shape[0], dtype=q.dtype, device=q.device)
        info["dist_from_eye"] = float(torch.linalg.matrix_norm(q - eye))
    return q.to(a.dtype), info


def newton_schulz(g: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """Muon-style iterative orthogonalization; converges to polar factor of g."""
    a, b, c = _NS_COEFFS
    x = _as_float(g)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (torch.linalg.matrix_norm(x) + 1e-12)
    for _ in range(iters):
        s = x @ x.T
        x = a * x + (b * s + c * (s @ s)) @ x
    if transposed:
        x = x.T
    return x


def polar_decompose(a: torch.Tensor, method: str = "svd"):
    """Full polar decomposition a = Q @ H with Q (semi-)orthogonal, H symmetric PSD.

    For m x n with m >= n: Q is m x n (Stiefel), H is n x n SPD.
    For m < n we decompose a = H @ Q instead (left polar) and return
    (q, h, side) with side in {"right", "left"} such that:
      side == "right": a ≈ q @ h;  side == "left": a ≈ h @ q.
    """
    if a.shape[0] >= a.shape[1]:
        q, _ = polar_factor(a, method=method)
        h = q.T.to(torch.float32) @ _as_float(a)
        h = 0.5 * (h + h.T)  # symmetrize numerical noise
        return q, h.to(a.dtype), "right"
    q, _ = polar_factor(a, method=method)
    h = _as_float(a) @ q.T.to(torch.float32)
    h = 0.5 * (h + h.T)
    return q, h.to(a.dtype), "left"


def cayley_log(r: torch.Tensor, atol: float = 1e-4):
    """Inverse Cayley: skew Q with R = (I - Q) inv(I + Q). Requires R near SO(n).

    Returns (q, ok). ok=False when R is outside the Cayley domain (det < 0 or
    I + R ill-conditioned) — callers must handle (this is the V3 landmine).
    """
    if r.shape[0] != r.shape[1]:
        raise ValueError("cayley_log needs a square matrix")
    r32 = _as_float(r)
    n = r32.shape[0]
    eye = torch.eye(n, dtype=r32.dtype, device=r32.device)
    det = torch.linalg.det(r32)
    if det <= 0:
        return torch.zeros_like(r32), False
    ipr = eye + r32
    # Guard: eigenvalue of R near -1 makes I + R singular.
    if torch.linalg.matrix_norm(ipr, ord=-2) < atol:
        return torch.zeros_like(r32), False
    q = torch.linalg.solve(ipr.T, (eye - r32).T).T  # (I - R) @ inv(I + R)
    q = 0.5 * (q - q.T)  # project to skew (kills numerical asymmetry)
    return q, True


def cayley_exp(q: torch.Tensor) -> torch.Tensor:
    """Cayley map: skew Q -> rotation R = (I - Q) inv(I + Q) in SO(n)."""
    q32 = _as_float(q)
    n = q32.shape[0]
    eye = torch.eye(n, dtype=q32.dtype, device=q32.device)
    r = torch.linalg.solve((eye + q32).T, (eye - q32).T).T
    return r
