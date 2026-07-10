"""mopd_merge — geometry-aware merging of per-domain OPD branches.

Implements the decomposition recipe for the MOPD orthogonal-aggregation project
(see NOTES_independent_assessment.md, Addendum 2):

  per-domain update  →  orthogonal "direction" (polar factor / Procrustes rotation)
                     +  "magnitude" (residual / SPD factor / scalar)
  directions merged on the manifold (so(d) via Cayley, with OrthoMerge-style ×c)
  magnitudes merged linearly (Task Arithmetic) or with TIES.

All operators are pure PyTorch, CPU-safe, and stream tensor-by-tensor so that
1.5B-scale models merge within modest RAM.
"""

from .linalg import polar_factor, cayley_log, cayley_exp, newton_schulz
from .decompose import ord_decompose, ord_reconstruct
from .operators import (
    merge_models,
    OP_REGISTRY,
)
from .diagnostics import delta_cosine_report, split_half_reliability
from .ckpt_io import TensorSource, save_merged_model

__all__ = [
    "polar_factor",
    "cayley_log",
    "cayley_exp",
    "newton_schulz",
    "ord_decompose",
    "ord_reconstruct",
    "merge_models",
    "OP_REGISTRY",
    "delta_cosine_report",
    "split_half_reliability",
    "TensorSource",
    "save_merged_model",
]
