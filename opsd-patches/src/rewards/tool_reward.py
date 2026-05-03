"""Tool/TIR reward for Tool-Star training data.

Strategy:
  1. Extract <answer>...</answer> from model response
  2. Normalize: strip whitespace, strip \\boxed{} if present
  3. EM match vs ground_truth (also normalized) → 1.0
  4. Substring fallback → 0.5
  5. Optional: execute <python>...</python> blocks and compare result (advanced; off by default)

Used both at train-time accuracy reporting and at val-time eval.
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandbox import run_python

ANS_RE = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.DOTALL)
PY_RE = re.compile(r"<python>\s*(.+?)\s*</python>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def _strip_boxed(s: str) -> str:
    m = BOXED_RE.search(s)
    return m.group(1) if m else s


def _norm(s: str) -> str:
    if not s:
        return ""
    s = _strip_boxed(s.strip())
    return " ".join(s.lower().split())


def _extract_answer(text: str) -> str:
    if not text:
        return ""
    m = ANS_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: last \boxed{} in the response
    boxed_matches = BOXED_RE.findall(text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    return ""


def compute_score(solution_str, ground_truth, **kwargs):
    if not solution_str:
        return 0.0
    pred_raw = _extract_answer(solution_str)
    if not pred_raw:
        return 0.0
    pred = _norm(pred_raw)
    gt = _norm(str(ground_truth or ""))
    if not gt:
        # No GT to compare — give structural credit only
        return 0.5
    if pred == gt:
        return 1.0
    # Numeric match before substring (handles "8" vs "8.0" vs "$8$" etc.)
    try:
        if float(pred) == float(gt):
            return 1.0
    except Exception:
        pass
    if gt in pred or pred in gt:
        return 0.5
    return 0.0
