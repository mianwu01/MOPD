"""Medical-o1 reasoning reward placeholder.

Free-form medical reasoning (HuatuoGPT-o1 training distribution) is hard to
auto-grade without a judge model. Give cheap structural credit:
  - 1.0 if response contains "## Final Response" (HuatuoGPT-o1 format) AND
        the GT answer keywords appear in it
  - 0.5 if response contains "## Thinking" or "## Final Response" (structural)
  - 0.0 otherwise
"""
import re


def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


def compute_score(solution_str, ground_truth, **kwargs) -> float:
    if not solution_str:
        return 0.0
    has_final = "## final response" in solution_str.lower() or "final response" in solution_str.lower()
    has_thinking = "## thinking" in solution_str.lower()
    structural = 0.5 if (has_final or has_thinking) else 0.0
    if not ground_truth:
        return structural
    # Try keyword overlap with GT
    gt_words = set(_norm(str(ground_truth)).split())
    sol_words = set(_norm(solution_str).split())
    if not gt_words:
        return structural
    overlap = len(gt_words & sol_words) / max(1, len(gt_words))
    if overlap > 0.5 and has_final:
        return 1.0
    return structural
