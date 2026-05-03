"""
Medical MCQ reward for MedQA-USMLE-style 4-option questions.

Extracts the model's predicted letter (A/B/C/D) from the response. Order of
extraction (most strict to lenient):
  1. \\boxed{X}                       — preferred (matches Qwen "boxed" prompt)
  2. "Answer: X" / "answer is X"      — common fallback
  3. Final isolated A/B/C/D in text   — last-resort

Returns 1.0 on exact letter match with ground_truth, else 0.0.
"""
import re
from typing import Optional


_BOXED_RE = re.compile(r"\\boxed\{\s*([A-Za-z])\s*\}")
_ANSWER_RE = re.compile(r"(?:answer\s*(?:is|:)|the\s+correct\s+answer\s+is)\s*[\(\[\*]*\s*([A-Za-z])", re.IGNORECASE)
_FINAL_LETTER_RE = re.compile(r"\b([A-Da-d])\b(?:\s*[\.\)]?\s*)$")


def _extract_letter(text: str) -> Optional[str]:
    if not text:
        return None
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).upper()
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    # Last-line / last-token fallback
    last = text.strip().splitlines()[-1] if text.strip() else ""
    m = _FINAL_LETTER_RE.search(last)
    if m:
        return m.group(1).upper()
    return None


def compute_score(solution_str: str, ground_truth, **kwargs) -> dict:
    gt = str(ground_truth).strip().upper()
    if len(gt) != 1 or gt not in "ABCDE":
        # Defensive: malformed ground truth — score 0 but keep pred for debugging.
        pred = _extract_letter(solution_str) or ""
        return {"score": 0.0, "acc": 0.0, "pred": pred}
    pred = _extract_letter(solution_str)
    correct = (pred is not None and pred == gt)
    return {"score": float(correct), "acc": float(correct), "pred": pred or ""}


if __name__ == "__main__":
    cases = [
        ("Reasoning... \\boxed{D}", "D", 1.0),
        ("After analysis, the answer is C.", "C", 1.0),
        ("...so I think the correct answer is (B).", "B", 1.0),
        ("Final answer:\nA", "A", 1.0),
        ("This is option D not A.\n\nFinal: A", "A", 1.0),
        ("\\boxed{A}", "B", 0.0),
        ("hmm not sure", "B", 0.0),
        ("\\boxed{Z}", "A", 0.0),
    ]
    for sol, gt, exp in cases:
        out = compute_score(sol, gt)
        ok = "✓" if out["score"] == exp else "✗"
        print(f"  {ok}  gt={gt}  pred={out['pred']!r}  score={out['score']}  expected={exp}")
