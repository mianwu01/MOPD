"""Search reward: EM/F1 on <answer>...</answer> against GT (which may be a list of targets).

For SearchR1-style data the GT is the gold final answer. We do NOT validate
intermediate <search> queries — without a retrieval service the student's
<search> calls return empty observations, so we only score the final answer.

Used for:
  - train-time accuracy on search_nqhq
  - val-time eval on HotpotQA / NQ test sets

Paper-grade upgrade: full BM25 over Wikipedia would let us validate the agent loop
end-to-end. Scoped as future work.
"""
import re
import string
import sys

ANS_RE = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.DOTALL)


def _norm(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles + punct."""
    if not s:
        return ""
    s = s.lower().strip()
    # Strip articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Strip punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _f1(pred: str, gt: str) -> float:
    p_tokens = _norm(pred).split()
    g_tokens = _norm(gt).split()
    if not p_tokens or not g_tokens:
        return 0.0
    common = set(p_tokens) & set(g_tokens)
    if not common:
        return 0.0
    # Multi-set intersection size
    p_count = {t: p_tokens.count(t) for t in set(p_tokens)}
    g_count = {t: g_tokens.count(t) for t in set(g_tokens)}
    same = sum(min(p_count.get(t, 0), g_count.get(t, 0)) for t in common)
    if same == 0:
        return 0.0
    precision = same / len(p_tokens)
    recall = same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def _extract_answer(text: str) -> str:
    if not text:
        return ""
    m = ANS_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: last line of response
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def _to_target_list(gt) -> list:
    """Normalize GT into a list of candidate strings."""
    if gt is None:
        return []
    if isinstance(gt, str):
        return [gt]
    if isinstance(gt, dict):
        # nq_hotpotqa format: {target: [list]}
        if "target" in gt:
            t = gt["target"]
            if isinstance(t, (list, tuple)):
                return [str(x) for x in t]
            return [str(t)]
    if isinstance(gt, (list, tuple)):
        return [str(x) for x in gt]
    return [str(gt)]


def compute_score(solution_str, ground_truth, **kwargs):
    """Score response against any of the gold answers. Returns max EM/F1 over targets."""
    if not solution_str:
        return 0.0
    pred = _extract_answer(solution_str)
    if not pred:
        return 0.0
    targets = _to_target_list(ground_truth)
    if not targets:
        return 0.0
    pred_n = _norm(pred)
    best = 0.0
    for t in targets:
        t_n = _norm(t)
        if not t_n:
            continue
        if pred_n == t_n:
            return 1.0
        # Token-level F1 fallback (HotpotQA / NQ standard)
        f1 = _f1(pred, t)
        if f1 > best:
            best = f1
    return best


if __name__ == "__main__":
    # Quick self-test
    print("EM:", compute_score("<answer>Paris</answer>", "Paris"))
    print("EM list:", compute_score("<answer>Paris</answer>", ["paris", "London"]))
    print("F1:", compute_score("<answer>The capital of France is Paris</answer>", "Paris, France"))
    print("Wrong:", compute_score("<answer>London</answer>", "Paris"))
    print("dict GT:", compute_score("<answer>2718</answer>", {"target": ["2,718", "2718"]}))
