"""Code reward — runs candidate code against unit tests in a sandbox.

For OPD with KodCode/V1: each train row's `extra_info.test` carries pytest tests
that import from `solution`. Returns 1.0 if all tests pass, 0.0 otherwise.

For HumanEval+ val: rows have `extra_info.test` (HumanEval+ test code) and
`extra_info.entry_point`. Same pytest path works.
"""
import re
import sys
import os

# Allow rewards/ to import sandbox
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandbox import run_pytest, run_check_pattern


_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)
_DEF_LINE = re.compile(r"^(def\s+\w+\s*\(.*?\):.*)", re.MULTILINE | re.DOTALL)


def extract_code(text: str) -> str:
    """Extract a Python code block from a model response.

    Order:
      1. last fenced ```python ... ``` block (most likely the final answer)
      2. anything starting from the first `def ` to end of string
      3. raw text (last resort)
    """
    if not text:
        return ""
    blocks = _CODE_BLOCK.findall(text)
    if blocks:
        # Prefer last block (final answer)
        return blocks[-1].strip()
    m = _DEF_LINE.search(text)
    if m:
        return m.group(0)
    return text.strip()


def compute_score(solution_str, ground_truth, **kwargs):
    """Run candidate solution against unit tests; 1.0 pass / 0.0 fail.

    Required: kwargs['extra_info']['test']  — pytest test code as string.
    Optional: kwargs['extra_info']['timeout']  — seconds, default 30.
    """
    if not solution_str:
        return 0.0

    # verl passes data_source as a top-level kwarg, extra_info as a dict
    extra = kwargs.get("extra_info") or {}
    test_code = extra.get("test", "")
    timeout = int(extra.get("timeout", 30))

    if not test_code:
        # No tests available — fall back to structural credit (avoid 0 noise)
        return 0.5 if "def " in solution_str and "return " in solution_str else 0.0

    code = extract_code(solution_str)
    if not code or "def " not in code:
        return 0.0

    # Auto-detect test format
    if "def check(" in test_code:
        # HumanEval+/MBPP+ style: test defines check(candidate); call check(entry_point)
        ep = extra.get("entry_point") or ""
        if not ep:
            return 0.0
        res = run_check_pattern(code, test_code, ep, timeout=timeout)
    else:
        # KodCode style: tests do `from solution import ...`, use pytest
        res = run_pytest(code, test_code, timeout=timeout)
    return 1.0 if res["ok"] else 0.0
