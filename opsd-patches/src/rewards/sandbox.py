"""Subprocess-isolated Python execution sandbox for code/tool reward verification.

Design choices:
- subprocess.run with timeout (process-level isolation)
- temp dir as cwd (no host file pollution)
- stdlib + numpy + pandas allowed; network restricted via env (best-effort)
- 30s wall timeout default, 256MB memory cap via resource limits

Not a security boundary against adversarial code — just enough for our model-generated code.
"""
import os
import sys
import subprocess
import tempfile
from typing import Optional

DEFAULT_TIMEOUT = 30
DEFAULT_MEM_MB = 1024  # 1 GB per worker

# Use the same python as the parent process (opsd env)
PYTHON_BIN = sys.executable


def _set_limits(mem_mb: int):
    """Subprocess pre-exec hook — set memory limit + new pgrp."""
    import resource
    # AS = address space (virtual memory)
    bytes_lim = mem_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (bytes_lim, bytes_lim))
    except Exception:
        pass
    # Detach from parent's pgrp so we can kill the whole tree
    os.setpgrp()


def run_python(
    code: str,
    timeout: int = DEFAULT_TIMEOUT,
    mem_mb: int = DEFAULT_MEM_MB,
    stdin_data: str = "",
) -> dict:
    """Run a single Python script in subprocess. Returns {ok, stdout, stderr, exit, msg}."""
    with tempfile.TemporaryDirectory(prefix="opd_sandbox_") as td:
        script = os.path.join(td, "main.py")
        with open(script, "w") as f:
            f.write(code)
        try:
            r = subprocess.run(
                [PYTHON_BIN, script],
                cwd=td,
                timeout=timeout,
                capture_output=True,
                text=True,
                input=stdin_data,
                env={
                    "PATH": "/usr/bin:/bin",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                },
                preexec_fn=lambda: _set_limits(mem_mb),
            )
            return {
                "ok": r.returncode == 0,
                "stdout": (r.stdout or "")[-4000:],
                "stderr": (r.stderr or "")[-2000:],
                "exit": r.returncode,
                "msg": "ok" if r.returncode == 0 else f"exit {r.returncode}",
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": "", "exit": -1, "msg": "TIMEOUT"}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": "", "exit": -1, "msg": f"ERR:{type(e).__name__}:{e}"}


def run_check_pattern(
    solution_code: str,
    test_code: str,
    entry_point: str,
    timeout: int = DEFAULT_TIMEOUT,
    mem_mb: int = DEFAULT_MEM_MB,
) -> dict:
    """Run HumanEval+/MBPP+ style tests where test_code defines `check(candidate)`.

    Combines: solution + test + `check(entry_point)` + print sentinel.
    Returns ok=True only if entry_point is defined AND check() passes without exception.
    """
    sentinel = "_OPD_SANDBOX_OK_8f3a2c1b"
    combined = (
        solution_code
        + "\n\n"
        + test_code
        + f"\n\ncheck({entry_point})\n"
        + f"print('{sentinel}')\n"
    )
    res = run_python(combined, timeout=timeout, mem_mb=mem_mb)
    ok = res["ok"] and sentinel in (res.get("stdout") or "")
    return {"ok": ok, "exit": res["exit"], "msg": res["msg"], "tail": res.get("stderr", "")[-500:]}


def run_pytest(
    solution_code: str,
    test_code: str,
    timeout: int = DEFAULT_TIMEOUT,
    mem_mb: int = DEFAULT_MEM_MB,
) -> dict:
    """Write solution.py + test_solution.py, run pytest, return result.

    KodCode tests do `from solution import xxx`, so the candidate is written
    to `solution.py` and pytest discovers test functions in `test_solution.py`.
    """
    with tempfile.TemporaryDirectory(prefix="opd_pytest_") as td:
        sol_path = os.path.join(td, "solution.py")
        tst_path = os.path.join(td, "test_solution.py")
        with open(sol_path, "w") as f:
            f.write(solution_code)
        with open(tst_path, "w") as f:
            f.write(test_code)
        try:
            r = subprocess.run(
                [PYTHON_BIN, "-m", "pytest", "-x", "-q", "--no-header",
                 "--disable-warnings", "test_solution.py"],
                cwd=td,
                timeout=timeout,
                capture_output=True,
                text=True,
                env={
                    "PATH": "/usr/bin:/bin",
                    "PYTHONPATH": td,
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                },
                preexec_fn=lambda: _set_limits(mem_mb),
            )
            ok = r.returncode == 0
            tail = (r.stdout or "")[-1500:] + "\n" + (r.stderr or "")[-500:]
            return {"ok": ok, "exit": r.returncode, "msg": "passed" if ok else "failed", "tail": tail}
        except subprocess.TimeoutExpired:
            return {"ok": False, "exit": -1, "msg": "TIMEOUT", "tail": ""}
        except Exception as e:
            return {"ok": False, "exit": -1, "msg": f"ERR:{type(e).__name__}:{e}", "tail": ""}
