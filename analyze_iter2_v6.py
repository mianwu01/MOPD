"""Analyze 5-teacher Iter2 baseline trajectory from training log.

Outputs:
  - logs/iter2_v6_random_summary.md  — markdown table for paper
  - logs/iter2_v6_random_summary.csv — raw data for plotting
"""
import re, json, os
from pathlib import Path

LOG = "/home/ubuntu/MOPD/logs/iter2_v6_random.log"
OUT_MD = "/home/ubuntu/MOPD/logs/iter2_v6_random_summary.md"
OUT_CSV = "/home/ubuntu/MOPD/logs/iter2_v6_random_summary.csv"

DATASETS = ["math", "aime24", "aime25", "medqa", "code_humaneval", "tool_star", "search_nqhq"]
METRICS = ["mean@4", "best@4"]


def parse_step_metrics(text):
    """Find each step boundary and the val numbers (if any) emitted at that step."""
    pat = re.compile(r"step:(\d+) - .*?training/global_step:(\d+)", re.DOTALL)
    steps = {}
    for ds in DATASETS:
        for m in METRICS:
            pat_m = re.compile(rf"val-(?:core|aux)/{ds}/(?:reward|acc)/{m}:np\.float64\(([0-9.]+)\)")
            # collect values per occurrence (one per val event)
            values = pat_m.findall(text)
            steps.setdefault(ds, {}).setdefault(m, values)
    return steps


def find_val_steps(text):
    """At which step did val happen? Look for `step:X` blocks containing `val-aux/math`."""
    blocks = re.findall(r"step:(\d+) - .*?(?=step:\d+ - |\Z)", text, re.DOTALL)
    val_steps = []
    for blk in blocks:
        m_step = re.search(r"step:(\d+)", blk[:30])
        if m_step and "val-aux/math/reward/mean@4" in blk[:50000]:
            val_steps.append(int(m_step.group(1)))
    return val_steps


def main():
    text = Path(LOG).read_text()

    # Find which steps had val events
    blocks = re.split(r"step:(\d+) - ", text)
    val_steps = []
    val_values = {ds: {m: [] for m in METRICS} for ds in DATASETS}
    # iterate blocks: blocks[0] = preamble, then alternating step_num, content
    for i in range(1, len(blocks), 2):
        step_num = int(blocks[i])
        content = blocks[i + 1] if i + 1 < len(blocks) else ""
        if "val-aux/math/reward/mean@4" in content[:30000]:
            val_steps.append(step_num)
            for ds in DATASETS:
                for m in METRICS:
                    pat = re.compile(rf"val-(?:core|aux)/{ds}/(?:reward|acc)/{m}:np\.float64\(([0-9.]+)\)")
                    matches = pat.findall(content)
                    val_values[ds][m].append(float(matches[0]) if matches else None)

    # Step 0 baseline is at step 0 (val_before_train) — special: it precedes the first training step
    # Look for early val (before any step:1)
    pre_train = blocks[0]
    if "val-aux/math/reward/mean@4" in pre_train[:30000]:
        val_steps = [0] + val_steps
        for ds in DATASETS:
            for m in METRICS:
                pat = re.compile(rf"val-(?:core|aux)/{ds}/(?:reward|acc)/{m}:np\.float64\(([0-9.]+)\)")
                matches = pat.findall(pre_train)
                val_values[ds][m] = [float(matches[0]) if matches else None] + val_values[ds][m]

    # Write CSV
    with open(OUT_CSV, "w") as f:
        f.write("step," + ",".join(f"{ds}_mean@4,{ds}_best@4" for ds in DATASETS) + "\n")
        for i, s in enumerate(val_steps):
            row = [str(s)]
            for ds in DATASETS:
                for m in METRICS:
                    v = val_values[ds][m][i] if i < len(val_values[ds][m]) else None
                    row.append(f"{v:.4f}" if v is not None else "")
            f.write(",".join(row) + "\n")
    print(f"saved {OUT_CSV}")

    # Write MD
    with open(OUT_MD, "w") as f:
        f.write("# Iter2 5-teacher × 5-domain baseline (v6 random sampler bs=320 ablation)\n\n")
        f.write(f"Run: 80 steps, stratified sampler, all 5 teachers active.\n")
        f.write(f"Wandb: meanwork/OPSD-multi-teacher-baseline/runs/x9tnzd5t\n\n")

        # mean@4 trajectory table
        f.write("## mean@4 trajectory\n\n")
        f.write("| dataset | " + " | ".join(f"step {s}" for s in val_steps) + " | Δ vs step 0 | peak |\n")
        f.write("|" + "|".join(["---"] * (len(val_steps) + 3)) + "|\n")
        for ds in DATASETS:
            vals = val_values[ds]["mean@4"]
            row = [ds]
            for v in vals:
                row.append(f"{v:.4f}" if v is not None else "—")
            base = vals[0] if vals[0] is not None else 0
            last = vals[-1] if vals[-1] is not None else 0
            delta = last - base
            valid_vals = [v for v in vals if v is not None]
            peak_step = val_steps[vals.index(max(valid_vals))] if valid_vals else "—"
            row.append(f"{delta:+.4f}")
            row.append(str(peak_step))
            f.write("| " + " | ".join(row) + " |\n")

        # best@4 trajectory
        f.write("\n## best@4 trajectory (sample coverage)\n\n")
        f.write("| dataset | " + " | ".join(f"step {s}" for s in val_steps) + " |\n")
        f.write("|" + "|".join(["---"] * (len(val_steps) + 1)) + "|\n")
        for ds in DATASETS:
            vals = val_values[ds]["best@4"]
            row = [ds]
            for v in vals:
                row.append(f"{v:.4f}" if v is not None else "—")
            f.write("| " + " | ".join(row) + " |\n")

        # Δ summary
        f.write("\n## Summary: net improvement\n\n")
        f.write("| domain | step 0 | step 80 final | Δ (mean@4) | comment |\n|---|---|---|---|---|\n")
        comments = {
            "math": "MATH-500 (500 problems)",
            "aime24": "30 problems × n=4 — binomial noise dominates",
            "aime25": "30 problems × n=4 — binomial noise dominates",
            "medqa": "MedQA-USMLE test (~750 problems × MCQ)",
            "code_humaneval": "HumanEval+ (164 problems, real sandbox)",
            "tool_star": "Tool-Star held-out 500 problems",
            "search_nqhq": "7 SearchR1 benchmarks (NQ/HotpotQA/2Wiki/MuSiQue/Bamboogle/PopQA/TriviaQA, 100 each)",
        }
        for ds in DATASETS:
            vals = val_values[ds]["mean@4"]
            base = vals[0] if vals and vals[0] is not None else 0
            last = vals[-1] if vals and vals[-1] is not None else 0
            delta = last - base
            f.write(f"| {ds} | {base:.4f} | {last:.4f} | {delta:+.4f} | {comments.get(ds, '')} |\n")

    print(f"saved {OUT_MD}")
    print(f"\n=== val steps captured: {val_steps}")
    print(f"=== preview ===")
    print(Path(OUT_MD).read_text())


if __name__ == "__main__":
    main()
