"""
Convert open-r1/OpenR1-Math-220k to the verl GRPO data format used by OPSD.
Verl row schema: {data_source, prompt (chat list), ability, reward_model: {style, ground_truth}, extra_info}
Output: data/openr1_math_grpo/{train.parquet, val_math500.parquet, val_aime24.parquet, val_aime25.parquet}
"""
import os, sys
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import datasets, pandas as pd, json, random
from pathlib import Path

ROOT = Path("/home/ubuntu/OPSD_OnPolicyDistillation/data/openr1_math_grpo")
ROOT.mkdir(parents=True, exist_ok=True)
BOXED = "Let's think step by step and output the final answer within \\boxed{}."


def to_verl(problem: str, answer: str, ds: str, idx: int) -> dict:
    user = problem.rstrip() + "\n\n" + BOXED
    return {
        "data_source": ds,
        "prompt": [{"role": "user", "content": user}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {"split": "train" if "train" in ds else "test", "index": idx},
    }


def main(max_train=20000):
    print("=== loading OpenR1-Math-220k (default split) ===", flush=True)
    ds = datasets.load_dataset("open-r1/OpenR1-Math-220k", split="train")
    print(f"raw rows: {len(ds)}, cols: {list(ds.features)}", flush=True)

    # Filter rows with a clean answer + complete reasoning
    keep = [r for r in ds if r.get("answer") and isinstance(r["answer"], str) and r["answer"].strip()]
    print(f"with non-empty answer: {len(keep)}", flush=True)
    rng = random.Random(42)
    rng.shuffle(keep)
    if max_train and len(keep) > max_train:
        keep = keep[:max_train]
        print(f"sub-sampled to {max_train}", flush=True)

    rows = [to_verl(r["problem"], r["answer"], "math", i) for i, r in enumerate(keep)]
    train_df = pd.DataFrame(rows)
    train_path = ROOT / "train.parquet"
    train_df.to_parquet(train_path)
    print(f"train -> {train_path} ({len(train_df)} rows)", flush=True)

    # Reuse iter1 val sets (math500/aime24/aime25 already in verl format)
    iter1 = Path("/home/ubuntu/OPSD_OnPolicyDistillation/data/iter1_mixed")
    for f in ("val_math500.parquet", "val_aime24.parquet", "val_aime25.parquet"):
        os.system(f"cp -v {iter1/f} {ROOT/f}")

    print("\n=== final layout ===")
    for f in sorted(ROOT.iterdir()):
        n = len(pd.read_parquet(f))
        print(f"  {f.name}: {n} rows")


if __name__ == "__main__":
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    main(max_train=cap)
