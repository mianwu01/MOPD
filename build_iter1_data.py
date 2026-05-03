"""
Build Stage 2 Iter1 mixed dataset: math_dapo + medqa.

Outputs to /home/ubuntu/OPSD_OnPolicyDistillation/data/iter1_mixed/:
  train.parquet         — shuffled mix of math + medical (size capped per side)
  val_math500.parquet   — copy from existing math eval
  val_aime24.parquet    — copy
  val_aime25.parquet    — copy
  val_medqa.parquet     — MedQA dev split (1272 q)
"""
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTHONNOUSERSITE", "1")

import datasets
import pandas as pd

OPSD_DATA = Path("/home/ubuntu/OPSD_OnPolicyDistillation/data")
OUT = OPSD_DATA / "iter1_mixed"
OUT.mkdir(parents=True, exist_ok=True)

BOXED_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
MED_INSTRUCTION = "Let's think step by step and output the final answer (A, B, C, or D) within \\boxed{}."


def build_medqa_row(d, split: str, idx: int):
    opts = d["Options"]
    options_text = "\n".join(f"({k}) {v}" for k, v in sorted(opts.items()))
    question = d["Question"].rstrip()
    user_msg = f"{question}\n\n{options_text}\n\n{MED_INSTRUCTION}"
    return {
        "data_source": "medqa",
        "prompt": [{"role": "user", "content": user_msg}],
        "ability": "medical",
        "reward_model": {"style": "rule", "ground_truth": d["Correct Option"]},
        "extra_info": {"split": split, "index": idx, "subject": d.get("Subject Name", "")},
    }


def main():
    # 1. Math train: reuse the existing prepared math train (already in OPSD format).
    math_train = pd.read_parquet(OPSD_DATA / "grpo_processed" / "train.parquet")
    print(f"math_dapo train rows: {len(math_train)}  cols: {list(math_train.columns)}")

    # 2. Medical train: convert from raw MedQA train
    med_train_raw = pd.read_parquet("/tmp/medqa_full/data/train-00000-of-00001.parquet")
    print(f"medqa train rows: {len(med_train_raw)}")

    med_rows = [build_medqa_row(r["data"], "train", i) for i, r in med_train_raw.iterrows()]
    med_df = pd.DataFrame(med_rows)
    print(f"medqa converted: {len(med_df)} rows  cols: {list(med_df.columns)}")

    # 3. Cap each domain at min(math, medqa) for balanced 1:1 mix in Iter1.
    cap = min(len(math_train), len(med_df))
    print(f"per-domain cap: {cap} (mixing 1:1)")
    rng = random.Random(42)

    math_idx = rng.sample(range(len(math_train)), cap)
    med_idx = rng.sample(range(len(med_df)), cap)
    math_sub = math_train.iloc[math_idx].reset_index(drop=True)
    med_sub = med_df.iloc[med_idx].reset_index(drop=True)

    mixed = pd.concat([math_sub, med_sub], ignore_index=True)
    # Shuffle so sort_by_data_source has work to do at trainer side.
    mixed = mixed.sample(frac=1, random_state=42).reset_index(drop=True)
    mixed.to_parquet(OUT / "train.parquet")
    print(f"mixed train -> {OUT / 'train.parquet'}  ({len(mixed)} rows; "
          f"math={mixed['data_source'].eq('math_dapo').sum()}, "
          f"medqa={mixed['data_source'].eq('medqa').sum()})")

    # 4. Val sets — math three from existing, medqa from dev split.
    for name in ("val_math500.parquet", "val_aime24.parquet", "val_aime25.parquet"):
        src = OPSD_DATA / "grpo_processed" / name
        dst = OUT / name
        os.system(f"cp -v {src} {dst}")

    med_dev = pd.read_parquet("/tmp/medqa_full/data/dev-00000-of-00001.parquet")
    med_val_rows = [build_medqa_row(r["data"], "val", i) for i, r in med_dev.iterrows()]
    pd.DataFrame(med_val_rows).to_parquet(OUT / "val_medqa.parquet")
    print(f"val_medqa -> {OUT / 'val_medqa.parquet'}  ({len(med_dev)} rows)")

    print("\n=== final layout ===")
    for f in sorted(OUT.iterdir()):
        n = len(pd.read_parquet(f))
        print(f"  {f.name}: {n} rows")


if __name__ == "__main__":
    main()
