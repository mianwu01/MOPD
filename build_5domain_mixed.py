"""Build 5-domain mixed parquet for OPSD Iter2.

Layout (5 domains × 4K rows = 20K total):
  math_dapo        : Qwen2.5-Math    ← BytedTsinghua-SIA/DAPO-Math-17k-dedup (existing)
  medical_o1       : HuatuoGPT-o1   ← FreedomIntelligence/medical-o1-reasoning-SFT [en]
  code_kodcode     : Qwen2.5-Coder  ← KodCode/KodCode-V1
  tool_star        : Tool-Light     ← dongguanting/Tool-Star-SFT-54K
  search_nqhq      : SearchR1-v0.3  ← PeterJinGo/nq_hotpotqa_train (already verl format)

Reward functions follow at val time. Vanilla OPD doesn't depend on rewards
during training — they're for accuracy metrics + future Stage 3 OPD+RL.
"""
import os, sys, re, random
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import pandas as pd
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("/home/ubuntu/OPSD_OnPolicyDistillation/data/5domain_mixed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_DOMAIN = 4000
SEED = 42
rng = random.Random(SEED)

BOXED = "\n\nLet's think step by step and output the final answer within \\boxed{}."


def to_verl(data_source, prompt_msgs, ability, ground_truth, idx, extra=None):
    return {
        "data_source": data_source,
        "prompt": prompt_msgs,
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"data_source": data_source, "index": int(idx), "split": "train", **(extra or {})},
    }


# ---------- 1. math_dapo (reuse existing iter1_mixed) ----------
def build_math():
    print("=== math_dapo ===", flush=True)
    src = pd.read_parquet("/home/ubuntu/OPSD_OnPolicyDistillation/data/iter1_mixed/train.parquet")
    src = src[src["data_source"] == "math_dapo"].reset_index(drop=True)
    print(f"  available: {len(src)}", flush=True)
    sample = src.sample(n=min(PER_DOMAIN, len(src)), random_state=SEED).reset_index(drop=True)
    rows = []
    for i, r in sample.iterrows():
        rows.append({**r.to_dict(), "extra_info": {**r["extra_info"], "index": int(i), "split": "train"}})
    print(f"  → {len(rows)} rows", flush=True)
    return rows


# ---------- 2. medical_o1 ----------
def build_medical():
    print("=== medical_o1 ===", flush=True)
    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    print(f"  available: {len(ds)}", flush=True)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    take = idx[:PER_DOMAIN]
    rows = []
    for j, i in enumerate(take):
        ex = ds[int(i)]
        q = (ex.get("Question") or "").strip()
        resp = (ex.get("Response") or "").strip()
        if not q or not resp:
            continue
        # Use Question as user prompt; let HuatuoGPT-o1's "## Thinking" come naturally
        rows.append(to_verl(
            data_source="medical_o1",
            prompt_msgs=[{"role": "user", "content": q}],
            ability="medical",
            ground_truth=resp,  # full response; not used as exact-match reward, but kept for record
            idx=j,
        ))
    print(f"  → {len(rows)} rows", flush=True)
    return rows


# ---------- 3. code_kodcode ----------
def build_code():
    print("=== code_kodcode ===", flush=True)
    ds = load_dataset("KodCode/KodCode-V1", split="train")
    print(f"  available: {len(ds)}", flush=True)
    # Filter: keep "instruct" style + has tests (verifiable)
    keep_idx = [i for i in range(len(ds)) if ds[i]["style"] == "instruct" and ds[i].get("test")]
    print(f"  with style=instruct + test: {len(keep_idx)}", flush=True)
    rng.shuffle(keep_idx)
    take = keep_idx[:PER_DOMAIN]
    rows = []
    for j, i in enumerate(take):
        ex = ds[int(i)]
        q = ex["question"].strip()
        sol = ex.get("solution", "").strip()
        rows.append(to_verl(
            data_source="code_kodcode",
            prompt_msgs=[{"role": "user", "content": q}],
            ability="code",
            ground_truth=sol,  # reference solution; not used as exact-match reward
            idx=j,
            extra={"test": ex.get("test", ""), "subset": ex.get("subset")},
        ))
    print(f"  → {len(rows)} rows", flush=True)
    return rows


# ---------- 4. tool_star ----------
def build_tool():
    print("=== tool_star ===", flush=True)
    ds = load_dataset("dongguanting/Tool-Star-SFT-54K", split="train")
    print(f"  available: {len(ds)}", flush=True)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    take = idx[:PER_DOMAIN]
    ans_re = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.DOTALL)
    rows = []
    for j, i in enumerate(take):
        ex = ds[int(i)]
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()
        if not inp:
            continue
        m = ans_re.search(out)
        gt = m.group(1) if m else ""
        msgs = []
        if instr:
            msgs.append({"role": "system", "content": instr})
        msgs.append({"role": "user", "content": inp})
        rows.append(to_verl(
            data_source="tool_star",
            prompt_msgs=msgs,
            ability="tool",
            ground_truth=gt,
            idx=j,
        ))
    print(f"  → {len(rows)} rows", flush=True)
    return rows


# ---------- 5. search_nqhq (already verl format; use pandas direct to avoid datasets-lib schema bug) ----------
def build_search():
    print("=== search_nqhq ===", flush=True)
    src = pd.read_parquet("/home/ubuntu/.cache/huggingface/hub/datasets--PeterJinGo--nq_hotpotqa_train/snapshots/b7d80abfee334a7a91cb377544f09180d58b34f6/train.parquet")
    print(f"  available: {len(src)}", flush=True)
    sample = src.sample(n=min(PER_DOMAIN, len(src)), random_state=SEED).reset_index(drop=True)
    rows = []
    for j, r in sample.iterrows():
        prompt_msgs = list(r["prompt"])
        # nq_hotpotqa stores ground_truth = {target: ndarray([...])}; flatten to first string.
        rm = r["reward_model"]
        gt_raw = rm["ground_truth"] if isinstance(rm, dict) else {}
        if isinstance(gt_raw, dict) and "target" in gt_raw:
            tgt = gt_raw["target"]
            try:
                gt_str = str(tgt[0]) if len(tgt) > 0 else ""
            except Exception:
                gt_str = str(tgt)
        else:
            gt_str = str(gt_raw)
        rows.append({
            "data_source": "search_nqhq",
            "prompt": prompt_msgs,
            "ability": "search",
            "reward_model": {"style": "rule", "ground_truth": gt_str},
            "extra_info": {"data_source": "search_nqhq", "index": int(j), "split": "train",
                           "original_source": str(r.get("data_source", ""))},
        })
    print(f"  → {len(rows)} rows", flush=True)
    return rows


def main():
    parts = []
    for fn in (build_math, build_medical, build_code, build_tool, build_search):
        parts.extend(fn())
    rng.shuffle(parts)
    df = pd.DataFrame(parts)
    out = OUT_DIR / "train.parquet"
    df.to_parquet(out)
    print(f"\n=== SUMMARY ===")
    print(f"total rows: {len(df)}")
    print(f"by data_source:\n{df['data_source'].value_counts()}")
    print(f"saved -> {out}")

    # copy val files
    src_dir = Path("/home/ubuntu/OPSD_OnPolicyDistillation/data/iter1_mixed")
    for f in ("val_math500.parquet", "val_aime24.parquet", "val_aime25.parquet", "val_medqa.parquet"):
        dst = OUT_DIR / f
        if not dst.exists():
            os.system(f"cp -v {src_dir/f} {dst}")
    print(f"\nval files copied from {src_dir}")


if __name__ == "__main__":
    main()
