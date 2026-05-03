"""Hold-out val from Tool-Star-SFT-54K: 500 rows NOT in our training sample.

We seeded train sample with rng=42. Use rng=99 for val; intersection with train
is filtered out by index.
"""
import os, re, random, json
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import pandas as pd
from datasets import load_dataset

OUT = "/home/ubuntu/OPSD_OnPolicyDistillation/data/5domain_mixed/val_tool.parquet"
N_VAL = 500
TRAIN_SEED = 42
TRAIN_SIZE = 4000  # match build_5domain_mixed.py PER_DOMAIN
VAL_SEED = 99

ANS_RE = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.DOTALL)

ds = load_dataset("dongguanting/Tool-Star-SFT-54K", split="train")
n_total = len(ds)
print(f"available: {n_total}")

# Reproduce train sample to identify training indices to EXCLUDE from val
all_idx = list(range(n_total))
random.Random(TRAIN_SEED).shuffle(all_idx)
train_used = set(all_idx[:TRAIN_SIZE])
print(f"train indices used: {len(train_used)}")

# Sample val from disjoint set
free = [i for i in range(n_total) if i not in train_used]
random.Random(VAL_SEED).shuffle(free)

rows = []
for i in free:
    if len(rows) >= N_VAL:
        break
    ex = ds[int(i)]
    out = ex.get("output") or ""
    m = ANS_RE.search(out)
    if not m:
        continue
    gt = m.group(1).strip()
    if not gt or len(gt) > 200:  # filter overly long answers
        continue
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    msgs = []
    if instr:
        msgs.append({"role": "system", "content": instr})
    msgs.append({"role": "user", "content": inp})
    rows.append({
        "data_source": "tool_star",
        "prompt": msgs,
        "ability": "tool",
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {"data_source": "tool_star", "index": int(i), "split": "test", "task_id": f"toolstar_val_{i}"},
    })

df = pd.DataFrame(rows)
df.to_parquet(OUT)
print(f"saved {len(df)} rows -> {OUT}")
