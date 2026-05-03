"""Build val_search.parquet — balanced sample from PeterJinGo/nq_hotpotqa_train test.

Test set spans 7 SearchR1-paper benchmarks: nq, hotpotqa, popqa, triviaqa,
2wikimultihopqa, musique, bamboogle. Sample ~100 each → ~700 rows total.
"""
import os, random
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import pandas as pd

SRC = "/home/ubuntu/.cache/huggingface/hub/datasets--PeterJinGo--nq_hotpotqa_train/snapshots/b7d80abfee334a7a91cb377544f09180d58b34f6/test.parquet"
OUT = "/home/ubuntu/OPSD_OnPolicyDistillation/data/5domain_mixed/val_search.parquet"
PER_SOURCE = 100
SEED = 7

df = pd.read_parquet(SRC)
print(f"loaded {len(df)} rows; sources: {df['data_source'].value_counts().to_dict()}")

rng = random.Random(SEED)
parts = []
for src, g in df.groupby("data_source"):
    take = min(PER_SOURCE, len(g))
    sample = g.sample(n=take, random_state=SEED).reset_index(drop=True)
    print(f"  {src}: take {take}")
    for j, r in sample.iterrows():
        prompt = list(r["prompt"])
        rm = r["reward_model"]
        gt_raw = rm["ground_truth"] if isinstance(rm, dict) else {}
        # Normalize GT to a list of candidate strings, then store the raw structure
        if isinstance(gt_raw, dict) and "target" in gt_raw:
            tgt = gt_raw["target"]
            try:
                gt_list = [str(x) for x in tgt]
            except Exception:
                gt_list = [str(tgt)]
        else:
            gt_list = [str(gt_raw)]
        # Use first target as primary; reward fn will accept list too
        primary = gt_list[0] if gt_list else ""
        parts.append({
            "data_source": "search_nqhq",  # unified routing key matching train
            "prompt": prompt,
            "ability": "search",
            "reward_model": {"style": "rule", "ground_truth": primary, "all_targets": gt_list},
            "extra_info": {
                "data_source": "search_nqhq",
                "index": int(j),
                "split": "test",
                "subset": str(src),
                "task_id": f"search_{src}_{j}",
                "all_targets": gt_list,
            },
        })

out_df = pd.DataFrame(parts)
out_df.to_parquet(OUT)
print(f"\nsaved {len(out_df)} rows -> {OUT}")
print(f"by subset:\n{out_df['extra_info'].apply(lambda x: x['subset']).value_counts()}")
