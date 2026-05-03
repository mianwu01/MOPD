"""Build val_humaneval.parquet from evalplus/humanevalplus (164 problems).

Verl row format:
  data_source: code_humaneval
  prompt: [{role: user, content: "<problem prompt + 'Complete the function:'>"}]
  ability: code
  reward_model: {style: rule, ground_truth: <entry_point>}
  extra_info: {test: <test code>, entry_point: <fn name>, task_id: ...}
"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import pandas as pd
from datasets import load_dataset

OUT = "/home/ubuntu/OPSD_OnPolicyDistillation/data/5domain_mixed/val_humaneval.parquet"

ds = load_dataset("evalplus/humanevalplus", split="test")
print(f"loaded {len(ds)} problems")

rows = []
for i, ex in enumerate(ds):
    # The prompt is the function signature + docstring.
    # Standard HumanEval evaluation passes this directly to the model.
    user_content = (
        ex["prompt"].rstrip()
        + "\n\nComplete the function above. Return only the full Python code in a markdown code block."
    )
    rows.append({
        "data_source": "code_humaneval",
        "prompt": [{"role": "user", "content": user_content}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": ex["entry_point"]},
        "extra_info": {
            "data_source": "code_humaneval",
            "index": i,
            "split": "test",
            "task_id": ex["task_id"],
            "entry_point": ex["entry_point"],
            "test": ex["test"],
            # canonical_solution kept for sanity verification, not used at val time
            "canonical_solution_full": ex["prompt"] + ex["canonical_solution"],
        },
    })

df = pd.DataFrame(rows)
df.to_parquet(OUT)
print(f"saved {len(df)} rows -> {OUT}")
print(f"cols: {df.columns.tolist()}")
