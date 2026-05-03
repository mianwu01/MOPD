"""Quick MATH-500 eval for merged step_156 ckpt. Mirrors verl val config: temp=0.7, top_p=0.8, top_k=20, n=4, max_resp=2048."""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import sys, json
from pathlib import Path

MODEL = "/home/ubuntu/openr1_outputs/qwen25_7b_grpo_step156_hf"
VAL = "/home/ubuntu/OPSD_OnPolicyDistillation/data/openr1_math_grpo/val_math500.parquet"
SRC = "/home/ubuntu/OPSD_OnPolicyDistillation/src"
OUT = "/home/ubuntu/openr1_outputs/eval_step156_math500.json"

sys.path.insert(0, SRC)


def main():
    import pandas as pd, numpy as np
    from rewards.math_reward import compute_score
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    df = pd.read_parquet(VAL)
    print(f"loaded {len(df)} val rows", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts, gts = [], []
    for _, r in df.iterrows():
        chat = list(r["prompt"])
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
        gts.append(r["reward_model"]["ground_truth"])

    print(f"first prompt:\n{prompts[0][:500]}", flush=True)

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=False,
    )

    params = SamplingParams(n=4, temperature=0.7, top_p=0.8, top_k=20, max_tokens=2048)
    outs = llm.generate(prompts, params)

    scores = []
    for o, gt in zip(outs, gts):
        sc = [compute_score(c.text, gt) for c in o.outputs]
        scores.append(sc)

    S = np.array(scores)
    mean_at_4 = float(S.mean(axis=1).mean())
    best_at_4 = float(S.max(axis=1).mean())

    print(f"\n=== STEP_156 MATH-500 EVAL (n=4) ===", flush=True)
    print(f"mean@4 = {mean_at_4:.4f}", flush=True)
    print(f"best@4 = {best_at_4:.4f}", flush=True)
    print(f"verl reported step 160: mean@4=0.7340, best@4=0.7922", flush=True)

    with open(OUT, "w") as f:
        json.dump({"model": MODEL, "n": 4, "mean@4": mean_at_4, "best@4": best_at_4, "n_problems": len(df)}, f, indent=2)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
