"""Quick eval of Qwen2.5-Math-7B-Instruct on MATH-500/AIME24/AIME25 val sets.

Sanity: does the teacher itself score well on our test format? If teacher
≈ student baseline, distillation can't help.
"""
import os, sys, json, argparse
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
sys.path.insert(0, "/home/ubuntu/OPSD_OnPolicyDistillation/src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/ubuntu/models/qwen2.5/teacher-math-math")
    ap.add_argument("--out_json", default="/home/ubuntu/MOPD/logs/teacher_math_eval.json")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    import pandas as pd, numpy as np
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from rewards.math_reward import compute_score

    val_files = {
        "math500": "/home/ubuntu/OPSD_OnPolicyDistillation/data/iter1_mixed/val_math500.parquet",
        "aime24":  "/home/ubuntu/OPSD_OnPolicyDistillation/data/iter1_mixed/val_aime24.parquet",
        "aime25":  "/home/ubuntu/OPSD_OnPolicyDistillation/data/iter1_mixed/val_aime25.parquet",
    }

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=False)
    params = SamplingParams(n=args.n, temperature=0.7, top_p=0.8, top_k=20, max_tokens=2048)

    results = {}
    for name, fp in val_files.items():
        print(f"\n=== {name} ===", flush=True)
        df = pd.read_parquet(fp)
        prompts = []
        gts = []
        for _, r in df.iterrows():
            chat = list(r["prompt"])
            text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
            gts.append(r["reward_model"]["ground_truth"])

        outs = llm.generate(prompts, params)
        scores = []
        for o, gt in zip(outs, gts):
            sc = [compute_score(c.text, gt) for c in o.outputs]
            scores.append(sc)
        S = np.array(scores)
        mean_at_n = float(S.mean(axis=1).mean())
        best_at_n = float(S.max(axis=1).mean())
        results[name] = {"mean@n": mean_at_n, "best@n": best_at_n, "n": args.n, "n_problems": len(df)}
        print(f"  mean@{args.n}={mean_at_n:.4f}  best@{args.n}={best_at_n:.4f}  ({len(df)} problems)", flush=True)

    print("\n=== SUMMARY (teacher Qwen2.5-Math-7B-Instruct) ===")
    for k, v in results.items():
        print(f"  {k:>10}: mean@{args.n}={v['mean@n']:.4f}  best@{args.n}={v['best@n']:.4f}")

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.out_json}")


if __name__ == "__main__":
    main()
