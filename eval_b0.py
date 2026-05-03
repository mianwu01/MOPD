"""
Stage 1 B0 — raw eval of Qwen2.5-1.5B-Instruct on MATH-500 / AIME24 / AIME25.

Standalone vLLM offline pipeline (no ray, no verl) so we don't conflict with
another in-progress session sharing this machine.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONNOUSERSITE", "1")

# Make OPSD's math_reward importable
sys.path.insert(0, "/home/ubuntu/OPSD_OnPolicyDistillation/src")
from rewards.math_reward import compute_score  # noqa: E402


BOXED_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def load_dataset_aime24(path: str):
    import pandas as pd
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        rows.append({"problem": r["problem"], "answer": str(r.get("solution", r.get("Answer")))})
    return rows


def load_dataset_jsonl(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append({"problem": d["problem"], "answer": str(d["answer"])})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/ubuntu/models/qwen2.5/student-1.5b-instruct")
    p.add_argument("--data-dir", default="/home/ubuntu/OPSD_OnPolicyDistillation/data")
    p.add_argument("--out", default="/home/ubuntu/MOPD/eval_b0_results.json")
    p.add_argument("--n", type=int, default=16, help="samples per prompt (mean@n)")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    p.add_argument("--max-model-len", type=int, default=20480, help="prompt+gen budget")
    p.add_argument("--datasets", nargs="+", default=["math500", "aime24", "aime25"])
    args = p.parse_args()

    print(f"[B0] model={args.model}")
    print(f"[B0] n={args.n}  temp={args.temperature}  top_p={args.top_p}  top_k={args.top_k}  max_tokens={args.max_tokens}")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False,
        trust_remote_code=True,
    )
    print(f"[B0] vLLM loaded in {time.time()-t0:.1f}s")

    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    DATASETS = {
        "math500": (f"{args.data_dir}/MATH-500/test.jsonl", load_dataset_jsonl),
        "aime24": (f"{args.data_dir}/AIME_2024/aime_2024_problems.parquet", load_dataset_aime24),
        "aime25": (f"{args.data_dir}/AIME_2025/train.jsonl", load_dataset_jsonl),
    }

    all_results = {"model": args.model, "n": args.n, "datasets": {}}
    for name in args.datasets:
        if name not in DATASETS:
            print(f"[B0] skipping unknown dataset: {name}")
            continue
        path, loader = DATASETS[name]
        rows = loader(path)
        print(f"\n[B0] === {name}  ({len(rows)} problems × n={args.n}) ===")

        prompts = []
        for r in rows:
            user_msg = r["problem"].rstrip() + "\n\n" + BOXED_INSTRUCTION
            messages = [{"role": "user", "content": user_msg}]
            prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        gen_t0 = time.time()
        outs = llm.generate(prompts, sp)
        gen_dt = time.time() - gen_t0
        print(f"[B0] {name}: generated in {gen_dt:.1f}s ({len(prompts)*args.n} samples, "
              f"{sum(len(o.outputs) for o in outs)} returned)")

        # outs[i].outputs is list of n samples for problem i
        per_problem = []
        n_correct_total = 0
        for i, out in enumerate(outs):
            gt = rows[i]["answer"]
            scores = []
            for o in out.outputs:
                scores.append(compute_score(o.text, gt))
            mean_at_n = sum(scores) / len(scores) if scores else 0.0
            pass_at_n = 1.0 if any(s >= 1.0 for s in scores) else 0.0
            per_problem.append({"idx": i, "mean@n": mean_at_n, "pass@n": pass_at_n,
                                "n_correct": int(sum(scores)), "n_samples": len(scores)})
            n_correct_total += sum(scores)

        n_problems = len(per_problem)
        agg_mean = sum(p["mean@n"] for p in per_problem) / max(1, n_problems)
        agg_pass = sum(p["pass@n"] for p in per_problem) / max(1, n_problems)
        print(f"[B0] {name}: mean@{args.n} = {agg_mean*100:.2f}%   pass@{args.n} = {agg_pass*100:.2f}%   "
              f"({int(n_correct_total)} correct / {n_problems*args.n} samples)")

        all_results["datasets"][name] = {
            "n_problems": n_problems,
            f"mean@{args.n}": agg_mean,
            f"pass@{args.n}": agg_pass,
            "gen_seconds": gen_dt,
            "per_problem": per_problem,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[B0] results saved to {args.out}")
    print("\n=== SUMMARY ===")
    for k, v in all_results["datasets"].items():
        print(f"  {k:8s}  mean@{args.n} = {v[f'mean@{args.n}']*100:.2f}%   pass@{args.n} = {v[f'pass@{args.n}']*100:.2f}%   ({v['n_problems']} problems)")


if __name__ == "__main__":
    main()
