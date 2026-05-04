"""Real SearchR1 eval WITH retrieval.

For each val_search row:
  1. Initialize prompt (SearchR1-format with <search>/<information>/<answer>)
  2. Multi-turn loop (max 4 turns):
     - Generate with vLLM until </search> or </answer>
     - If </search> appeared: extract query, call retrieval_server, append <information>...</information>
     - If </answer> appeared: stop
  3. Extract <answer>, EM/F1 vs gold

Compares to "fake" search numbers from current paper baseline.
Uses the SearchR1 retrieval_server.py (running locally at http://127.0.0.1:8000)
"""
import os, sys, re, json, argparse, requests
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.insert(0, "/home/ubuntu/OPSD_OnPolicyDistillation/src")

import pandas as pd
from collections import defaultdict


SEARCH_TAG_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def call_retrieval(query: str, k: int = 3, server_url: str = "http://127.0.0.1:8000/retrieve") -> str:
    """Call SearchR1's FastAPI retrieval server. Return formatted <information> string."""
    try:
        r = requests.post(server_url, json={"queries": [query], "topk": k, "return_scores": False}, timeout=30)
        r.raise_for_status()
        docs = r.json()["result"][0]
        # docs is list of dicts with 'contents' or {'title','text'}
        chunks = []
        for d in docs:
            if isinstance(d, dict):
                txt = d.get("contents") or d.get("text") or str(d)
            else:
                txt = str(d)
            chunks.append(txt.strip())
        return "\n".join(chunks)
    except Exception as e:
        return f"(retrieval error: {type(e).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Path to verl FSDP ckpt OR HF safetensors dir")
    ap.add_argument("--val_parquet", default="/home/ubuntu/OPSD_OnPolicyDistillation/data/5domain_mixed/val_search.parquet")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--max_turns", type=int, default=4)
    ap.add_argument("--max_response_per_turn", type=int, default=512)
    ap.add_argument("--n_samples", type=int, default=1, help="Samples per prompt (mean@n)")
    ap.add_argument("--retrieval_url", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="Limit val rows for quick test")
    args = ap.parse_args()

    out_json = args.out_json or args.ckpt_dir.rstrip("/") + "/eval_search_with_retrieval.json"

    # Lazy import vLLM
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from rewards.search_reward import compute_score, _norm, _f1, _to_target_list

    # Sanity check retrieval server
    print(f"[boot] testing retrieval server at {args.retrieval_url}", flush=True)
    test = call_retrieval("capital of France", k=2, server_url=args.retrieval_url)
    if test.startswith("(retrieval error"):
        print(f"  ❌ retrieval server not reachable: {test}")
        sys.exit(1)
    print(f"  ✓ retrieval server returned {len(test)} chars", flush=True)

    print(f"[boot] loading val: {args.val_parquet}", flush=True)
    df = pd.read_parquet(args.val_parquet)
    if args.limit: df = df.head(args.limit)
    print(f"  {len(df)} rows; subsets: {df['extra_info'].apply(lambda x: x.get('subset') if isinstance(x, dict) else None).value_counts().to_dict()}", flush=True)

    print(f"[boot] loading model from {args.ckpt_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt_dir)
    llm = LLM(model=args.ckpt_dir, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=8192, enforce_eager=False)

    # Per-turn sampling: stop on </search> or </answer> to detect tool calls
    stop_strs = ["</search>", "</answer>"]
    params = SamplingParams(n=args.n_samples, temperature=0.7, top_p=0.8, top_k=20,
                            max_tokens=args.max_response_per_turn, stop=stop_strs, include_stop_str_in_output=True)
    # Final-turn params: force <answer> mode (only stop on </answer>) — model is steered away from another <search>
    final_params = SamplingParams(n=args.n_samples, temperature=0.7, top_p=0.8, top_k=20,
                                  max_tokens=args.max_response_per_turn, stop=["</answer>"], include_stop_str_in_output=True)

    results = []
    by_subset = defaultdict(list)

    for i, row in df.iterrows():
        prompt_msgs = list(row["prompt"])
        gt = row["reward_model"]["ground_truth"]
        _at = row["extra_info"].get("all_targets")
        if _at is not None and len(_at) > 0:
            gt_list = list(_at)
        else:
            gt_list = [gt]
        subset = row["extra_info"].get("subset", "unknown")

        # Multi-turn rollout
        prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_response = ""
        n_searches = 0
        for turn in range(args.max_turns):
            is_final = (turn == args.max_turns - 1)
            sp = final_params if is_final else params
            if is_final and "<answer>" not in full_response:
                # Steer toward answer on the last turn
                full_response += "\n<answer> "
            outs = llm.generate([prompt_text + full_response], sp)
            seg = outs[0].outputs[0].text
            full_response += seg
            if "</answer>" in seg:
                break
            if "</search>" in seg and not is_final:
                m = SEARCH_TAG_RE.search(full_response[-2000:])
                if m:
                    q = m.group(1).strip()
                    info = call_retrieval(q, k=args.topk, server_url=args.retrieval_url)
                    full_response += f"\n<information>{info}</information>\n"
                    n_searches += 1
                    continue
            # No more tool calls and no answer; break
            if is_final:
                break

        # Score
        score_em = compute_score(full_response, gt_list)
        m_ans = ANSWER_TAG_RE.search(full_response)
        pred = m_ans.group(1).strip() if m_ans else ""
        results.append({
            "subset": subset, "n_searches": n_searches, "score": score_em,
            "pred": pred, "gold": gt_list[0] if gt_list else "",
            "task_id": row["extra_info"].get("task_id", str(i)),
            "full_response_tail": full_response[-300:],
        })
        by_subset[subset].append(score_em)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(df)}] avg score so far: {sum(r['score'] for r in results)/len(results):.4f}", flush=True)

    # Aggregate
    overall = sum(r["score"] for r in results) / max(1, len(results))
    print(f"\n=== SearchR1 WITH retrieval — overall {overall:.4f} ===")
    summary = {"overall": overall, "by_subset": {}}
    for s, vals in by_subset.items():
        m = sum(vals) / len(vals)
        summary["by_subset"][s] = {"mean": m, "n": len(vals)}
        print(f"  {s:>20}  n={len(vals):3d}  mean={m:.4f}")

    with open(out_json, "w") as f:
        json.dump({"summary": summary, "rows": results}, f, indent=2)
    print(f"\nsaved -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
