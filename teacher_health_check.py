"""CPU sanity-check teachers before any training, to catch yukang-style collapse.

For each teacher:
  - Load tokenizer + model on CPU (bf16)
  - 3 probes: math, medical, general chat
  - 30-token greedy generation
  - VALIDATE: not all-same-token, ≥10 unique tokens, no NaN logits

Usage: python teacher_health_check.py [<path1> <path2> ...]
       or with no args, defaults to the 5 new teachers.
"""
import os, sys, time
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_TEACHERS = [
    "/home/ubuntu/models/qwen2.5/teacher-code-coder",
    "/home/ubuntu/models/qwen2.5/teacher-math-math",
    "/home/ubuntu/models/qwen2.5/teacher-medical-huatuo",
    "/home/ubuntu/models/qwen2.5/teacher-tool-light",
    "/home/ubuntu/models/qwen2.5/teacher-search-r1v3",
]

PROBES = [
    ("math",    "Compute 17 * 23 step by step."),
    ("medical", "What is the typical first-line treatment for hypertension in adults?"),
    ("chat",    "Briefly explain what photosynthesis is."),
]


def check_one(path: str, max_new: int = 30) -> dict:
    name = os.path.basename(path.rstrip("/"))
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s, vocab_size={tok.vocab_size}", flush=True)

    failures = []
    for tag, probe in PROBES:
        msgs = [{"role": "user", "content": probe}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None)
        new_ids = out[0, ids.shape[1]:].tolist()
        decoded = tok.decode(new_ids, skip_special_tokens=False)
        uniq = len(set(new_ids))
        all_same = uniq == 1
        bad_repeat = uniq < 5
        nan = any(torch.isnan(model(ids).logits).flatten().tolist()[:10])
        status = "OK"
        if all_same:
            status = f"FAIL: all tokens same (id={new_ids[0]})"
            failures.append((tag, status))
        elif bad_repeat:
            status = f"WARN: only {uniq} unique tokens in 30"
            failures.append((tag, status))
        elif nan:
            status = "FAIL: NaN logits"
            failures.append((tag, status))
        print(f"  [{tag:8s}] uniq={uniq:>2}  status={status}", flush=True)
        print(f"    out: {repr(decoded[:120])}", flush=True)

    del model
    return {"name": name, "path": path, "failures": failures}


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TEACHERS
    results = [check_one(p) for p in paths]
    print("\n=== SUMMARY ===", flush=True)
    bad = []
    for r in results:
        marker = "✅" if not r["failures"] else "❌"
        print(f"  {marker} {r['name']}: {len(r['failures'])} fail/warn")
        for tag, msg in r["failures"]:
            print(f"      [{tag}] {msg}")
        if any("FAIL" in m for _, m in r["failures"]):
            bad.append(r["name"])
    if bad:
        print(f"\n🚨 BAD TEACHERS (do NOT use for training): {bad}", flush=True)
        sys.exit(1)
    print("\n✅ ALL TEACHERS PASSED", flush=True)


if __name__ == "__main__":
    main()
