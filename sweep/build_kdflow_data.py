#!/usr/bin/env python3
"""Build KDFlow-format training JSONL per domain (+ optional split halves).

KDFlow multi-teacher rows need:
  {"messages": [{"role": "user", "content": ...}], "teacher_routing_key": "<domain>"}

Presets map a domain to a HF dataset + prompt field. Run on the cluster (needs
`datasets` + network). --split_half writes {out}.halfA/.halfB by even/odd row,
which feeds the split-half reliability diagnostic (train one branch per half).

NOTE (elicitation fix): prompts are stored as plain user turns; the chat
template is applied by KDFlow at tokenization time with each model's own
template, so teachers are scored on their native format. Do NOT bake a generic
"think step by step" system prompt into the data.

Usage:
  python3 sweep/build_kdflow_data.py --preset math    --out data/kdflow/math.jsonl --limit 6000
  python3 sweep/build_kdflow_data.py --preset medical --out data/kdflow/medical.jsonl --limit 6000 --split_half
  python3 sweep/build_kdflow_data.py --dataset your/hf-dataset --field question \
      --domain custom --out data/kdflow/custom.jsonl
  cat data/kdflow/{math,medical,code,search,tool}.jsonl | shuf > data/kdflow/mixed_5domain.jsonl
"""

import argparse
import json
import os
import random

PRESETS = {
    # domain: (hf dataset, config, split, prompt field)
    "math": ("open-r1/DAPO-Math-17k-Processed", "en", "train", "prompt"),
    "medical": ("GBaker/MedQA-USMLE-4-options", None, "train", "question"),
    "code": ("BAAI/TACO", None, "train", "question"),
    "search": ("google-research-datasets/natural_questions", "default", "train", "question"),
    "tool": ("Team-ACE/ToolACE", None, "train", "conversations"),
}


def extract_prompt(row, field):
    v = row[field]
    if isinstance(v, str):
        return v
    if isinstance(v, dict) and "text" in v:      # natural_questions style
        return v["text"]
    if isinstance(v, list) and v and isinstance(v[0], dict):  # conversations style
        for turn in v:
            if turn.get("from") in ("human", "user") or turn.get("role") == "user":
                return turn.get("value") or turn.get("content")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--dataset")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--field", default=None)
    p.add_argument("--domain", default=None, help="routing key (defaults to preset name)")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_half", action="store_true",
                   help="also write .halfA/.halfB files for the reliability probe")
    args = p.parse_args()

    if args.preset:
        ds_name, config, split, field = PRESETS[args.preset]
        domain = args.domain or args.preset
    else:
        if not (args.dataset and args.field and args.domain):
            p.error("without --preset, provide --dataset --field --domain")
        ds_name, config, split, field = args.dataset, args.config, args.split, args.field
        domain = args.domain

    from datasets import load_dataset
    ds = load_dataset(ds_name, config, split=split) if config else load_dataset(ds_name, split=split)

    rows = []
    for r in ds:
        prompt = extract_prompt(r, field)
        if prompt and len(prompt.strip()) > 0:
            rows.append({"messages": [{"role": "user", "content": prompt.strip()}],
                         "teacher_routing_key": domain})
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{domain}: {len(rows)} rows -> {args.out}")

    if args.split_half:
        for tag, sel in (("halfA", 0), ("halfB", 1)):
            path = args.out.replace(".jsonl", f".{tag}.jsonl")
            with open(path, "w") as f:
                for i, r in enumerate(rows):
                    if i % 2 == sel:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {tag}: {len(rows)//2} rows -> {path}")


if __name__ == "__main__":
    main()
