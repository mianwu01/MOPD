#!/usr/bin/env python3
"""Materialize a PEFT (LoRA/OFT) branch into a full HF checkpoint.

KDFlow saves PEFT branches as adapter checkpoints; the merge library and the
bake-off operate on full weights. For OFT branches this folds R into W = R@W0,
after which ord_decompose recovers the rotation exactly (up to block structure).

Usage:
  python3 sweep/materialize_peft.py --base <base_model_dir> \
      --adapter <peft_ckpt_dir> --out <full_ckpt_out_dir>
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype,
                                                 trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.out)
    print(f"materialized {args.adapter} -> {args.out}")


if __name__ == "__main__":
    main()
