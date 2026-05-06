"""Alignment diagnostic for OPD: does distillation actually align student to teacher?

Computes 4 metrics on a held-out set of MATH-500 prompts:
  A. token-level KL(student || teacher) on student-sampled trajectories
  B. argmax agreement rate (token-level)
  C. top-5 cover rate (teacher's top-1 ∈ student's top-5)
  D. teacher-CE under student (negative log-likelihood of teacher's response under student)

Run on multiple ckpts (baseline 1.5B, ExpC step 39, v10b step 20, v10f step 39, ...)
to produce a comparison table answering "does our OPD pipeline actually distill?"

Usage:
  python eval_alignment.py \
      --student /path/to/student_ckpt \
      --teacher /path/to/teacher_ckpt \
      --n_problems 100 \
      --out_json /home/ubuntu/MOPD/logs/alignment_<tag>.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="HF path or local dir for student model")
    ap.add_argument("--teacher", required=True, help="HF path or local dir for teacher model")
    ap.add_argument("--val_parquet", default="/home/ubuntu/OPSD_OnPolicyDistillation/data/iter1_mixed/val_math500.parquet")
    ap.add_argument("--n_problems", type=int, default=100, help="subset size of val_math500")
    ap.add_argument("--max_response", type=int, default=512, help="cap response length for diagnostic")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--student_gpu", type=int, default=0)
    ap.add_argument("--teacher_gpu", type=int, default=1)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForCausalLM

    student_dev = torch.device(f"cuda:{args.student_gpu}")
    teacher_dev = torch.device(f"cuda:{args.teacher_gpu}")

    print(f"[align] loading student from {args.student}", flush=True)
    student_tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    student_lm = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(student_dev).eval()

    print(f"[align] loading teacher from {args.teacher}", flush=True)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    teacher_lm = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(teacher_dev).eval()

    student_V = student_lm.config.vocab_size
    teacher_V = teacher_lm.config.vocab_size
    common_V = min(student_V, teacher_V)
    print(f"[align] student V={student_V}, teacher V={teacher_V}, common V={common_V}", flush=True)

    df = pd.read_parquet(args.val_parquet).head(args.n_problems)
    print(f"[align] loaded {len(df)} problems from {args.val_parquet}", flush=True)

    metrics = {
        "kl_per_token": [],          # KL(student || teacher) — A
        "argmax_agree": [],          # argmax(s) == argmax(t) — B
        "top5_cover": [],            # argmax(t) ∈ top-5(s) — C
        "teacher_ce_under_student": [],  # -log p_s(t_token) — D, on teacher's own responses
        "n_tokens_per_problem": [],
    }

    rng = torch.Generator(device=student_dev).manual_seed(42)

    for prob_idx, (_, row) in enumerate(df.iterrows()):
        chat = list(row["prompt"])
        prompt_text = student_tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        s_input = student_tok(prompt_text, return_tensors="pt").to(student_dev)
        prompt_len = s_input.input_ids.shape[1]

        with torch.no_grad():
            gen = student_lm.generate(
                **s_input,
                max_new_tokens=args.max_response,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.8,
                top_k=20,
                pad_token_id=student_tok.eos_token_id,
                generation_config=None,
            )
        full_ids_s = gen[0]
        response_ids = full_ids_s[prompt_len:]
        n_resp = response_ids.shape[0]
        if n_resp == 0:
            continue

        with torch.no_grad():
            s_logits = student_lm(full_ids_s.unsqueeze(0)).logits[0]
        s_logits_resp = s_logits[prompt_len - 1 : prompt_len - 1 + n_resp]

        teacher_text = teacher_tok.decode(full_ids_s, skip_special_tokens=False)
        t_input_ids = teacher_tok(teacher_text, return_tensors="pt").input_ids.to(teacher_dev)
        with torch.no_grad():
            t_logits = teacher_lm(t_input_ids).logits[0]
        t_logits_resp = t_logits[-(n_resp + 1) : -1] if n_resp > 0 else None
        if t_logits_resp is None or t_logits_resp.shape[0] != n_resp:
            min_len = min(s_logits_resp.shape[0], t_logits_resp.shape[0] if t_logits_resp is not None else 0)
            if min_len == 0:
                continue
            s_logits_resp = s_logits_resp[:min_len]
            t_logits_resp = t_logits_resp[:min_len]

        s_logits_resp = s_logits_resp[..., :common_V].float()
        t_logits_resp = t_logits_resp[..., :common_V].float().to(student_dev)

        s_lp = F.log_softmax(s_logits_resp, dim=-1)
        t_lp = F.log_softmax(t_logits_resp, dim=-1)
        s_p = s_lp.exp()
        kl_st = (s_p * (s_lp - t_lp)).sum(dim=-1)
        metrics["kl_per_token"].append(float(kl_st.mean().item()))

        s_argmax = s_lp.argmax(dim=-1)
        t_argmax = t_lp.argmax(dim=-1)
        metrics["argmax_agree"].append(float((s_argmax == t_argmax).float().mean().item()))

        s_top5 = s_lp.topk(5, dim=-1).indices
        cover = (t_argmax.unsqueeze(-1) == s_top5).any(dim=-1).float().mean().item()
        metrics["top5_cover"].append(float(cover))

        with torch.no_grad():
            t_input_ids_full = t_input_ids[0]
            t_gen = teacher_lm.generate(
                t_input_ids_full[:prompt_len].unsqueeze(0).to(teacher_dev),
                max_new_tokens=args.max_response,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.8,
                top_k=20,
                pad_token_id=teacher_tok.eos_token_id,
                generation_config=None,
            )
            teacher_resp_ids = t_gen[0][prompt_len:]
            n_t_resp = teacher_resp_ids.shape[0]
            if n_t_resp > 0:
                full_with_t_resp = torch.cat([
                    s_input.input_ids[0], teacher_resp_ids[: args.max_response].to(student_dev)
                ])
                s_logits_on_t = student_lm(full_with_t_resp.unsqueeze(0)).logits[0]
                target = full_with_t_resp[prompt_len:]
                logits_for_target = s_logits_on_t[prompt_len - 1 : prompt_len - 1 + target.shape[0]]
                logits_for_target = logits_for_target[..., :common_V].float()
                target_safe = target.clamp(max=common_V - 1)
                ce = F.cross_entropy(logits_for_target, target_safe, reduction="mean")
                metrics["teacher_ce_under_student"].append(float(ce.item()))

        metrics["n_tokens_per_problem"].append(n_resp)

        if (prob_idx + 1) % 10 == 0:
            print(f"  [{prob_idx+1}/{len(df)}] mean KL={sum(metrics['kl_per_token'])/len(metrics['kl_per_token']):.3f} "
                  f"argmax={sum(metrics['argmax_agree'])/len(metrics['argmax_agree']):.3f} "
                  f"top5={sum(metrics['top5_cover'])/len(metrics['top5_cover']):.3f}",
                  flush=True)

    summary = {
        "student_path": args.student,
        "teacher_path": args.teacher,
        "n_problems_used": len(metrics["kl_per_token"]),
        "mean_kl_student_teacher": float(sum(metrics["kl_per_token"]) / max(1, len(metrics["kl_per_token"]))),
        "mean_argmax_agreement": float(sum(metrics["argmax_agree"]) / max(1, len(metrics["argmax_agree"]))),
        "mean_top5_cover": float(sum(metrics["top5_cover"]) / max(1, len(metrics["top5_cover"]))),
        "mean_teacher_ce_under_student": float(
            sum(metrics["teacher_ce_under_student"]) / max(1, len(metrics["teacher_ce_under_student"]))
        ) if metrics["teacher_ce_under_student"] else None,
        "median_response_len": float(sorted(metrics["n_tokens_per_problem"])[len(metrics["n_tokens_per_problem"])//2])
            if metrics["n_tokens_per_problem"] else None,
    }
    print("\n=== ALIGNMENT SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(args.out_json, "w") as f:
        json.dump({"summary": summary, "per_problem": metrics}, f, indent=2)
    print(f"\nsaved -> {args.out_json}")


if __name__ == "__main__":
    main()
