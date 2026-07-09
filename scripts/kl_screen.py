#!/usr/bin/env python
"""
Teacher screening by reverse-KL, per the MOPD handoff §3/§7 pitfall:
  "Before using any teacher, measure KL(student || teacher) on a small neutral prompt set.
   A teacher that is a large outlier ... tends to make reverse-KL collapse into repetition."

On-policy distillation minimizes reverse-KL = KL(student || teacher) on the STUDENT's own
generations. So we screen exactly that quantity:

  1. Student generates short continuations on a neutral prompt set (its own distribution).
  2. For each candidate teacher, teacher-force over (prompt + student_gen); compute per-token
     KL(student(.|ctx) || teacher(.|ctx)) averaged over response tokens and prompts.
  3. Report mean reverse-KL per teacher + ratio to the median. Flag large outliers
     (far teachers) that risk repetition collapse.

Also reports forward-KL and top-1 agreement for context, plus each model's vocab size
(KDFlow requires all teachers share the student's vocab).

Runs on ONE gpu in bf16; ~17GB (1.5B student + one 7B teacher) so it coexists with other
jobs in the free headroom. Teachers are loaded/freed one at a time.
"""
import argparse, json, gc, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

NEUTRAL_PROMPTS = [
    "Explain why the sky appears blue during the day.",
    "Write a short friendly email declining a meeting invitation.",
    "Summarize the plot of a hero's journey in three sentences.",
    "What are three practical tips for staying focused while working from home?",
    "Describe how a bicycle stays upright when moving.",
    "Give a simple recipe for a vegetable soup.",
    "What is the difference between weather and climate?",
    "Write a two-line poem about autumn leaves.",
    "Explain the concept of supply and demand to a ten-year-old.",
    "List four common causes of procrastination and one fix for each.",
    "How does a refrigerator keep food cold?",
    "Draft a polite reminder message about an overdue library book.",
    "What are the benefits of regular walking for health?",
    "Explain what a hypothesis is in everyday terms.",
    "Describe the water cycle in a few sentences.",
    "Give three tips for a first-time house plant owner.",
    "What makes a good password, and why?",
    "Write a short thank-you note to a mentor.",
    "Explain the phrase 'correlation is not causation' with an example.",
    "How do noise-cancelling headphones work, roughly?",
    "Suggest a simple weekend plan for someone new to a city.",
    "What is compound interest and why does it matter?",
    "Describe the taste and texture of a ripe mango.",
    "Explain how vaccines help the body fight disease, in plain language.",
    "Write a brief product description for a reusable water bottle.",
    "What are some ways to reduce household food waste?",
    "Explain why ice floats on water.",
    "Give three interview tips for a nervous candidate.",
    "Describe how rainbows form.",
    "Write a short caption for a photo of a mountain sunrise.",
    "What is the purpose of a table of contents in a book?",
    "Explain the difference between a want and a need.",
    "How can someone start a small vegetable garden?",
    "Write a one-sentence motivational message for a Monday morning.",
    "What are the main food groups and why is balance important?",
    "Explain how a search engine finds relevant pages, at a high level.",
    "Describe a calm morning routine that takes thirty minutes.",
    "What is the role of sleep in learning and memory?",
    "Write a short apology for arriving late to dinner.",
    "Explain what recycling does and why it helps.",
]


def load(path, dtype, device):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, trust_remote_code=True, attn_implementation="sdpa"
    ).to(device).eval()
    return tok, model


@torch.no_grad()
def student_rollouts(tok, model, prompts, device, max_new_tokens, temperature, seed):
    """Generate the student's own continuations (on-policy). Returns list of (full_ids, resp_start)."""
    torch.manual_seed(seed)
    outs = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
        plen = ids.shape[1]
        gen = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        outs.append((gen[0].detach().cpu(), plen))
    return outs


@torch.no_grad()
def logprobs_over_response(model, full_ids, resp_start, device, shared_vocab):
    """Return log-softmax over the shared vocab for each response-position next-token dist.
    Shape [T_resp, shared_vocab]."""
    ids = full_ids.unsqueeze(0).to(device)
    logits = model(ids).logits[0]  # [T, V]
    # next-token prediction: position t predicts token t+1. Response tokens are [resp_start, T).
    # distributions that GENERATE response tokens live at positions [resp_start-1, T-1).
    lo, hi = resp_start - 1, ids.shape[1] - 1
    logits = logits[lo:hi, :shared_vocab]
    return F.log_softmax(logits.float(), dim=-1)  # [T_resp, shared_vocab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teachers", nargs="+", required=True,
                    help="name=path pairs, e.g. math_expert=/path/to/Qwen2.5-Math-7B-Instruct")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_prompts", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/mnt/cpfs/yangboxue/opsd/MOPD/runs/kl_screen.json")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16
    prompts = NEUTRAL_PROMPTS[: args.n_prompts]

    print(f"[screen] student = {args.student}")
    s_tok, s_model = load(args.student, dtype, device)
    s_vocab = s_model.get_output_embeddings().weight.shape[0]
    print(f"[screen] student vocab_size = {s_vocab}; generating {len(prompts)} rollouts...")
    t0 = time.time()
    rollouts = student_rollouts(s_tok, s_model, prompts, device, args.max_new_tokens,
                                args.temperature, args.seed)
    resp_lens = [full.shape[0] - rs for full, rs in rollouts]
    print(f"[screen] rollouts done in {time.time()-t0:.1f}s; "
          f"mean resp len = {sum(resp_lens)/len(resp_lens):.1f} tokens")

    # student logprobs per rollout (kept on cpu to save mem)
    s_lp = [logprobs_over_response(s_model, full, rs, device, s_vocab).cpu()
            for full, rs in rollouts]

    results = {}
    for spec in args.teachers:
        name, path = spec.split("=", 1)
        print(f"\n[screen] teacher '{name}' = {path}")
        t_tok, t_model = load(path, dtype, device)
        t_vocab = t_model.get_output_embeddings().weight.shape[0]
        shared = min(s_vocab, t_vocab)
        vocab_match = (t_vocab == s_vocab)
        rkl_tokens, fkl_tokens, top1_agree, ntok = 0.0, 0.0, 0, 0
        for (full, rs), slp in zip(rollouts, s_lp):
            tlp = logprobs_over_response(t_model, full, rs, device, shared).cpu()
            sl = slp[:, :shared]
            # renormalize student over shared vocab if vocab sizes differ
            if shared != s_vocab:
                sl = F.log_softmax(sl, dim=-1)
            ps = sl.exp()
            # reverse-KL: KL(student || teacher) = sum ps*(log ps - log pt)  [the OPD objective]
            rkl = (ps * (sl - tlp)).sum(-1)
            # forward-KL for context: KL(teacher || student)
            pt = tlp.exp()
            fkl = (pt * (tlp - sl)).sum(-1)
            rkl_tokens += rkl.sum().item()
            fkl_tokens += fkl.sum().item()
            top1_agree += (sl.argmax(-1) == tlp.argmax(-1)).sum().item()
            ntok += rkl.shape[0]
        results[name] = {
            "path": path, "vocab_size": t_vocab, "vocab_match_student": vocab_match,
            "mean_reverse_kl": rkl_tokens / ntok, "mean_forward_kl": fkl_tokens / ntok,
            "top1_agreement": top1_agree / ntok, "n_response_tokens": ntok,
        }
        print(f"  vocab={t_vocab} match={vocab_match} | "
              f"reverse_KL={results[name]['mean_reverse_kl']:.3f} "
              f"forward_KL={results[name]['mean_forward_kl']:.3f} "
              f"top1_agree={results[name]['top1_agreement']:.3f}")
        del t_model, t_tok
        gc.collect(); torch.cuda.empty_cache()

    # ratios to median reverse-KL -> outlier flags
    rkls = sorted(v["mean_reverse_kl"] for v in results.values())
    median = rkls[len(rkls) // 2]
    print("\n===== SCREEN SUMMARY (reverse-KL = the OPD collapse-risk metric) =====")
    print(f"{'teacher':<22}{'revKL':>9}{'xMedian':>9}{'fwdKL':>9}{'top1':>7}  vocab  flag")
    for name, v in sorted(results.items(), key=lambda kv: kv[1]["mean_reverse_kl"]):
        ratio = v["mean_reverse_kl"] / median if median > 0 else float("inf")
        flag = ""
        if not v["vocab_match_student"]:
            flag += "VOCAB_MISMATCH "
        if ratio >= 3.0:
            flag += "FAR_OUTLIER "  # handoff: far teachers collapse rkl -> repetition
        v["x_median"] = ratio
        print(f"{name:<22}{v['mean_reverse_kl']:>9.3f}{ratio:>9.2f}{v['mean_forward_kl']:>9.3f}"
              f"{v['top1_agreement']:>7.3f}  {v['vocab_size']}  {flag}")
    print("\nGuidance: prefer the LOWEST reverse-KL expert per domain. A FAR_OUTLIER (>=3x median,"
          "\nor absolute rkl >> others) is the collapse risk — use jsd/srkl + lower lr, or swap.")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"student": args.student, "student_vocab": s_vocab,
                   "config": vars(args), "results": results, "median_reverse_kl": median}, f, indent=2)
    print(f"\n[screen] wrote {args.out}")


if __name__ == "__main__":
    main()
