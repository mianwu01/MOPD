"""
Control A: baseline KL matrix.

Goal: disambiguate "size effect" (1.5B vs 7B) vs "post-Instruct shift effect"
(GRPO/SFT on top of 7B-Instruct) in Iter1's math=36 / medical=0.4 numbers.

Method: compute reverse-KL(student || X) on response tokens for X ∈ {
  vanilla-7B-Instruct, 5 teachers
}, using STUDENT-generated responses (matches OPD's on-policy setup).

Output: a small JSON + printed table.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONNOUSERSITE", "1")
sys.path.insert(0, "/home/ubuntu/OPSD_OnPolicyDistillation/src")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from opd.losses import _align_vocab


STUDENT = "/home/ubuntu/models/qwen2.5/student-1.5b-instruct"

TEACHERS = {
    "vanilla-7b-instruct":  "/home/ubuntu/models/qwen2.5/vanilla-7b-instruct",
    "math-yukang":          "/home/ubuntu/models/qwen2.5/teacher-math-yukang",
    "medical-umls":         "/home/ubuntu/models/qwen2.5/teacher-medical-umls",
    "search-searchr1":      "/home/ubuntu/models/qwen2.5/teacher-search-searchr1",
    "tool-toolrl":          "/home/ubuntu/models/qwen2.5/teacher-tool-toolrl",
    "code-svs":             "/home/ubuntu/models/qwen2.5/teacher-code-svs",
}


# 8 math + 8 medqa prompts, real samples used in Iter1 train+eval.
PROMPTS = [
    # math (DAPO-style boxed instruction)
    ("math",  "Compute $\\sum_{n=1}^{10} n^2$. Let's think step by step and output the final answer within \\boxed{}."),
    ("math",  "Find the value of $x$ such that $2x + 7 = 19$. Let's think step by step and output the final answer within \\boxed{}."),
    ("math",  "What is the area of a triangle with vertices at (0,0), (4,0), (0,3)? Let's think step by step and output the final answer within \\boxed{}."),
    ("math",  "Convert the rectangular coordinates (0, 3) to polar coordinates. Let's think step by step and output the final answer within \\boxed{}."),
    ("math",  "Find the derivative of $f(x) = x^3 - 2x + 5$. Let's think step by step and output the final answer within \\boxed{}."),
    ("math",  "Solve $x^2 - 5x + 6 = 0$. Let's think step by step and output the final answer within \\boxed{}."),
    ("math",  "How many ways can 5 books be arranged on a shelf? Let's think step by step and output the final answer within \\boxed{}."),
    ("math",  "What is $\\log_2 32$? Let's think step by step and output the final answer within \\boxed{}."),
    # medqa (4-option MCQ format)
    ("medqa", "A 65-year-old man with no known medical history presents with chest pain radiating to the left arm. ECG shows ST elevation in leads II, III, aVF. Which artery is most likely occluded?\nA. Left anterior descending\nB. Right coronary\nC. Left circumflex\nD. Left main\nLet's think step by step and output the final answer letter within \\boxed{}."),
    ("medqa", "Which of the following is the most common cause of community-acquired pneumonia?\nA. Mycoplasma pneumoniae\nB. Streptococcus pneumoniae\nC. Klebsiella pneumoniae\nD. Haemophilus influenzae\nLet's think step by step and output the final answer letter within \\boxed{}."),
    ("medqa", "A patient on long-term lithium therapy presents with polyuria. What's the mechanism?\nA. Central diabetes insipidus\nB. Nephrogenic diabetes insipidus\nC. SIADH\nD. Renal tubular acidosis\nLet's think step by step and output the final answer letter within \\boxed{}."),
    ("medqa", "Which vitamin deficiency causes scurvy?\nA. Vitamin A\nB. Vitamin C\nC. Vitamin D\nD. Vitamin K\nLet's think step by step and output the final answer letter within \\boxed{}."),
    ("medqa", "A child presents with a strawberry-colored tongue, fever, and a sandpaper rash. Most likely diagnosis?\nA. Kawasaki disease\nB. Scarlet fever\nC. Toxic shock syndrome\nD. Measles\nLet's think step by step and output the final answer letter within \\boxed{}."),
    ("medqa", "Which medication is contraindicated in pregnancy due to teratogenicity?\nA. Penicillin\nB. Acetaminophen\nC. Isotretinoin\nD. Ranitidine\nLet's think step by step and output the final answer letter within \\boxed{}."),
    ("medqa", "What hormone is primarily responsible for milk production?\nA. Oxytocin\nB. Prolactin\nC. Estrogen\nD. Progesterone\nLet's think step by step and output the final answer letter within \\boxed{}."),
    ("medqa", "Which finding is most specific for systemic lupus erythematosus?\nA. ANA positive\nB. Anti-dsDNA positive\nC. Rheumatoid factor positive\nD. Elevated ESR\nLet's think step by step and output the final answer letter within \\boxed{}."),
]


def gen_responses(student_path: str, prompts: list[tuple[str, str]], max_new: int = 128, batch_size: int = 8) -> list[dict]:
    """Have the student generate one response per prompt. These responses
    define the "on-policy" token positions where we then measure cross-model KL.
    """
    print(f"[gen] loading student {student_path}")
    tok = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(student_path, torch_dtype=torch.bfloat16,
                                                  attn_implementation="flash_attention_2",
                                                  trust_remote_code=True).to("cuda:0")
    model.eval()

    out = []
    for i, (domain, p) in enumerate(prompts):
        msgs = [{"role": "user", "content": p}]
        prompt_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            gen = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        full_ids = gen[0]
        prompt_len = prompt_ids.shape[1]
        response_ids = full_ids[prompt_len:].tolist()
        out.append({"domain": domain, "prompt": p, "prompt_ids": prompt_ids[0].tolist(),
                    "response_ids": response_ids, "full_ids": full_ids.tolist(), "prompt_len": prompt_len})
        decoded = tok.decode(response_ids, skip_special_tokens=True)[:80]
        print(f"  [{i+1}/{len(prompts)}] {domain:5s} ← {decoded!r}")
    del model
    torch.cuda.empty_cache()
    return out


def get_logits(model_path: str, samples: list[dict]) -> list[torch.Tensor]:
    """Forward each [prompt+response] through model, return logits sliced to response positions only."""
    print(f"[fwd] loading {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                  attn_implementation="flash_attention_2",
                                                  trust_remote_code=True).to("cuda:0")
    model.eval()
    out = []
    for s in samples:
        full_ids = torch.tensor(s["full_ids"], device="cuda:0").unsqueeze(0)
        with torch.no_grad():
            logits = model(full_ids).logits[0]  # (T, V)
        # response positions: positions that PREDICT response tokens
        # token at index i predicts token at index i+1; response spans [prompt_len, T)
        # so prediction logits for response are at indices [prompt_len-1, T-1)
        resp_start = s["prompt_len"] - 1
        resp_end = full_ids.shape[1] - 1
        resp_logits = logits[resp_start:resp_end].float().cpu()  # (R, V)
        out.append(resp_logits)
    del model
    torch.cuda.empty_cache()
    return out


def reverse_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(p_softmax || q_softmax), averaged over tokens. Aligns vocab via _align_vocab."""
    p_l, q_l = _align_vocab(p_logits, q_logits)
    p_lp = F.log_softmax(p_l, dim=-1)
    q_lp = F.log_softmax(q_l, dim=-1)
    # KL(p || q) = sum p * (log p - log q)
    p = p_lp.exp()
    kl_per_tok = (p * (p_lp - q_lp)).sum(dim=-1)
    return kl_per_tok.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ubuntu/MOPD/logs/control_a_kl.json")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--include-vanilla", action="store_true",
                    help="include vanilla-7b-instruct (only if downloaded)")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    targets = list(TEACHERS.items())
    if not args.include_vanilla:
        targets = [(n, p) for n, p in targets if n != "vanilla-7b-instruct"]
        print("[main] skipping vanilla-7b (not downloaded yet)")

    # 1. student generates responses on the 16 prompts
    samples = gen_responses(STUDENT, PROMPTS, max_new=args.max_new)

    # 2. cache student's logits at response positions
    print(f"\n[main] forwarding student through {len(samples)} samples")
    student_logits = get_logits(STUDENT, samples)

    # 3. for each teacher, forward and compute KL(student || teacher) per sample
    results = {"per_teacher": {}, "per_sample": {}}
    for tname, tpath in targets:
        print(f"\n[main] forwarding {tname}")
        teacher_logits = get_logits(tpath, samples)

        per_dom = {"math": [], "medqa": []}
        for s, s_l, t_l in zip(samples, student_logits, teacher_logits, strict=True):
            kl = reverse_kl(s_l, t_l)
            per_dom[s["domain"]].append(kl)

        m_math = sum(per_dom["math"]) / max(1, len(per_dom["math"]))
        m_med  = sum(per_dom["medqa"]) / max(1, len(per_dom["medqa"]))
        results["per_teacher"][tname] = {"kl_on_math_resp": m_math, "kl_on_medqa_resp": m_med,
                                          "kl_avg": (m_math + m_med) / 2}
        results["per_sample"][tname] = per_dom
        print(f"  KL(student||{tname:25s})  math={m_math:7.3f}  medqa={m_med:7.3f}")

    print("\n=== summary ===")
    print(f"{'teacher':<25s}  {'KL on math resp':>15s}  {'KL on medqa resp':>17s}")
    for tname, info in results["per_teacher"].items():
        print(f"{tname:<25s}  {info['kl_on_math_resp']:15.3f}  {info['kl_on_medqa_resp']:17.3f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
