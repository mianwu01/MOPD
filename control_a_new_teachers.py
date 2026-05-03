"""Control A redo: KL distance from student to each new teacher.

Method matches old control_a_baseline_kl.py exactly:
  - Student greedy-generates 128 tokens on 8 math + 8 medqa prompts
  - Forward each teacher through [prompt+response]
  - Compute reverse-KL(student || teacher) at response positions
  - Use opd.losses._align_vocab (152064→151936 truncation)

Goal: identify any yukang-style outlier teacher (KL >> peers) before training.
"""
import os, json, sys
os.environ.setdefault("PYTHONNOUSERSITE", "1")
sys.path.insert(0, "/home/ubuntu/OPSD_OnPolicyDistillation/src")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from opd.losses import _align_vocab

STUDENT = "/home/ubuntu/models/qwen2.5/student-1.5b-instruct"

TEACHERS = {
    "vanilla-7b-instruct":  "/home/ubuntu/models/qwen2.5/vanilla-7b-instruct",  # size-effect baseline
    "code-coder":            "/home/ubuntu/models/qwen2.5/teacher-code-coder",
    "math-math":             "/home/ubuntu/models/qwen2.5/teacher-math-math",
    "medical-huatuo":        "/home/ubuntu/models/qwen2.5/teacher-medical-huatuo",
    "tool-light":            "/home/ubuntu/models/qwen2.5/teacher-tool-light",
    "search-r1v3":           "/home/ubuntu/models/qwen2.5/teacher-search-r1v3",
}

# Same 8 math + 8 medqa prompts as before — comparable with old Control A numbers
PROMPTS = [
    ("math",  "Compute $\\sum_{n=1}^{10} n^2$. Let's think step by step and output the final answer within \\boxed{}."),
    ("math",  "Find the value of $x$ such that $2x + 7 = 19$. Let's think step by step and output the final answer within \\boxed{}."),
    ("math",  "What is the area of a triangle with vertices at (0,0), (4,0), (0,3)? Let's think step by step and output the final answer within \\boxed{}."),
    ("math",  "Convert the rectangular coordinates (0, 3) to polar coordinates. Let's think step by step and output the final answer within \\boxed{}."),
    ("math",  "Find the derivative of $f(x) = x^3 - 2x + 5$. Let's think step by step and output the final answer within \\boxed{}."),
    ("math",  "Solve $x^2 - 5x + 6 = 0$. Let's think step by step and output the final answer within \\boxed{}."),
    ("math",  "How many ways can 5 books be arranged on a shelf? Let's think step by step and output the final answer within \\boxed{}."),
    ("math",  "What is $\\log_2 32$? Let's think step by step and output the final answer within \\boxed{}."),
    ("medqa", "A 65-year-old man with no known medical history presents with chest pain radiating to the left arm. ECG shows ST elevation in leads II, III, aVF. Which artery is most likely occluded?\nA. Left anterior descending\nB. Right coronary\nC. Left circumflex\nD. Left main\nLet's think step by step and output the final answer letter within \\boxed{}."),
    ("medqa", "Which of the following is the most common cause of community-acquired pneumonia?\nA. Mycoplasma pneumoniae\nB. Streptococcus pneumoniae\nC. Klebsiella pneumoniae\nD. Haemophilus influenzae\nLet's think step by step and output the final answer letter within \\boxed{}."),
    ("medqa", "A patient on long-term lithium therapy presents with polyuria. What's the mechanism?\nA. Central diabetes insipidus\nB. Nephrogenic diabetes insipidus\nC. SIADH\nD. Renal tubular acidosis\nLet's think step by step and output the final answer letter within \\boxed{}."),
    ("medqa", "Which vitamin deficiency causes scurvy?\nA. Vitamin A\nB. Vitamin C\nC. Vitamin D\nD. Vitamin K\nLet's think step by step and output the final answer letter within \\boxed{}."),
    ("medqa", "A child presents with a strawberry-colored tongue, fever, and a sandpaper rash. Most likely diagnosis?\nA. Kawasaki disease\nB. Scarlet fever\nC. Toxic shock syndrome\nD. Measles\nLet's think step by step and output the final answer letter within \\boxed{}."),
    ("medqa", "Which medication is contraindicated in pregnancy due to teratogenicity?\nA. Penicillin\nB. Acetaminophen\nC. Isotretinoin\nD. Ranitidine\nLet's think step by step and output the final answer letter within \\boxed{}."),
    ("medqa", "What hormone is primarily responsible for milk production?\nA. Oxytocin\nB. Prolactin\nC. Estrogen\nD. Progesterone\nLet's think step by step and output the final answer letter within \\boxed{}."),
    ("medqa", "Which finding is most specific for systemic lupus erythematosus?\nA. ANA positive\nB. Anti-dsDNA positive\nC. Rheumatoid factor positive\nD. Elevated ESR\nLet's think step by step and output the final answer letter within \\boxed{}."),
]

DEVICE = "cuda:0"


def gen_student_responses(max_new=128):
    print(f"[gen] loading student {STUDENT}", flush=True)
    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    samples = []
    for i, (domain, p) in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": p}]
        prompt_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        full = gen[0]
        plen = prompt_ids.shape[1]
        samples.append({"domain": domain, "prompt_len": plen, "full_ids": full.tolist()})
        decoded = tok.decode(full[plen:].tolist(), skip_special_tokens=True)[:80]
        print(f"  [{i+1:2}/16] {domain:5s} ← {decoded!r}", flush=True)
    del model; torch.cuda.empty_cache()
    return samples


def get_resp_logits(path, samples):
    print(f"[fwd] {path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    outs = []
    for s in samples:
        full = torch.tensor(s["full_ids"], device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits = model(full).logits[0]  # (T, V)
        resp_start = s["prompt_len"] - 1
        resp_end = full.shape[1] - 1
        outs.append(logits[resp_start:resp_end].float().cpu())
    del model; torch.cuda.empty_cache()
    return outs


def reverse_kl(p_logits, q_logits):
    p_l, q_l = _align_vocab(p_logits, q_logits)
    p_lp = F.log_softmax(p_l, dim=-1)
    q_lp = F.log_softmax(q_l, dim=-1)
    p = p_lp.exp()
    return (p * (p_lp - q_lp)).sum(dim=-1).mean().item()


def main():
    OUT = "/home/ubuntu/MOPD/logs/control_a_new_teachers.json"
    samples = gen_student_responses()
    print(f"\n[main] caching student logits", flush=True)
    student_logits = get_resp_logits(STUDENT, samples)

    results = {}
    for tname, tpath in TEACHERS.items():
        if not os.path.exists(tpath):
            print(f"[skip] {tname}: {tpath} missing", flush=True)
            continue
        try:
            t_logits = get_resp_logits(tpath, samples)
        except Exception as e:
            print(f"[ERR ] {tname}: {e}", flush=True)
            continue
        kls = {"math": [], "medqa": []}
        for s, sl, tl in zip(samples, student_logits, t_logits):
            kls[s["domain"]].append(reverse_kl(sl, tl))
        m_math = sum(kls["math"]) / len(kls["math"])
        m_med  = sum(kls["medqa"]) / len(kls["medqa"])
        results[tname] = {"kl_math": m_math, "kl_medqa": m_med, "kl_avg": (m_math + m_med) / 2}
        print(f"  KL(student || {tname:20s})  math={m_math:7.3f}  medqa={m_med:7.3f}", flush=True)

    print("\n=== SUMMARY ===")
    print(f"{'teacher':<22s}  {'KL math':>10s}  {'KL medqa':>10s}  {'avg':>10s}")
    for t, r in results.items():
        marker = "  ⚠️ OUTLIER" if r["kl_avg"] > 5 else ""
        print(f"{t:<22s}  {r['kl_math']:10.3f}  {r['kl_medqa']:10.3f}  {r['kl_avg']:10.3f}{marker}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
