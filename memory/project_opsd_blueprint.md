---
name: OPSD project blueprint (Qwen2.5 unified-base strategy)
description: Definitive project plan for the multi-teacher OPD + RL PhD project. Locks in unified-family teachers around Qwen2.5-7B-Instruct, lists per-domain teacher checkpoints, stages, and remaining work.
type: project
originSessionId: a4b9a9c8-cefb-4306-842b-75d91aaf5c8e
---
# Decision (locked-in 2026-05-02)

**Unified base family: `Qwen/Qwen2.5-7B-Instruct`**

Why this beat the prior Qwen3-1.7B-Base plan:
- 5 of 6 target domains already have publicly-released **Qwen2.5-7B-Instruct + RL** teacher checkpoints. Saves weeks of self-training.
- Aligns with the PhD's listed codebases (Search-R1, ToolRL produce Qwen2.5-7B-derived weights).
- TIP paper's "Qwen2.5 pair" experiment also uses Qwen2.5-1.5B as student → existing prior-work alignment.
- Methodology: identical base across all 6 teachers (strict same-family OPD, no tokenizer hacks, no cross-family weirdness).

**Cost paid:** prior Stage 1 data (B0/B2 on Qwen3-1.7B-Base) is now historical only — not the project's anchor numbers. Easy to redo on Qwen2.5.

# Roles fixed

| Role | Model | Source | Notes |
|---|---|---|---|
| **Student** | `Qwen/Qwen2.5-1.5B-Instruct` | HF official | Matches TIP "Qwen2.5 pair"; ~4.7× capacity gap to 7B teachers |
| **Math teacher** | `Yukang/Qwen2.5-7B-Open-R1-GRPO` | HF community | GRPO on `open-r1/OpenR1-Math-220k`, base = Qwen2.5-7B-Instruct (verified) |
| **Search teacher** | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3` | HF | Search-R1 v0.3 collection; NQ/HotpotQA GRPO; Qwen2 arch hidden=3584 layers=28 (verified) |
| **Tool-call teacher** | `emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold` | HF | ToolRL collection's only Qwen2.5-7B-Instruct ckpt; trained on ToolRL 4k (ToolACE/Hammer/xLAM derivatives) |
| **Code teacher** | `RLVR-SvS/SvS-Qwen-Code-7B` | HF | Base = Qwen2.5-7B-Instruct, RLVR on coding tasks |
| **Medical teacher** | `prithivMLmods/Qwen-UMLS-7B-Instruct` | HF | Base = Qwen2.5-7B-Instruct. **SFT on UMLS** (NOT GRPO/RL). Less specialized than peer teachers; flag in paper |
| **Open-domain hard-to-verify teacher** | (self-train via RuscaRL) | TBD | RuscaRL `Qwen2.5-7B-Instruct/healthbench_RuscaRL.sh` exists. No public ckpt — must train. Defer to Stage 2 expansion |

**Backup math teacher**: `FutureMa/Qwen2.5-7B-Instruct-GRPO-Math` (GRPO on NuminaMath-TIR 500 — smaller training, useful as 2nd math teacher for ablation).

**Code GGUF alt**: `anshul6273/Qwen2.5-7B-Atcoder-Reasoning-v1-GGUF` exists but GGUF only — no safetensors → can't load via verl/transformers without conversion. Skip unless needed.

# Empirical baseline KL (Control A, 2026-05-03)

Measured `KL(student || teacher)` on response tokens of 16 representative prompts (8 math + 8 medqa), using student's greedy-generated responses. Method matches OPSD `losses.compute_reverse_kl_loss + _align_vocab` exactly. Saved to `MOPD/logs/control_a_kl.json`, analysis in `MOPD/logs/control_a_analysis.md`.

| Teacher | KL on math | KL on medqa | Notes |
|---|---:|---:|---|
| vanilla-7b-instruct | 0.10 | 0.39 | size-effect baseline (1.5B vs 7B both Instruct) |
| **math-yukang** | **37.0** | **43.8** | **OUTLIER — 400× further than peers** |
| medical-umls (SFT) | 0.07 | 0.24 | Even closer than vanilla 7B (UMLS SFT shrunk size gap) |
| search-searchr1 (GRPO) | 0.09 | 0.35 | Matches vanilla 7B |
| tool-toolrl (GRPO) | 0.10 | 0.38 | Matches vanilla 7B |
| code-svs (DAPO) | 0.10 | 0.37 | Matches vanilla 7B |

**Reframed teacher landscape**:
- Iter1's "80× math vs medical asymmetry" is **not** a "GRPO vs SFT" pattern. It's **specific to math-yukang**: extensive Open-R1 long-CoT GRPO rewrote the base distribution wholesale.
- Other GRPO/DAPO teachers (search/tool/code) are nearly **indistinguishable from vanilla 7B-Instruct** on generic chat prompts. Their RL only changed task-specific output formats (`<search>` tags, tool calls, code blocks). On vanilla math/medqa prompts they look like base 7B.
- Per-teacher loss normalization is needed **only when math-yukang is in the bag**. Iter2 with {search + tool + code + medical} (no math-yukang) might not need rescaling at all.
- For paper framing: this is interesting on its own — distillation-time "teacher distance" depends heavily on which RL training was applied, not just whether RL was applied.

**Backup plan if math-yukang continues to destabilize**: swap to `FutureMa/Qwen2.5-7B-Instruct-GRPO-Math` (smaller GRPO training on NuminaMath-TIR 500, expected lower KL — to be measured).

# 4-stage plan

| Stage | Goal | Status |
|---|---|---|
| **Stage 0** | Smoke test (pipeline end-to-end on smallest config). Done historically on Qwen3 (still meaningful — proved OPSD codebase works) | ✅ legacy, redo on Qwen2.5 cheap |
| **Stage 1** | B0 (raw eval) + B2 (single-teacher OPD on math) anchors on Qwen2.5-1.5B-Instruct + math teacher (Yukang) | ⏳ to redo |
| **Stage 2** | Multi-teacher OPD baseline. Iteration 1: math + 1 other (probably search). Iteration 2: math + 3-4 of {search, tool, code}. Final: all 6 if open-domain self-trained | ⏳ to start |
| **Stage 3** | OPD + RL novelty (extend `OPD_REWARD_BETA` hook in OPSD repo) | ⏳ deferred |

# Stage 2 engineering still required

The OPSD codebase has **NO multi-teacher support** (all `ref_module_fsdp` is singular; verl ref slot is single-teacher by design). To implement (per task #14 analysis):

1. **Config schema**: replace `actor_rollout_ref.ref.model.path` (single str) → list of paths + `data_source → teacher_idx` routing map
2. **OPDWorker init**: build N `ref_module_fsdp` instances with FSDP+CPU offload each
3. **Phase 1 of `update_opd`**: group batch by `data_source`, for each teacher: load → fwd → cache logits → offload
4. **Phase 2**: unchanged (student processes all prompts using combined cached teacher logits)

Estimated **200-400 LoC** across `src/opd/opd_worker.py` + `src/opd/main_opd.py` + new config keys.

# Per-domain data + reward function still required

| Domain | Train data candidate | Eval data | Reward fn |
|---|---|---|---|
| Math | DAPO-Math-17k-dedup (already prepared) | MATH-500, AIME24/25 (already prepared) | `src/rewards/math_reward.py` (already exists) |
| Search | NQ + HotpotQA (Search-R1 paper's data) | same / 2WikiMultiHopQA | EM (exact match against retrieved answer); requires retrieval setup |
| Tool | ToolACE / Hammer / xLAM derived (ToolRL collection) | API-Bench / unseen tools | function-call execution + correctness |
| Code | APPS / LiveCodeBench / CodeContests | HumanEval+ / LiveCodeBench | unit-test execution (sandbox required) |
| Medical | MedQA-USMLE (10178 Q) + MedXpertQA | MedQA test split | MCQ answer match (boxed/letter); see open-medical-r1 `verify` for reference |
| Open-domain | RuscaRL training set (HealthBench / ResearchQA / RaR) | same | rubric-based judge model |

**Search/Tool/Code/Medical rewards are non-trivial** — each needs its own implementation. Most expensive: code (sandbox), search (retrieval), tool (execution).

# Pragmatic Stage 2 sequencing

1. **Stage 2 Iteration 1 (math + medical)**: math has full pipeline ready. Medical's MCQ reward is simple (boxed answer match). Validates multi-teacher routing on 2 simple domains.
2. **Stage 2 Iteration 2 (math + search)**: harder but Search-R1 paper has reward implementation.
3. **Stage 2 Iteration 3 (math + tool + code)**: most complex; tool & code rewards require execution infra.
4. **Stage 2 final (5+ domains)**: include open-domain teacher self-trained via RuscaRL.

# Hardware (unchanged)

- 8× A100-SXM4-80GB
- ~1.7 TB RAM
- 9.1 TB disk free
- Other Claude session (`MBRLLM/AgentGym-RL` GRPO) periodically uses GPUs 2-7 — coordinate or wait.

# Repo modifications carried over from Qwen3 era (still useful)

- `scripts/opd/train_opd.sh`: env-driven `ROLLOUT_NAME`, `OUTPUT_DIR`, `SKIP_PREPARE`, `GPUS_PER_NODE`
- `scripts/eval/eval_math.sh`, `scripts/grpo/train_grpo.sh`, `scripts/grpo/train_grpo_native.sh`: all support `ROLLOUT_NAME` env override
- `scripts/opd/run_smoke.sh`, `scripts/opd/run_stage1_b2.sh`, `scripts/grpo/run_stage1_b1.sh`: wrappers; need their MODEL_PATH/TEACHER_MODEL_PATH defaults updated for Qwen2.5

# Dep notes

- `ray` upgraded to 2.55.1 (verl 0.7.1 needs ≥ 2.41)
- `vllm` 0.19.0 (working, but cumem_allocator OOM after long sleep/wake cycles — observed at step ~14 of 162 with bs=256)
- `sglang` not installed — use vLLM (`ROLLOUT_NAME=vllm` patch)
- Tensor parallel TP_SIZE=2 fails on this machine (`custom_all_reduce.cuh:455 'invalid argument'`); use TP_SIZE=1 always

# Historical Qwen3-era data (informational only)

For reference — these numbers proved the OPSD codebase + hdong0 teacher worked end-to-end. They are NOT the project's anchor numbers post-switch.

| Bench | Qwen3-1.7B-Base raw | Qwen3-1.7B-Base + 1ep OPD (hdong0/Qwen3-8B-Base GRPO DAPO teacher) |
|---|---|---|
| MATH-500 mean@? | 50.0% (n=8) | 60.6% (n=16) |
| AIME24 mean@? | 2.1% (n=8) | 2.9% (n=16) |
| AIME25 mean@? | 2.9% (n=8) | 2.5% (n=16) |

Stage 0 smoke test ran for 2 steps in 4 min total wall, validating the full vLLM + FSDP+offload + reverse-KL OPD pipeline. Repo's `OPD_REWARD_BETA` hook (Stage 3 seed) confirmed present in `train_opd.sh` and `opd_trainer.yaml`.
