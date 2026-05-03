---
name: OPSD project — current stage and progress
description: Live snapshot of which research stage we're in, what's verified, what's next. Update after every meaningful step.
type: project
originSessionId: a4b9a9c8-cefb-4306-842b-75d91aaf5c8e
---
# Project base configuration (locked-in 2026-05-02)

- **Family**: Qwen2.5-7B-Instruct (everywhere)
- **Student**: `Qwen/Qwen2.5-1.5B-Instruct` (vocab 151936)
- **Teachers downloaded**: 5 public RL'd Qwen2.5-7B-Instruct ckpts at `/home/ubuntu/models/qwen2.5/teacher-{math-yukang, search-searchr1, tool-toolrl, code-svs, medical-umls}/` (all vocab 152064; truncated to student V at loss via `_align_vocab`)

# Plan trim (user-driven)

User removed Stage 1 B2; Stage 2 multi-teacher is the deliverable. Current sequence: setup → Stage 2 routing impl → Iter1 (math + medical) → analyze → Iter1.5 (with fixes) → Iter2 (more domains).

# Stage 2 multi-teacher routing (CODE COMPLETE)

Implementation in `src/opd/` (saved to `MOPD/opsd-patches/src/opd/`):
- `config/opd_trainer.yaml` — `actor_rollout_ref.opd_mt.teachers` schema (worker-visible)
- `losses.py` — `_align_vocab` helper (3 losses + 2 samplers)
- `batch_builder.py` — generic `_carry_non_tensor_keys` (data_source, uid, reward_model)
- `opd_trainer.py` — `_sort_opd_batch_by_data_source` does **chunk-interleave-by-teacher** (granularity=ubs, truncate to multiple of n_dp), `_pad_opd_batch_for_dispatch` propagates non_tensor, per-source reward dispatch via `opd.reward_fns`
- `opd_worker.py` — `init_model()` builds N-1 extra ref FSDP via `_build_model_optimizer`, contextmanager-protected tokenizer/processor save+restore, vocab compat assert, `_route_micro_batch` returns None on cross-teacher boundary mb (drop), Phase 1 iterates teachers in stable `self._teacher_modules` order, per-teacher metrics emitted with pre-populated zero buckets so verl's `reduce_metrics` works across DP ranks

Tests (CPU-only, all pass): `MOPD/test_vocab_align.py`, `MOPD/test_stage2_routing.py`.

# Stage 2 Iter1 RESULT (math + medical) — 2026-05-02 night

**Run**: `outputs/Qwen2.5-1.5B-Instruct-Stage2-Iter1-mathmed-mt-reverse_kl-lr1e-6-bs256/` (aborted at step 40)

**Pipeline functioned correctly** — multi-teacher routing engaged (math 7 mb/rank, medical 7 mb/rank, identical across ranks), per-domain reward dispatch worked, vocab align fired, FSDP collectives synced (after chunk-interleave + n_dp-multiple truncate fix).

**But the run COLLAPSED**:

| step | MATH-500 mean@4 | MedQA mean@4 | math_loss | medical_loss |
|---|---|---|---|---|
| 0 (baseline) | **0.5095** | **0.3337** | – | – |
| 20 | 0.4415 (−13%) | 0.3500 (+5%) | 35.1 | 0.33 |
| 40 | **0.0630** (−88%) 💥 | **0.1834** (−45%) 💥 | 11.6 | 0.14 |

**Root cause** (from step-40 generation dump, see `MOPD/logs/iter1_results.md`):
- Math GRPO teacher far from student (loss 36→11) dominates updates 80x over medical SFT teacher (loss 0.4→0.14)
- Reverse-KL is mode-seeking → on a far teacher with verbose CoT distribution, the **easiest local minimum is repetition** ("The distance is 3. The angle is 90°. The distance is 3. The angle is 90°. ...")
- Step 40 generations: mean output **7871 chars**, only 8.8% contain `\boxed{}`, 91% are stuck in literal repetition until max_response cuts off
- Entropy crashed 1.84 → 0.38 (degenerate confident policy), `tool_ratio` 89%→32% (CoT 6× longer)
- No on-policy reward signal in vanilla OPD → student wins teacher-likelihood while losing answer correctness

# Diagnosis update (2026-05-03 morning) — DEEPER ROOT CAUSE FOUND

**Original Iter1 hypothesis** (asymmetric multi-teacher signals) was **wrong**. Two new controls overturned it:

## Control A — pairwise KL(student || X) on 16 representative prompts (math + medqa)

| Teacher | KL on math | KL on medqa | Note |
|---|---:|---:|---|
| vanilla-7b-instruct | 0.10 | 0.39 | "size effect" baseline |
| **math-yukang** | **37.0** | **43.8** | **OUTLIER — 400× peers** |
| medical-umls (SFT) | 0.07 | 0.24 | nearly identical to student |
| search-searchr1 (GRPO) | 0.09 | 0.35 | nearly vanilla 7B |
| tool-toolrl (GRPO) | 0.10 | 0.38 | nearly vanilla 7B |
| code-svs (DAPO) | 0.10 | 0.37 | nearly vanilla 7B |
| FutureMa(merged LoRA r=8 GRPO) | 0.105 | 0.380 | nearly vanilla 7B; LoRA + 500-sample training too tame |

**math-yukang is alone in being far from the student.** Other RL teachers only changed task-specific output formats (`<search>`, tool calls, code blocks) — on neutral chat prompts they look like base 7B. Open-R1 long-CoT R1-style GRPO uniquely rewrote math-yukang's general distribution.

## Control B — single-teacher math-yukang OPD (no medical, all else identical to Iter1)

| step | MATH-500 mean@4 | math_loss | entropy | llm_tok | tool_ratio |
|---|---|---|---|---|---|
| 0 | 50.95% | – | – | – | – |
| 20 | **31.95% (−37%)** | 32.64 | 1.53 | 153K | 85% |
| 39 | **0.00% (−100%)** | **0.106** | **0.0089** | **1048576 (顶满)** | **0%** |

Single-teacher math-yukang **also collapses**, faster than Iter1 (step 30 vs step 40). loss converges to 0.1 (perfect teacher fit) but acc=0% (model completely broken). 同款 repetition fixed point.

**Conclusion: Iter1 collapse is caused by `math-yukang × current OPD recipe`, not multi-teacher imbalance.** Per-teacher loss normalization (originally fix #3) is irrelevant for math-only collapse.

# Active experiments (4 ablations)

| Exp | Variable changed | Hypothesis | Wrapper |
|---|---|---|---|
| **A** | reverse-KL → JSD | mode-seeking is the trap | `scripts/opd/exp_A_yukang_jsd.sh` |
| **B** | lr 1e-6 → 1e-7 | step size too large for far teacher | `scripts/opd/exp_B_yukang_lowlr.sh` |
| **C** | yukang → FutureMa(merged) | math-yukang specifically too far | `scripts/opd/exp_C_futurema.sh` |
| **D** | + opd.reward_beta=0.5 | vanilla OPD lacks reward signal | `scripts/opd/exp_D_yukang_rewardbeta.sh` |

Run order: C → D → A → B. Each ~25-30 min on 8 H100. C running as of 2026-05-03 05:30.

## Exp C RESULT (FutureMa(merged) + reverse-KL, 2026-05-03 06:06)

| | step 0 baseline | ExpC step 20 | **ExpC step 39 (final)** |
|---|---|---|---|
| MATH-500 mean@4 | 50.95% | 52.30% | **54.10% (+3.15)** |
| AIME24 | 1.67% | 2.50% | **2.50% (+0.83)** |
| AIME25 | 0.83% | 0.83% | **1.67% (+0.83)** |
| math_loss | – | 0.144 | 0.135 |
| llm_tok | – | 246K | 244K (vs Control B 1.05M) |

**Diagnosis confirmed**: math-yukang is the toxic teacher. With FutureMa swap (KL=0.105 vs yukang's 37.0), 1 epoch OPD gives mild but real improvements on all 3 math benchmarks, no collapse. Pipeline works correctly.

Caveats: FutureMa is a tiny LoRA r=8 trained on 500-sample NuminaMath-TIR — its distribution is barely different from vanilla 7B. +3.15% MATH-500 is likely cap of what such a "tame" teacher can teach.

## Exp D RESULT (yukang + reward_beta=0.5, 2026-05-03 06:50)

| step | MATH-500 | AIME24 | AIME25 | llm_tok | tool_ratio | n_correct |
|---|---|---|---|---|---|---|
| 0 | 50.95% | 1.67% | 0.83% | – | – | – |
| 20 | **28.0%** | 0.00% | 0.00% | 151K | 86% | 1/256 |
| 30 | – | – | – | 661K | 37% | 6/256 |
| 33 | – | – | – | **1048576** (顶满) | 0% | 0 |
| 39 | **0.0%** | **0.0%** | **0.0%** | 1048576 | 0% | 0 |

**reward_beta=0.5 failed to save yukang** — collapse trajectory ~5 steps behind Control B but reaches identical fixed point. Why:
- Binary 0/1 reward × bs=256 × early train acc 5-10% → most batches have 1-3 correct samples
- softmax(reward/0.5) gives correct samples weight ~5-7×, wrong samples ~0.7×
- 5× upweight on 1 sample out of 256 = ~2% gradient direction shift — too weak to overcome yukang's 80× signal magnitude
- At step 23+, when batches have 0 correct, reward_beta DEGENERATES TO UNIFORM (1.0/1.0) → effectively vanilla OPD → collapse continues
- **Implication for Stage 3 OPD+RL**: vanilla `opd.reward_beta` insufficient. Need either: (a) continuous/dense reward, (b) `rollout_n>1` to bootstrap reward variance per prompt, (c) much stronger `reward_beta` (e.g., 0.1) to sharpen the weighting

## Exp A (yukang + JSD) — ABORTED 2026-05-03 07:00 after critical finding

## 🚨 CRITICAL FINDING: math-yukang is a BROKEN checkpoint (2026-05-03 07:00)

User pointed at HF discussion `Yukang/Qwen2.5-7B-Open-R1-GRPO/discussions/2` where Helios03 reported:
> "The training trajectory of the model indicates that the model has failed. **During the actual use of the model for inference, it only outputs spaces.**"

### Verification
1. **CPU inference** (`MOPD/logs/verify_yukang.log`): on 3 different prompts (math, arithmetic, chat), yukang generates token id `220` repeatedly — that's the ASCII space token in Qwen2.5 tokenizer. **All outputs are pure spaces.**
2. **trainer_state.json analysis**: yukang's own GRPO training collapsed:
   - Step 1: reward=1.76, accuracy=14%, mean_completion=700 tokens, kl=0.0002
   - Step 5858 (final): reward=0.0, accuracy=0.0, mean_completion=2048 (顶满 max), min=max=2048 (zero variance), grad_norm=0.0, frac_reward_zero_std=1.0, learning_rate=0.0
3. **The saved ckpt IS the post-collapse model**. It's not "trained well but distribution-far"; it's literally a broken model that always outputs spaces.

### Implication for our Iter1 / Control B / ExpD failures
- The "math_loss=37" we measured was real but the cause was **degenerate teacher logits** (uniform-ish distribution that's "far" from any sensible model), not "GRPO domain-shift".
- Reverse-KL mode-seeking was a red herring; root cause was **distilling from a broken teacher**.
- Iter1.5 fix list (lr / loss-type / reward_beta) can't help — they all assume a valid teacher.

### Implication for path forward
- **ExpC (FutureMa swap) succeeded** because FutureMa is a coherent (if mild) teacher → confirms OPD pipeline is fine.
- **ExpA / ExpB / ExpD all chase non-existent root causes** — aborted ExpA after this finding.
- **Need new math teacher**: search HF for other Qwen2.5-7B math GRPO ckpts (with sanity-check inference before training!), or stick with FutureMa, or self-train (task #18 style).

### Lesson learned for future ckpt selection
**Always CPU-inference-test a teacher checkpoint before using it for distillation.** A few prompts × 30 tokens × CPU is cheap (~5 min for a 7B) and catches collapsed ckpts before wasting GPU hours.

User decision recorded: "如果最后都不想那么我们再考虑重新训练的事情" — if all 4 fail, only then consider self-training a teacher (task #18 open-medical-r1 GRPO style for math).

# Old Iter1.5 plan (DEPRECATED by Control A/B findings)

Per-teacher loss normalization (#3) and resume-from-step-20 (#1) no longer load-bearing fixes — they targeted the wrong root cause. Other fixes (lr, JSD, reward_beta, teacher swap) survive in current ABCD experiments.
7. **Reward-weighted distillation** via existing `opd.reward_beta` knob (couples OPD signal to ground-truth reward)

Combine #1+#2+#3+#4+#5 first; if still bad, escalate to #6 or #7.

# Iter1 saved artefacts

- `outputs/.../global_step_20/` (~19 GB) — best survivor ckpt
- `outputs/.../global_step_40/` (~19 GB) — collapsed ckpt
- `outputs/.../{20,40}.jsonl` — 7328 generations each
- `outputs/.../gpu_memory_monitor.csv` — 5s-resolution memory + util
- `MOPD/logs/iter1.log` — full training log
- `MOPD/logs/iter1_results.md` — detailed analysis with loss/metric trajectories + smoking-gun generation sample

# Open task: open-medical-r1 GRPO (#18)

Replace UMLS-SFT teacher with self-trained Qwen2.5-7B-Instruct + GRPO on MedQA/MedMCQA. Helps with the math/medical signal asymmetry that drove Iter1 collapse. Defer until Iter1.5 results clarify whether the SFT-teacher signal is salvageable with proper loss balancing.

# Important env caveats (carry forward)

- Always use absolute path `/home/ubuntu/miniconda3/envs/opsd/bin/{python,pip}` — `which python` resolves to conda BASE, not opsd
- Set `PYTHONNOUSERSITE=1` — `~/.local` has stale sklearn that breaks transformers; same for numpy/pandas
- `pip install` needs `--ignore-installed` if user-site has the pkg
- `hf download <repo>` for models, `--repo-type dataset` for datasets (else 401)
- vllm 0.19 pins torch 2.10.0 + transformers 4.56-5.x
- flash-attn 2.8.3 installed in opsd env (compiled from source ~3 min)
- For multi-teacher: dataloader random shuffling causes per-batch math/medqa imbalance (~bs/2 ± std) → trainer truncates to multiple of n_dp; with bs=256 truncation drops ~6-12% rows per batch, with bs=128 drops up to 25%. Larger bs is safer.

# Files of interest

- Handoff package `/home/ubuntu/MOPD/`:
  - Setup: `install_opsd_env.sh`, `download_models.sh`, `download_math_data.sh`
  - Iter1: `build_iter1_data.py` (medqa + math mix), `logs/iter1.log`, `logs/iter1_results.md`
  - Tests: `test_vocab_align.py`, `test_stage2_routing.py`
  - Patches: `opsd-patches/src/{opd/{worker,trainer,batch_builder,losses,config},rewards/medical_reward.py}/`, `opsd-patches/scripts/opd/{train_opd_mt.sh, run_stage2_smoke.sh, run_stage2_iter1.sh}`
- OPSD repo `/home/ubuntu/OPSD_OnPolicyDistillation/`:
  - `src/opd/` — all multi-teacher code merged
  - `src/rewards/medical_reward.py` — MCQ extractor (\boxed{}, "Answer:", final-letter fallbacks)
  - `data/iter1_mixed/` — 20356 mixed train + 4 val files
  - `scripts/opd/train_opd_mt.sh` — generic multi-teacher launcher (env-driven, uses `actor_rollout_ref.opd_mt`)
  - `scripts/opd/run_stage2_iter1.sh` — Iter1 wrapper with bs=256, val_n=4, math+medical
- Models: `/home/ubuntu/models/qwen2.5/`
- Conda env: `/home/ubuntu/miniconda3/envs/opsd/`

# Snapshot trigger

Update after each of: code merge (✓), smoke (✓), Iter1 result (✓), Iter1.5 result, Iter2 result, open-medical-r1 GRPO ckpt ready.
