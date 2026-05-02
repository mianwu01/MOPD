---
name: OPSD project — current stage and progress
description: Live snapshot of which research stage we're in, what's verified, what's next. Update after every meaningful step.
type: project
originSessionId: a4b9a9c8-cefb-4306-842b-75d91aaf5c8e
---
# Project base configuration (locked-in 2026-05-02)

See `project_opsd_blueprint.md` for the full blueprint. TL;DR:

- **Family**: Qwen2.5-7B-Instruct (everywhere)
- **Student**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Teachers**: 5 public RL'd Qwen2.5-7B-Instruct ckpts (math/search/tool/code/medical-SFT) + 1 self-trained (open-domain RuscaRL, deferred)

# Current stage: pre-Stage 1 redo (Qwen2.5 era)

**Status as of 2026-05-02**: prior Stage 1 work was on Qwen3 family, now superseded. Need to redo Stage 1 on Qwen2.5 family + then proceed to Stage 2.

# Immediate next actions (in order)

1. **Download new student + teachers** (CPU/network only; no GPU contention). Plan:
   - `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB) — student
   - `Yukang/Qwen2.5-7B-Open-R1-GRPO` (~15 GB) — math teacher
   - `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3` (~15 GB) — search teacher
   - `emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold` (~15 GB) — tool teacher
   - `RLVR-SvS/SvS-Qwen-Code-7B` (~15 GB) — code teacher
   - `prithivMLmods/Qwen-UMLS-7B-Instruct` (~15 GB) — medical teacher (SFT)
   - **Total ~80 GB**. Plenty of disk (9 TB free).

2. **Redo Stage 1 B0** (Qwen2.5-1.5B-Instruct raw eval on MATH-500 / AIME24 / AIME25) once GPUs free up. ~30-60 min.

3. **Redo Stage 1 B2** (single-teacher OPD: Qwen2.5-1.5B-Instruct student + Yukang math teacher, 1 epoch DAPO-Math-17k). ~2-3h on 8 GPUs.

4. **Stage 2 implementation** (multi-teacher routing in OPDWorker, 200-400 LoC). Can start anytime — pure code work, no GPU.

5. **Stage 2 Iteration 1** (math + medical, 2 teachers, simplest reward functions). First multi-teacher run.

# Open blockers

- Other Claude session running on GPUs 2-7 (`MBRLLM/AgentGym-RL` GRPO, started 01:24, still active hours later). For Stage 1 redo, can use GPUs 0-1 with TP_SIZE=1 + smaller batch — but expect ~10× slower than 8-GPU. Or wait for it to finish.

# Risks (current strategy)

- **Tokenizer compat across teachers**: must verify all 5 teachers share Qwen2.5-7B vocab (152064). Only blocker if any teacher accidentally has resized embeddings — easy to test once downloaded.
- **Medical teacher is SFT not GRPO**: weaker per-token signal than peer teachers. Acceptable for paper if framed as "open-source ecosystem reality"; or replace later with self-trained GRPO version.
- **Search/Tool/Code reward implementations**: each requires non-trivial setup (retrieval / sandbox). Stage 2 Iteration 1 picking math+medical avoids this initially.
- **Open-domain RuscaRL teacher**: must self-train. ~1-2 weeks of work and compute when we get there. Defer.

# Repo modifications standing (carried over)

In `/home/ubuntu/OPSD_OnPolicyDistillation`:

- `scripts/opd/train_opd.sh`: env-driven `ROLLOUT_NAME`, `OUTPUT_DIR`, `SKIP_PREPARE`, `GPUS_PER_NODE`. Keep these patches.
- `scripts/eval/eval_math.sh`, `scripts/grpo/train_grpo*.sh`: `ROLLOUT_NAME=${ROLLOUT_NAME:-sglang}` patch. Keep.
- New wrapper scripts (`run_smoke.sh`, `run_stage1_b1.sh`, `run_stage1_b2.sh`): need MODEL_PATH/TEACHER_MODEL_PATH defaults updated to Qwen2.5 paths once downloaded.

# Snapshot trigger

Update this file after each of: downloads complete; B0 redo done; B2 redo done; Stage 2 multi-teacher code merged; first Stage 2 trial run done.
