# OPSD project handoff

This repo is a **handoff package** for resuming a PhD research project on multi-teacher On-Policy Distillation (OPD) + RL on a new machine. If you are a Claude assistant freshly opened on the new machine: **read `memory/` first** — it contains the full project context, decisions made, and blueprint.

## What this project is

The user is a PhD researcher building an **open-source multi-teacher On-Policy Distillation (OPD) pipeline**, then layering an **OPD + RL hybrid** novelty on top. The codebase is `https://github.com/HJSang/OPSD_OnPolicyDistillation` (a minimal verl-based single-teacher OPD harness — needs extension for multi-teacher).

Inspirations / referenced prior work:
- DeepSeek V4, GLM-5, MiMo-V2-Flash, Qwen3 — all use OPD internally
- TIP paper (`arxiv 2604.14084`) and PACED paper (`arxiv 2603.11178`) — explicitly use the OPSD repo
- "Rethinking OPD" (`arxiv 2505.09388`) — thinking-pattern alignment finding

See `notes/zhihu.md` and `notes/zhihu2.md` for the user's curated background reading.
See `notes/phd_idea.md` for the user's original (terse) project plan.

## Where things are

| Path | What |
|---|---|
| **`memory/`** | Auto-loaded Claude memory on the prior machine. Contains the project blueprint, current stage, repo modifications, decisions made. **Start here.** |
| **`memory/project_opsd_blueprint.md`** | The locked-in "project constitution" — Qwen2.5-7B-Instruct unified base, per-domain teacher checkpoints, 4-stage plan |
| **`memory/project_opsd_stage_status.md`** | Live tracker of where we are right now |
| **`memory/feedback_autonomous_execution.md`** | User's preference: when they delegate and go offline, run end-to-end and persist state via memory + tasks |
| **`memory/user_role.md`** | User profile — research-peer level, prefers Chinese, deep familiarity with LLM RL |
| **`notes/phd_idea.md`** | User's original notes (very terse) |
| **`notes/zhihu*.md`** | Background reading — Chinese articles on OPD's role in post-training |
| **`opsd-patches/`** | Modified OPSD-repo scripts — see `opsd-patches/README.md` for how to apply |

## Where the project currently stands (2026-05-02)

**Recent strategic pivot**: switched from Qwen3-1.7B-Base student / single math teacher (hdong0) to **Qwen2.5-7B-Instruct unified-family** strategy because 5 of 6 domain teachers exist publicly on HF for that family. Stage 1 work on Qwen3 is now historical/archive only.

**Validated so far (Qwen3 era — proves OPSD codebase works)**:
- OPSD repo + verl + vLLM end-to-end pipeline runs cleanly
- Single-teacher OPD on math: Qwen3-1.7B-Base raw 50% → after 1-epoch OPD with Qwen3-8B-GRPO teacher = 60.6% on MATH-500 (mean@16)
- 200-400 LoC change identified for multi-teacher (none of it written yet)

**Pivot done (2026-05-02 evening)**:
- Project blueprint locked to Qwen2.5-7B-Instruct unified base
- 5 public teachers verified on HF (math/search/tool/code/medical-SFT)
- 1 still needs self-train (open-domain via RuscaRL, deferred)

**Immediate next actions on the new machine**:
1. Set up the OPSD repo + apply patches (`opsd-patches/README.md`)
2. Install deps (Python 3.10, verl, vllm, transformers, ray, tensordict, hf_transfer)
3. Download student `Qwen/Qwen2.5-1.5B-Instruct` + 5 teachers (~80 GB total; see blueprint)
4. Run Qwen2.5-era B0 (raw eval) and B2 (single-teacher OPD math) — these become the new project anchors
5. Implement multi-teacher routing in `src/opd/opd_worker.py` (see blueprint § "Stage 2 engineering still required")
6. Stage 2 Iteration 1: math + medical multi-teacher OPD

## Restoring memory on the new machine

The memory files in `memory/` were placed at `/home/ubuntu/.claude/projects/-home-ubuntu-OS/memory/` on the prior machine, where Claude auto-loaded them via the auto-memory feature. To restore:

**Option A — auto-memory path:** copy `memory/*.md` to `~/.claude/projects/<project-key>/memory/` on the new machine, where `<project-key>` matches your new working directory's slug. Claude will auto-load them.

**Option B — explicit context:** in your first message to the new Claude, point at `memory/MEMORY.md` and ask it to read everything in `memory/` to bootstrap context.

Either works. Option A is more automatic.

## Things that did NOT migrate

- **Model weights** (~32 GB Qwen3 stuff that's now archived) — re-download fresh Qwen2.5 models
- **Conda envs / pip installs** — recreate on new machine (see "deps notes" in blueprint)
- **Old training logs / output dirs** — historical only; the meaningful signal is summarized in memory
- **Background processes** — the prior machine had a parallel Claude session running on GPUs 2-7 with a different project (`MBRLLM/AgentGym-RL` GRPO). Not our concern now.

## Hardware assumed

- Ideally 8× A100-80GB or equivalent (the prior machine had this)
- ~1.7 TB RAM, ~9 TB disk free
- CUDA 12.x, Python 3.10
- If new machine differs significantly, re-evaluate batch sizes / TP_SIZE / gpu_memory_utilization in `opsd-patches/scripts/opd/train_opd.sh`

## Notes for the new Claude

The user has been driving decisions tightly through this conversation. They'll want you to:
- Surface decisions, don't decide unilaterally on substantive design choices
- Verify claims (especially HF model availability, base-model lineage) — don't trust upstream metadata blindly
- Persist state in memory + tasks so the next session doesn't have to re-discover anything
- Communicate concisely in Chinese; treat the user as a peer with deep LLM RL knowledge
