---
name: OS project (PhD idea drafting)
description: The /home/ubuntu/OS directory holds early PhD research planning for multi-teacher OPD + RL, not actual code
type: project
originSessionId: a4b9a9c8-cefb-4306-842b-75d91aaf5c8e
---
The `/home/ubuntu/OS` directory is NOT a code project — it is a planning/notes directory for the user's PhD research idea on multi-teacher On-Policy Distillation (OPD) + RL. Contents:
- `phd_idea.md` — short bullet-list of the research plan: reproduce multi-teacher OPD baseline (à la GLM-5 / MiMo), then add RL on top. Tasks planned: 3 LLM-based RL (math / code / medical, referencing ToolRL, Open-Medical-R1) + 2 Agentic RL (Search-R1, terminal-bench-rl) + 1 open-domain hard-to-verify (RuscaRL).
- `zhihu.md`, `zhihu2.md` — Chinese background articles on OPD's role in post-training (reverse KL, mode-seeking, catastrophic forgetting, multi-teacher fusion).

The actual codebase lives elsewhere: github.com/HJSang/OPSD_OnPolicyDistillation.

**Why:** User opened phd_idea.md and asked what the project is — confirming this directory is purely an idea-staging area, not where implementation happens.
**How to apply:** When the user works in this directory, treat tasks as research planning / writing / literature synthesis, not coding. For implementation work, expect them to switch to the OPSD repo.
