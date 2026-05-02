---
name: User wants autonomous execution with memory + todo tracking
description: When user delegates and goes offline, run plans end-to-end without checking back, persist state in memory + tasks
type: feedback
originSessionId: a4b9a9c8-cefb-4306-842b-75d91aaf5c8e
---
User explicitly wants autonomous execution when they delegate work and go away. Specifically:
1. Maintain memory and update by stage
2. Maintain a todo list (TaskCreate / TaskUpdate)
3. "严格执行" (strictly execute) — don't pause for approval mid-stream

**Why:** User said this on 2026-05-01 before sleeping, while delegating Stage 0 smoke test of the OPD project. Wants to wake up to results, not to a pile of pending questions.

**How to apply:**
- For long autonomous sessions, set up a TaskList up front with all known steps; mark in_progress / completed as I go.
- Update the relevant `project_*_stage_status.md` memory file at every stage boundary (not just at the end), so a future session can pick up exactly where this one left off.
- When I hit a fixable issue (missing dep, wrong file path, schema mismatch), fix it forward rather than stopping. Document the fix in memory.
- When I hit a non-fixable blocker (would need user input on a substantive design call), stop and write the question into the stage-status memory under an "Open questions / blocked" header, so user sees it on wake.
- Risky/destructive actions (force-push, deleting non-trivial work) still require pause — autonomy doesn't override the "executing actions with care" guidance.
- Final memory snapshot before sleep should include: what's done, current state, exact next-step command, and any decisions left for the user.
