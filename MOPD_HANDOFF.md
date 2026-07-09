# MOPD × Orthogonal Aggregation — Working Brief for Claude Code

> You are picking up a research project with **no prior context**. Read this whole file
> before doing anything. It contains the goal, the background, the exact codebase and where
> to modify it, a phased plan with **hard gates**, and the pitfalls. Follow the phase order.
> **Stop and report at every GATE — do not plow ahead.**

---

## TL;DR (read first)

- **Goal:** build multi-teacher on-policy distillation (MOPD) on top of the **KDFlow** repo, and
  research whether aggregating the per-domain training updates in a **geometry-aware way
  (direction + magnitude decoupled)** beats the standard **Euclidean weighted sum** that every
  frontier lab currently uses.
- **Work in PHASES with GATES.** Do **not** build the novel aggregation until (a) single-teacher
  distillation is stable (Gate A) and (b) cross-domain gradient conflict is *measured to be real*
  (Gate B). Building the fancy method before these gates is the main failure mode.
- **The metric that decides everything is `min-over-domains`** — the worst domain's recovery
  relative to its single-teacher ceiling — **not the average.** The classic failure is "average
  looks fine, one domain collapsed."
- **Isolate one variable at a time.** Keep any RL/reward weighting **off** (`reward_beta = 0`) for
  all of Phases 0–5 so we're only testing the *aggregation method*.

---

## 0. One-line description

Port the "aggregate on the manifold" idea — from Weiyang Liu's **OrthoMerge** (a model-*merging*
method), **upgraded to explicit direction + magnitude decoupling** — from post-hoc weight merging
**into the MOPD training loop's per-domain gradient aggregation.** So it acts on the *optimization*
(during training), not on frozen weights afterward. It is **not** model merging, and it is **not**
Muon.

---

## 1. Background you need (so you understand WHY each step exists)

**On-policy distillation (OPD).** A student model generates its own outputs (rollouts); a teacher
scores those student outputs (provides target distributions); the student minimizes a divergence
(typically **reverse-KL**) to the teacher on its *own* generations. This avoids the train/inference
distribution mismatch of old off-policy distillation. Origins: GKD (Agarwal 2023), MiniLLM (Gu
2024). The per-token KL can be treated as an advantage inside a GRPO-style RL loop.

**Multi-teacher OPD (MOPD).** Train `N` domain-expert teachers (math, code, medical, search, tool,
…), each on the **same backbone**, then distill all of them into **one** student via OPD so the
student becomes a generalist. This is the current standard "specialist consolidation" primitive
(DeepSeek-V4, MiMo-V2-Flash, GLM-5, Nemotron-Cascade 2, LongCat-2.0). **All of them route
per-prompt** (a math prompt is scored by the math teacher only — NOT all teachers scoring every
rollout) and handle conflict with full-vocab KL + token masking (IcePop) + tuned per-domain
weights. **None use a geometry-aware aggregation. That gap is the project.**

**The core problem: cross-domain gradient conflict ("see-saw").** Because one shared student is
pushed by many domains, the per-domain updates conflict. The labs sum them in Euclidean space:
`g_agg = Σ_i w_i g_i`. When directions conflict they partially cancel (triangle inequality:
`‖Σ_i g_i‖ ≤ Σ_i ‖g_i‖`) and the sum skews toward the largest/"loudest" teacher (often the one
whose distribution is farthest from the student). Result: one domain improves while others
regress. **This "conflicting directions cancel → magnitude collapses" is exactly the phenomenon
Weiyang Liu's OrthoMerge identifies for weight merging — we are moving it to gradients.**

**The method idea (the research core).** Instead of a Euclidean weighted sum, aggregate the
per-domain updates in a geometry-aware way. Weiyang's refinement (from a discussion — treat as the
target design): **decouple each update into direction + magnitude**, like polar/spherical
coordinates:
- **Direction** (lives on a sphere / orthogonal manifold) → aggregate with a **spherical /
  orthogonal-group** method that finds a *consensus direction* (so opposing directions don't cancel
  to zero, and hyperspherical energy is preserved).
- **Magnitude** (a scalar / length) → aggregate with a simple **linear / Task-Arithmetic-style**
  method (scalars don't collapse).
Then recombine into one update and apply it. Weiyang said this pipeline is plausible and **they
have tested the OFT-based version and it can work** — so the biggest implementation risk is
partially de-risked, but we still must reproduce and validate it ourselves.

**Muon is NOT this project's method, and NOT Weiyang's work.** Muon (Keller Jordan; scaled by
Moonshot) orthogonalizes a *single* gradient/momentum matrix's spectrum via Newton–Schulz — it's an
*optimizer* for single updates, unrelated to "how do multiple teachers combine." It may appear only
as an *optional* preprocessing brick (one way to turn a raw gradient into an orthogonal matrix), but
it is not the core and should not be presented as the contribution.

---

## 2. The codebase: KDFlow

**Repo:** `github.com/songmzhang/KDFlow` (an efficient, decoupled knowledge-distillation framework;
teacher on SGLang + student on FSDP2; transfers teacher hidden states, not full logits; supports
off-policy AND on-policy distillation; built-in algorithms include `vanilla_kd` and `dskd`;
divergences include reverse-KL / forward-KL / JSD / skewed-RKL; supports LoRA, multi-teacher).

**Why KDFlow (not our own fork):** we need a *known-good* engine so that "method doesn't work" is
never confounded with "engine has a bug." Our own opsd/verl fork could not get **single-teacher**
distillation stable, so we are restarting on KDFlow to remove that confound.

**What KDFlow's multi-teacher path actually does (verified by reading the code):**
- `examples/multi_teacher_distillation/` → `run_multi_teacher_on_policy_distillation.sh` calls
  `kdflow.cli.train_kd_on_policy` with `--multi_teacher_config`.
- `teacher_config.json` is just a `{routing_key: teacher_model_path}` map. Each data item carries a
  `teacher_routing_key`. **Mechanism = pure per-domain ROUTING:** each prompt/rollout is scored by
  its one routed teacher (`route_messages_to_teacher`; `_sort_opd_batch_by_data_source` chunk-
  interleaves; `_route_micro_batch` drops cross-teacher-boundary micro-batches; Phase-1 iterates
  teachers in a stable order).
- **Multi-teacher is restricted to `vanilla_kd`** (arguments raise otherwise).
- **There is NO per-teacher weighting, NO gradient aggregation logic, NO orthogonalization.** The
  combine step is: route → vanilla-KD loss per (routed) sample → normal backward on the mixed batch
  (i.e., a plain Euclidean sum, realized via gradient accumulation across per-domain micro-batches).
- **Sampler:** `cli/train_kd_on_policy.py` uses a plain `DistributedSampler(shuffle=True,
  drop_last=True)` over ALL prompts. No per-domain quota — each domain's share ≈ its share of the
  dataset. (So the per-domain "balance" knob is currently just *how much data of each domain you
  feed*.)
- **Batch/step args:** `rollout_batch_size`, `n_samples_per_prompt` (rollout args);
  `train_batch_size`, `micro_train_batch_size` (training args); `num_epochs`.
  Total update steps ≈ `num_epochs × |dataset| × n_samples_per_prompt / train_batch_size`.
  (You set epochs/data/batch, not a raw step count.)

**Where our modifications go (two places, both around the loss/gradient combine in the on-policy
trainer):**
1. **A per-teacher weighting baseline** (KDFlow lacks this — we must add it; it's the *real*
   opponent our method has to beat). Add per-domain loss coefficients (or a GradNorm-style
   normalizer) at the loss-combine step.
2. **The geometry-aware aggregation** (the method): replace the Euclidean sum of per-domain
   gradients with the direction/magnitude-decoupled aggregation. Because KDFlow routes (one teacher
   per micro-batch), the per-domain gradients naturally arrive as separate per-domain micro-batch
   gradients; the change is to **aggregate them explicitly before the optimizer step** instead of
   letting them accumulate as a plain sum.

**First, spend time locating and understanding this exact combine point** (the on-policy trainer's
per-domain loss computation and the gradient-accumulation/optimizer step). Everything we build hangs
off it.

---

## 3. Models, data, eval setup

- **Student:** `Qwen2.5-1.5B-Instruct`.
- **Teachers:** domain experts **all built on the same backbone** (`Qwen2.5-7B-Instruct` family),
  one per domain. Start with **2 teachers: math + medical**, then scale to ~5 (add code / search /
  tool).
- **AVOID pathological "far" teachers.** Before using any teacher, measure `KL(student ‖ teacher)`
  on a small neutral prompt set. A teacher that is a large outlier (e.g. a long-CoT R1-style GRPO
  math model that is ~100–400× farther than the others) tends to make reverse-KL **collapse into
  repetition** (this literally happened in our earlier attempt — single-teacher math collapsed
  MATH-500 from ~50% to <1% by degenerating into literal repetition). Prefer a more moderate math
  teacher, or handle it (see pitfalls).
- **Data:** one train split + one held-out split **per domain**. Ensure balanced per-domain batches
  (if KDFlow's plain shuffle under-samples a low-count domain, up-sample its data or add a
  stratified sampler — the earlier fork already had a working `StratifiedDataSourceSampler` for
  reference).
- **Eval harness MUST report per-domain numbers** (not a single aggregate). Fix a per-domain
  held-out benchmark for each domain (e.g. MATH-500 for math, MedQA for medical, LiveCodeBench for
  code, etc.). Record `mean@k` per domain.

---

## 4. The method to implement (this is Phase 4 — do NOT start here)

Target design (Weiyang-endorsed; some points are open, flagged in §8):

1. **Per-domain orthogonal adapter via OFT parameterization.** Instead of turning a raw gradient
   into an orthogonal object after the fact, give each domain `d` its **own OFT adapter** `R_d` on
   the shared student (OFT trains `W = R·W₀` with `W₀` frozen and `R` orthogonal, Cayley-
   parameterized `R = (I+Q)(I−Q)^{-1}`, `Q` skew-symmetric). The domain's OPD signal (its teacher's
   reverse-KL) updates **only its `R_d`**. Then each step you already have `K` orthogonal objects to
   aggregate — no need to force-orthogonalize a gradient, no Muon. Weiyang said the OFT-based version
   has been tested and can work.
2. **Decouple each per-domain update into direction + magnitude** (polar/spherical split).
3. **Aggregate the two parts separately:**
   - **Direction** → spherical / orthogonal-group aggregation (consensus direction; preserves
     hyperspherical energy). Candidate: OrthoMerge's Lie-algebra mean (map each rotation to its
     skew-symmetric `Q ∈ so(d)`, average in that flat space, map back via Cayley) — but per
     Weiyang, the cleaner framing is an explicit direction-on-sphere aggregation.
   - **Magnitude** → linear / **Task-Arithmetic-style** aggregation (TA/TIES-style; a scalar so it
     won't collapse).
4. **Recombine** into one update, apply to the student, next step.

Implementation guidance: **build the simplest version first** (e.g., plain per-domain gradient →
normalize to unit direction + scalar length → spherical mean of directions + TA-sum of lengths),
get it running and compared, *then* add the OFT-adapter parameterization and the OrthoMerge Lie-
algebra variant as ablations. Do not try to build the full novel stack in one shot.

---

## 5. Execution plan — PHASES with GATES (follow in order; report at each GATE)

**Phase 0 — Stand up KDFlow + reproduce the naive baseline (≈2–3 days)**
- Clone KDFlow, build the environment, get the **single-teacher on-policy** example running, then
  the **multi_teacher_distillation** example running.
- Wire up **per-domain eval**.
- Locate and document the multi-teacher loss/gradient combine point (where our two mods go).
- Deliverable: naive routed MOPD (math+medical) trains and evals per-domain.

**Phase 1 — Single-teacher alignment (do this FIRST; it's the main open worry) (≈1 week)**
- For each domain, distill from **one** teacher and confirm the student can **stably approach that
  teacher's ability on that domain without collapsing.**
- This directly answers "was our earlier failure the engine/recipe, or the method?": if
  single-teacher is stable on KDFlow (it wasn't on our fork), the engine was the problem.
- If a domain collapses (repetition / accuracy → 0 while loss → small), apply the pitfalls in §7
  (JSD instead of reverse-KL, lower lr, swap to a more moderate teacher).
- Record each domain's **single-teacher ceiling** (baseline for every later comparison).
- **GATE A — STOP & REPORT:** every domain aligns stably and you have its ceiling number. If a
  teacher can't be made stable, replace it. Do not proceed past A until all domains pass.

**Phase 2 — Naive multi-teacher + conflict diagnosis (the go/no-go gate) (≈3–5 days)**
- Run naive routed multi-teacher (math+medical, Euclidean sum). Measure:
  (a) how far below each domain's ceiling it falls (severity of see-saw);
  (b) **the actual gradient conflict:** per-domain gradient direction similarity (cosine), magnitude
  disparity, and conflict rate (fraction of steps with negative cosine between domain gradients).
- **GATE B — STOP & REPORT (most important gate):**
  - Clear negative cosine + magnitude imbalance + some domain suppressed → conflict is real →
    proceed.
  - Conflict is small / the tuned-weight baseline in Phase 3 already fixes it → **the method's delta
    is thin; stop and report.** In that case MOPD becomes a "compute + collaboration" vehicle, not a
    research contribution — do not sink weeks into the aggregation method.

**Phase 3 — Build the "tuned-weight" baseline (the REAL opponent) (≈1 week)**
- Add per-domain loss weights (loss-norm / GradNorm-style), tuned to best, plus KDFlow's full-vocab
  distillation. This is DeepSeek/MiMo's answer and is what our method must beat. **Do not compare
  our method against "no weighting at all" — that's a straw man.**
- Deliverable: best tuned-weight baseline numbers (mean **and** min-over-domains).

**Phase 4 — The method: OFT adapters + direction/magnitude decoupled aggregation (core, ≈2–3 weeks)**
- Implement per §4, simplest version first. Verify two sub-risks along the way: (i) does restricting
  each domain to an orthogonal adapter have enough capacity to actually learn the teacher; (ii)
  memory of `K` adapters (use OFTv2's memory-efficient form if needed).
- **Ask Weiyang for the details of their OFT small-test** (setup, scale, how direction vs magnitude
  were aggregated, which TA variant, any code/numbers) — this is the cheapest time-saver available.

**Phase 5 — Head-to-head + ablations + 2→5 teachers (≈2 weeks)**
- Main comparison: decoupled aggregation vs the Phase-3 tuned-weight baseline, judged on
  **min-over-domains**.
- Ablations: direction-only / magnitude-only / both; direction via OrthoMerge Lie-algebra vs direct
  spherical mean; magnitude via TA vs TIES vs a single scalar correction; OFT-adapter `R_d` vs
  Muon-style orthogonalized gradient.
- Scale 2 → 5 teachers.
- **GATE C — STOP & REPORT:** decoupled aggregation **clearly beats** the tuned-weight baseline on
  min-over-domains → real paper + open-source. If it only ties → the delta is thin; fall back to the
  vehicle framing.

**Phase 6 — Cleanup / open-source / paper** (only if Gate C passes): the first open-source
orthogonal-aggregation MOPD (fills a verified gap).

---

## 6. Metrics & logging (record these every run)

- **Per-domain held-out accuracy (`mean@k`)**, and each as a fraction of that domain's
  single-teacher ceiling.
- **`min-over-domains`** = the worst domain's fraction-of-ceiling. **This is the headline metric.**
- Training stability signals: output length, fraction of outputs hitting the length cap, entropy
  (watch for entropy collapse → repetition), whether accuracy drops while loss keeps dropping (the
  collapse signature).
- Conflict diagnostics (Phase 2+): per-domain gradient cosine, magnitude ratios, conflict rate.
- Always log the config that produced each number (teacher set, divergence, lr, weights, aggregation
  variant).

---

## 7. Hard rules & known pitfalls

- **Far-teacher reverse-KL collapse (we hit this).** A teacher far from the student can make
  reverse-KL collapse into repetition (loss → small, accuracy → 0). Mitigations: use **JSD** instead
  of pure reverse-KL, **lower the learning rate**, or **swap to a more moderate teacher**. Screen
  teachers by `KL(student ‖ teacher)` before use.
- **Do not skip the gates, and report at each one.** Gate B especially: if conflict isn't real or
  the tuned baseline already fixes it, the whole method phase isn't worth doing — surface that
  instead of proceeding.
- **`reward_beta = 0` for Phases 0–5.** Keep RL/reward weighting off so we test only the aggregation.
- **Compare against the *tuned* weighting baseline, not "no weighting."** (Multi-task gradient-
  surgery methods, e.g. PCGrad, have a reputation for only marginally beating well-tuned weighting —
  so the head-to-head must be honest.)
- **Metric is min-over-domains, not the mean.** Report both, but decide on min.
- **Build the simplest method version first**, then add OFT parameterization / OrthoMerge Lie-
  algebra / TIES as ablations. Don't build the full novel stack before it runs end-to-end once.
- **KDFlow's multi-teacher is routing + `vanilla_kd` only** — if you need a different divergence or
  per-teacher weighting, you're extending it; expect to modify the trainer, not just config.

---

## 8. Open questions — FLAG these, don't silently guess

- **How exactly to obtain a per-domain orthogonal object `R_d` from an OPD step** — OFT-adapter
  parameterization (preferred) vs Muon-style orthogonalization of the raw gradient. This is the one
  real design gap; Weiyang is advising on it.
- **Does OrthoMerge's magnitude correction transfer from weight-space to per-step gradient-space?**
  Proven for merging; unproven inside training.
- **Direction aggregation:** OrthoMerge's Lie-algebra mean vs a direct spherical (Stiefel/unit-
  sphere) mean — which is more natural/stable here.
- **Waiting on Weiyang's OFT small-test details** (setup/scale/aggregation/TA-variant/code). Use
  them once available; until then, use the simplest defaults and mark them provisional.

---

## 9. First actions for THIS session

1. Clone KDFlow, build the env, and run its **single-teacher on-policy** example end-to-end.
2. Run the **multi_teacher_distillation** example (2 teachers) and confirm per-domain eval works.
3. **Locate and document** the multi-teacher loss/gradient combine point in the on-policy trainer
   (this is where the weighting baseline and the aggregation method will plug in).
4. Pick **two moderate teachers (math + medical)** on the `Qwen2.5-7B-Instruct` backbone; screen them
   with `KL(student ‖ teacher)` and flag any far outlier.
5. Start **Phase 1 single-teacher alignment** for math and medical; report whether each aligns
   stably and its ceiling number. **Stop at Gate A and report before going further.**

*(Do not run any unrelated/business workloads on this cluster — it is for this research project only.)*
