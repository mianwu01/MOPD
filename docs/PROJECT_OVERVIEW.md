# MOPD × Orthogonal Aggregation — Project Overview

*Part I is for a reader new to the area. Part II is the technical breakdown of
every principle the method rests on. Part III maps principles to the code in
this repo.*

---

# Part I — The project, in plain language

## What problem are we solving?

Large language models can be **specialists** (one model that is great at math,
another great at medicine, another at coding) or **generalists**. Specialists
are easier to train — you optimize one skill without worrying about the others —
but nobody wants to deploy five models. So a standard industrial recipe is:

> Train N specialist "**teacher**" models, then compress all of their skills
> into one small "**student**" model.

The compression step is called **distillation**: the student learns to imitate
the teachers' output behavior rather than learning from raw data.

Our flavor is **multi-teacher on-policy distillation (MOPD)**:

- **On-policy** means the student writes its own answers, and the teacher then
  grades *those answers* token by token ("here's how likely I would have been
  to write each of your words"). This is better than imitating teacher-written
  text, because the student gets feedback on its own mistakes — the situations
  it will actually face at inference time.
- **Multi-teacher** means each training question is routed to the right
  specialist: math questions get graded by the math teacher, medical questions
  by the medical teacher, and so on.

## Where's the catch?

All the teachers' feedback ultimately updates **one shared student**. Every
step, the math feedback says "move the weights this way," the medical feedback
says "move that way," and the standard practice — used by essentially every
frontier lab — is to **just add the update vectors together** (possibly with
hand-tuned importance weights per domain).

Adding vectors can be lossy. If two updates point in conflicting directions,
their sum partially cancels; the louder domain dominates and quieter domains
can regress — a "see-saw" where math improves while coding quietly degrades.

## The idea

Weiyang Liu's group has a line of work on **orthogonal methods** for models:
instead of treating an update as a raw vector to be added, decompose it into

- a **direction** — *which way* to move, which lives on a curved space (think
  of directions as points on a sphere, or rotations), and
- a **magnitude** — *how far* to move, which is just a number.

Numbers can be averaged safely (no cancellation). Directions need to be
combined *on their curved space* — like averaging compass bearings properly
instead of averaging their x/y coordinates (averaging the coordinates of
north-east and north-west gives a shorter vector pointing north; averaging the
*bearings* gives a full-strength north). Their **OrthoMerge** paper showed this
matters when merging finished specialist models. **Our project migrates that
machinery from after-training merging into the MOPD training loop itself.**

So the plan, in one sentence:

> Train per-domain branches of the student with on-policy distillation, split
> each branch's update into direction + magnitude, combine the directions
> geometrically and the magnitudes linearly, and fold the result back into one
> student — and test whether this beats the well-tuned weighted sum.

## What makes this a *research* project rather than an engineering task

Two honest open questions, which the current experiment sweep is designed to
answer before we over-invest:

1. **Does the conflict actually occur?** Our own earlier 5-teacher experiment
   improved all five domains with a plain sum — no see-saw. The conflict may
   only appear with more genuinely specialized teachers, longer training, or a
   smaller student. We built instruments to *measure* interference directly
   (see Part II §6) instead of assuming it.
2. **Does the geometry earn its complexity?** The method must beat the *tuned*
   weighted sum — and simple controls like plain branch-averaging — under
   paired statistics, or it is not worth shipping.

---

# Part II — Technical principles

## 1. On-policy distillation (OPD) and reverse KL

Off-policy KD minimizes divergence on *teacher* (or dataset) text; the student
never gets feedback on its own trajectory distribution (exposure bias).
On-policy KD (GKD 2023, MiniLLM 2024) samples a rollout `y ~ π_s(·|x)` from the
student, then minimizes the **reverse KL** on those tokens:

```
L(x) = E_{y~π_s} Σ_t KL( π_s(·| x, y_<t) ‖ π_t(·| x, y_<t) )
     = E_{y~π_s} Σ_t Σ_v π_s(v) [log π_s(v) − log π_t(v)]
```

Reverse KL is **mode-seeking**: the student concentrates on regions the teacher
rates highly rather than smearing mass over the teacher's whole distribution —
usually what you want for a small student. Failure mode to watch: with a
badly-mismatched or broken teacher, the easiest way to "win" reverse KL can be
degenerate text (repetition, entropy collapse). We health-check every teacher
and monitor entropy/length during training.

`kd_ratio = 1.0` keeps the loss pure KD (no NLL mixing) so the aggregation
mechanism is the only variable under study.

## 2. Multi-teacher routing

Each prompt carries a `teacher_routing_key`; only that domain's teacher scores
the rollout (MiMo-style `π_domain`). Conflict therefore never happens *within*
a sample — it happens **at the shared parameters across samples/steps**, which
is exactly where the aggregation mechanism operates. In KDFlow the per-sample
teacher logits are computed by routing each sample's hidden states through its
teacher's `lm_head`, and all samples' token losses are summed into one
micro-batch loss — a **plain Euclidean sum** at the gradient level. That sum is
the baseline the project challenges.

## 3. Why per-step gradient geometry is a dead end (and what isn't)

Two facts kill naive "geometric gradient combination" at per-step scale:

- **Small-displacement collapse.** A per-step update is ~1e-4 relative
  displacement. Mapping such updates onto rotations puts them O(η) from the
  identity, and *any* Lie-group averaging agrees with plain Euclidean averaging
  to O(η²) — the geometry literally has nothing to act on. Norm-restoring
  scalars (×c) reduce to a learning-rate tweak that AdamW's scale invariance
  and gradient clipping absorb.
- **The √N null.** In d ≈ 10⁹ dimensions, independent gradients are nearly
  orthogonal, so ‖Σg‖ ≈ (1/√N)·Σ‖g‖ *without any conflict*. Norm shrinkage of
  the sum is not evidence of a problem; only **negative inner products beyond a
  noise-calibrated null** are.

What escapes both objections: **O(1) objects** — updates accumulated over K
steps (branch deltas), or orthogonal adapters trained in-loop (OFT rotations).
These are far from identity, low-noise, and the manifold operations on them are
non-degenerate. Hence the project's architecture is **branch → decompose →
merge**, not per-step surgery.

## 4. The decomposition: matrix polar coordinates

For any matrix `A` with thin SVD `A = U S Vᵀ`:

- the **nearest orthogonal matrix** to A (Frobenius) is the **polar factor**
  `Q = U Vᵀ` — this is what "the tool that finds the nearest orthogonal
  matrix" computes (exactly via SVD / orthogonal Procrustes, iteratively via
  Muon's **Newton–Schulz** iterations, or by construction via **OFT**);
- `A = Q · H` with `H = V S Vᵀ` symmetric positive semi-definite is the
  **polar decomposition** — the matrix generalization of `z = e^{iθ} · r`:
  Q is the "angle" (direction), H is the "radius" (magnitude);
- polar is **scale-invariant**: `polar(cA) = polar(A)` — direction extraction
  ignores how far the update went.

Three ways to get an orthogonal "direction" from a per-domain branch:

| Variant | Direction object | Magnitude object | Status |
|---|---|---|---|
| **V1: OFT branch** | the trained rotation `R_d` itself | (none — pure rotation) | clean math; capacity risk |
| **V2: ORD** (recommended) | `R_d = polar(W_d W₀ᵀ)` — nearest rotation from base to branch | residual `E_d = W_d − R_d W₀` | near-identity, det=+1 guaranteed |
| **V3: polar of delta** | `Q_d = polar(ΔW_d)` | SPD factor `H_d` | **landmine**: Q generically far from I, may have det=−1 → so(d) averaging undefined |

V2 is **OrthoMerge's Orthogonal-Residual Decoupling** applied to branch
checkpoints of the *student* — which also resolves the type problem that the
teachers (7B) and student (1.5B) share no parameter space: everything happens
in student space.

## 5. Merging: direction on the manifold, magnitude linearly

**Direction channel.** Rotations near the identity live in the Lie group SO(n);
their logarithms (here via the **Cayley transform**, an involution):

```
Q = (I − R)(I + R)⁻¹   (skew-symmetric ⟺ R ∈ SO(n), no −1 eigenvalue)
R = (I − Q)(I + Q)⁻¹
```

live in the flat vector space so(n) of skew-symmetric matrices, where averaging
is legitimate. OrthoMerge's recipe, which we implement verbatim:

```
Q_merged = c · Σ_d w_d Q_d,     c = (Σ_d w_d ‖Q_d‖) / ‖Σ_d w_d Q_d‖
R_merged = Cayley(Q_merged)
```

The **×c factor restores the norm** lost to cancellation between conflicting
rotations — the compass-bearing fix from Part I, and the single most important
ingredient in OrthoMerge's ablations. Note c has real content here because the
Q_d are O(1); on per-step gradients it would be an inert lr rescale (§3).

**Magnitude channel.** Scalars/residuals add safely:

- **Task Arithmetic (TA)**: `E_merged = Σ_d λ_d E_d` — a weighted sum with
  tunable per-domain λ (per-domain *normalization* handles heterogeneous
  teacher scales; that job must not be conflated with ×c, which fixes
  direction-channel cancellation).
- **TIES** as the interference-aware alternative: per entry, *trim* small
  values, *elect* the majority sign, average only the sign-agreeing entries —
  a discrete anti-cancellation mechanism for the residual channel.

**Rectangular matrices.** so(n) needs square R. ORD keeps R square by placing
it on the smaller side (`W ≈ R W₀` if out ≤ in, else `W ≈ W₀ R`) — the same
form OFT uses — so embeddings/`lm_head`/MLP projections all stay tractable.

## 6. Measuring conflict properly (the corrected Gate B)

Interference is a claim about **inner products of accumulated deltas**, judged
against noise:

- `cos(Δ_a, Δ_b)` per parameter group (embed / lm_head / attn / mlp / norm) —
  conflict is expected to localize in format-carrying parameters if it exists;
- `ρ = ‖ΣΔ‖ / Σ‖Δ‖` compared against the **no-conflict null 1/√N** (not 1);
- **split-half reliability** `r_d = cos(Δ_d^{halfA}, Δ_d^{halfB})`: if two
  halves of the *same* domain's training don't agree with each other, the
  deltas are noise at this K and no cross-domain conclusion is valid;
- **disattenuated alignment** `A_ab = cos(Δ_a, Δ_b)/√(r_a r_b)` — the
  measurement-error-corrected conflict statistic.

## 7. Experiment design principles

- **Ceilings at matched budget**: a domain's "100%" is its single-teacher
  branch trained with the same per-domain data/steps the multi-domain run gets.
- **min-over-domains** decides, but only under **paired per-problem statistics**
  (difference-in-differences, bootstrap CI, permutation test) — unpaired means
  on 164-problem suites cannot resolve 2pp effects.
- **Controls that keep any win attributable**: plain delta averaging (is it
  geometry, or just branch-training?), v2 without ×c (is it the norm
  restoration?), direction-only / magnitude-only (which channel?), TA/TIES
  (is a linear method enough?), and the tuned weighted-sum baseline (is
  in-loop weighting already sufficient?).
- **Teacher hygiene**: health-check each checkpoint (a broken teacher destroyed
  an early experiment); score teachers with their **native chat templates**;
  never ask a teacher to score past its trained context window (Qwen2.5-Math:
  4,096 positions — enforced in-loss by our `--teacher_max_len` patch).

## 8. Open forks (to resolve with Weiyang)

1. **V2 (ORD on branch weights) vs V3 (polar of deltas)** — V3's direction
   factors are generically outside the Cayley domain (det=−1 possible); we run
   it in the sweep guarded, to quantify rather than assume.
2. **Merge cadence**: merge once at branch-end (pure post-hoc) vs iterated
   K-step merge-and-rebranch (genuinely in-loop; keeps branches mergeable and
   the merged model near on-policy). Snapshots at K=20/40/60/80 let us sweep
   this offline first.
3. **OFT capacity (Gate A′)**: can a 1.5B student absorb a domain through
   spectrum-preserving rotations alone? The OFT-vs-full-param branch pairs
   answer this directly.
4. **FDA's role**: with branches in student space, FDA-style anchors become
   coherent (synthesize inputs whose induced gradient aligns with a branch's
   delta; use them as post-merge adaptation or suppressed-domain replay) — a
   later arm, pending the bake-off.

---

# Part III — Where everything lives

| Piece | Location |
|---|---|
| Merge library (polar, Cayley, ORD, all operators, diagnostics) | `mopd_merge/` |
| Unit tests (38 checks, CPU) | `test_merge_lib.py` |
| Offline merge bake-off CLI | `sweep/offline_merge_bakeoff.py` |
| Interference diagnostics CLI | `sweep/delta_diagnostics.py` |
| Paired per-problem statistics | `sweep/paired_compare.py` |
| PEFT→full checkpoint materializer | `sweep/materialize_peft.py` |
| Data builder (KDFlow JSONL + split halves) | `sweep/build_kdflow_data.py` |
| Branch / multi-teacher launchers | `sweep/train_branch.sh`, `sweep/train_multiteacher.sh` |
| Teacher roster + context caps | `sweep/configs/` |
| Execution plan + decision gates | `sweep/RUNBOOK.md` |
| KDFlow patches: per-teacher loss weights, teacher context cap, OFT student | `KDFlow/kdflow/{arguments,algorithms/vanilla_kd.py,loss/chunked_loss.py,models/model.py}` |
| Research log: assessment + addenda (collapse proof, landmines, gates) | `NOTES_independent_assessment.md` |
| KDFlow internals map | `NOTES_kdflow_combine_point.md` |

Key external references: GKD (Agarwal et al. 2023); MiniLLM (Gu et al. 2024);
OFT (Qiu et al., NeurIPS 2023) / OFTv2 (2506.19847); OrthoMerge (2602.05943);
FDA (2510.21223); AMO/Muon (2605.17806); Task Arithmetic (Ilharco et al. 2022);
TIES (Yadav et al. 2023); gradient-surgery skepticism: Kurin et al. 2022
(2201.04122), Xin et al. 2022 (2209.11379); CaMOPD (2605.27115).
