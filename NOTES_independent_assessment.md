# Independent assessment of the MOPD × Orthogonal Aggregation plan (Handoff Brief v2)

*2026-07-10. Produced by a 25-agent adversarial review (6 independent analysis lenses →
18 refutation attempts on every kill/major claim → completeness critic), grounded in this repo's
git history, the vendored KDFlow source, and reachable literature. Verdicts below are marked
CONFIRMED only where a dedicated adversarial verifier tried and failed to break the claim.*

## Verdict

**Do not build the method as specified. Reframe before spending GPU-weeks.** The brief's two
concrete builds fail for independent reasons — Route α (per-step gradient geometry) is
*mathematically vacuous at this step size*, and Route β (FDA anchors) *fails to type-check for a
1.5B student with 7B teachers* — and, more fundamentally, the phenomenon the whole project exists
to fix (**the multi-domain see-saw**) has **never been observed in this project's own experiments
with healthy teachers**. The salvageable core is real but different from the brief: an
**outer-loop branch-distill-merge architecture** where the geometry acts on K-step per-domain
deltas (the only object where OrthoMerge's math is non-degenerate), wrapped in a reframed,
diagnosis-first paper. Details and a revised plan below.

---

## Finding 1 (kill, CONFIRMED): the see-saw has never been observed here — and the brief omits the strongest counter-evidence

- The **Iter2 5-teacher naive Euclidean-sum baseline** (commit `b257602`, "paper-grade 80-step
  run") improved **all five domains simultaneously**: MATH-500 +2.3, MedQA +7.5, HumanEval+ +1.2,
  Tool-Star +14.3, SearchR1 +4.3. No domain regressed below baseline. The random-sampler ablation
  (`28e508b`) corroborates — batching composition didn't even matter.
- The only collapse in project history (Iter1) was root-caused to a **broken checkpoint**
  (space-emitting yukang), and it drove *both* domains down — not a see-saw.
- The only regressions ever observed with healthy teachers were **intra-domain** (multi-hop search
  declining because the student over-searches, commit `5b894f1`) — a behavior/reward pathology no
  cross-domain aggregation scheme touches.
- **Critically: the brief presents Qwen2.5-Math-7B-Instruct and Huatuo-o1 as "new teachers" whose
  Gate-B screen "will tell us whether real conflict exists" — but these exact teachers were
  already in the all-positive Iter2 run** (`teacher-math-math`, `teacher-medical-huatuo`, commits
  `cc55efe`/`b257602`). The de facto Gate-B experiment for this roster has already run, and it
  read "no conflict."
- Literature prior points the same way: tuned scalarization is notoriously hard to beat with
  gradient surgery (Kurin et al. 2022, arXiv 2201.04122; Xin et al., arXiv 2209.11379).

Caveat that keeps this short of a definitive kill: 80 steps may be too short for near-saturation
interference, evals are single-seed, and the KL screen has never actually been computed for the
current roster. But the burden of proof has flipped: **demonstrating that the see-saw exists at
all is now the project's first deliverable, not an assumed premise.**

## Finding 2 (kill, CONFIRMED — numerically verified): per-step "geometric" aggregation collapses to the tuned baseline

At KDFlow's operating point (lr 2e-6, AdamW betas (0.9, 0.98), `clip_grad_norm_(1.0)` in
`fsdp_strategy.py:352-367`), per-step updates are ~1e-4 relative displacement. At that scale:

- **Lie-algebra / Cayley consensus = the Euclidean average.** Mapping per-step updates to
  rotations puts them O(η) from identity; averaging in so(d) and mapping back reproduces the
  Euclidean average to O(η²) (~1e-8 relative — verified numerically by the review, to machine
  precision for the inverse-Cayley→average→Cayley pipeline). OrthoMerge's math has content
  because *post-hoc task vectors are O(1) displacements*; per-step gradients are not.
- **Direction/magnitude decoupling = a scalar-weighted normalized sum.** For any N, the spherical
  (even intrinsic Karcher) consensus direction is a positive weighted sum Σwᵢ·gᵢ/‖gᵢ‖ — exactly
  the family the Phase-3 GradNorm-style tuned baseline optimizes over. The entire method-vs-
  baseline delta is a scalar.
- **That scalar is then destroyed downstream**: AdamW is invariant to constant gradient rescaling
  (exact here, weight_decay=0), and the max_norm=1.0 clip deletes any upscaling whenever active.
  The ×c correction ≈ √N ≈ 2.24 with tiny fluctuations (concentration in d≈1.5e9).
- **The motivating "cancellation" is generic high-dimensional addition, not pathology**:
  ‖Σgᵢ‖ ≪ Σ‖gᵢ‖ with c≈√N is the *zero-conflict default* for near-orthogonal vectors. Conflict
  is a sign statement about inner products against a Pythagorean null — the triangle-inequality
  framing in brief §2 cannot distinguish the two.
- Per-domain Newton–Schulz orthogonalization before summing is *spectral preconditioning*, not
  conflict resolution: polar(−G) = −polar(G), so perfectly conflicting gradients stay perfectly
  conflicting. Including AMO also creates an **optimizer confound** — any win becomes "Muon helps
  OPD," not "geometry-aware aggregation works" — unless an AMO-on-the-naive-sum ablation is run.

Scope note: this covers the per-step aggregation as specified in brief §4 Route α. Variants that
accumulate O(1) displacements (outer-loop deltas, per-domain OFT adapters merged every K steps)
escape the collapse — that is exactly why the salvage (below) is outer-loop.

## Finding 3 (kill, CONFIRMED): Route β fails to type-check, and the roster breaks the shared-base premise anyway

- FDA constructs anchors whose induced gradient at θ₀ aligns with τᵢ = θ_ft − θ₀ — this is defined
  only within **one shared parameter space**. Teachers are 7B; the student is 1.5B. "Represent
  each teacher as anchors, jointly train the student on them" cannot be built as written.
- Even within 7B space, **Qwen2.5-Math-7B-Instruct is not a finetune of Qwen2.5-7B-Instruct** — it
  sits on the Qwen2.5-Math continue-pretrained base with different RoPE geometry (rope_theta 10k,
  4,096 positions vs 1M/32,768). τ_math against the Instruct backbone is continued-pretraining
  drift plus positional-geometry mismatch, not a task vector. The brief's "same backbone is a hard
  requirement" is violated by its own roster (for teacher-space operations; Route α/δ only need
  vocab-compatible functional access, which holds).
- **Actionable bug found in passing:** all existing recipes use MAX_PROMPT 2048 + MAX_RESPONSE
  4096 = up to 6,144 tokens, but the math teacher has 4,096 trained positions — every long rollout
  makes the teacher score tokens beyond its RoPE window (extrapolation garbage feeding the loss
  and every "far teacher" measurement). Fix before trusting any math-domain number.
- Coherent salvages of the FDA idea: (i) FDA-merge teachers at 7B → single-teacher OPD (loses the
  in-loop story); (ii) reinterpret anchors as **discrepancy-mined probe prompts** — per-domain
  synthetic prompts maximizing student-teacher disagreement, used as a cheap forward-only
  interference diagnostic and as targeted rollout augmentation for a suppressed domain
  (an anti-forgetting controller). Variant (ii) is the most tractable and pays off even if
  cancellation never materializes.

## Finding 4 (CONFIRMED): Gate B measures the wrong quantity, in both directions — and is currently unfalsifiable

- Neutral-prompt KL(student‖teacher) is dominated by the 1.5B→7B capacity gap plus prompt type
  (~0.10 math / ~0.3 medqa for *every* healthy teacher incl. vanilla 7B). It cannot separate
  specialized from unspecialized teachers, let alone conflicting from non-conflicting ones —
  conflict is a *pairwise property of gradients at the shared student*, which no per-teacher
  output-distribution scalar can see.
- **Measurement artifact (critic finding):** teachers are scored off-template — generic chat
  template, generic "think step by step…\boxed{}" instruction, no teacher-specific system prompts
  (`control_a_new_teachers.py`, `build_5domain_mixed.py`) — while Qwen2.5-Math's and Huatuo-o1's
  specializations live behind their own system prompts/templates. "Teachers look like vanilla 7B"
  is partly an elicitation artifact, not only a fact about the teachers.
- Per-micro-batch gradient cosines (the planned Phase-2 instrument) are attenuated toward 0 by
  gradient noise (SNR « 1 at ~26 samples/domain/step) — a threshold on raw per-step cosines is
  vacuous regardless of the truth.
- **The instrument has never fired on anything.** If Gate B reads "no conflict," that cannot
  currently be distinguished from "extraction is broken."

Redesign (verified as the right shape): (1) measure on **domain-pure accumulated/EMA gradients**
with a split-half within-domain null and noise-disattenuated alignment; (2) add the **cross-domain
transfer matrix** — train each domain solo for K steps, track every other domain's eval; (3) run a
**forced-conflict positive control first** (shared low-rank adapter + imbalanced mixture + higher
lr on a format-conflicting pair) to prove the instruments fire before trusting any "stop" reading;
(4) fix elicitation and the 4k-context bug before measuring anything.

## Finding 5 (kill, CONFIRMED): min-over-domains is not measurable at the planned compute

The min lands on the noisiest domain (HumanEval+, n=164, single-run SE ≈ 3.9pp); the min-statistic
is biased low with variance set by the worst domain; the observed baseline delta on that domain
(+1.2pp) is within noise. Detecting a ~2pp min-domain effect unpaired needs tens of seeds. Fixes:
drop AIME24/25 from the decision metric (0–2.5% for a 1.5B student — no resolution); enlarge code
eval (MBPP+ + LiveCodeBench, ~700 problems); use **paired per-problem difference-in-differences**
on identical eval sets; pre-register the effect size (the tuned baseline's min-over-domains deficit
vs ceilings must exceed ~3pp before Phase 4 is even built — otherwise there is provably nothing to
recover). Also: ceilings must be defined at **matched per-domain data budget**, or "recovery"
conflates interference with data starvation.

## Finding 6 (CONFIRMED): the novelty flag is half-planted

- **CaMOPD (arXiv 2605.27115)** already diagnoses gradient counteraction in multi-teacher OPD
  (general-recovery vs domain-preservation axis) and fixes it non-geometrically with decoupled
  alternating updates. It partially preempts the problem-identification claim and is also the
  cheapest conflict-mitigation baseline to implement — it must be read and included.
- The outer-loop variant's conceptual neighbors are ColD Fusion, DIMAT, MERIT (iterated
  train-merge). "First geometry-aware aggregation *inside the MOPD loop*" survives, but only as an
  empirical wedge, not a conceptual one. A "first X" framing invites a one-line novelty rejection.
- Two of the three toolkit papers (AMO, OrthoMerge) could not be fetched full-text from this
  environment; equation-level claims about them rest on the brief's paraphrases and should be
  re-verified against the PDFs before anything is built on them.

## Finding 7 (critic, unmined by the brief): specialization may not transfer into a 1.5B student at all

Commit `dff618b`'s v10 single-teacher series shows the genuinely-specialized Math-7B teacher gave
**no more student gain than vanilla Qwen2.5-7B-Instruct** (v10b peak 0.534@20 → 0.527@39; vanilla
7B 0.537; tame FutureMa 0.541; baseline ~0.51), with alignment KL saturating at a
teacher-distance-dependent floor. If teacher specialization doesn't transfer at this
student scale, **even perfect conflict resolution has near-zero headroom**, and the ceiling
denominators of min-over-domains are uninformative. This bounds the upside of *any* aggregation
method and needs to be understood (elicitation fix + 4k-context fix may move it) before the
project invests further. Note also v10b's unexplained peak-then-decline with a healthy far teacher
— the one whiff of a real reverse-KL pathology in the repo; the v10c/v10d mitigation results are
unrecorded and worth recovering.

Also flagged: **engine-migration comparability hole** — every anchor number (Iter2, ceilings,
Control A) is from the verl/OPSD+vLLM stack, while Phases 0–5 run on KDFlow+sglang with different
loss defaults (`kd_ratio=0.5` mixes an NLL term; `fkl` default divergence; per-sample re-merged
micro-batches). Baselines must be regenerated on KDFlow or the stacks bridged before any
cross-stack comparison is made.

---

## What survives, and what I would build

**The defensible architecture is Route δ (absent from the brief): outer-loop branch-distill-merge
in student space.** Each round: snapshot the student; run each domain's OPD branch for K steps
(domain-pure, persistent per-domain Adam state); treat the per-domain deltas Δθᵢ — genuine O(1),
low-noise, task-vector-like objects *of the 1.5B student* — as the things to aggregate
(per-matrix direction consensus + norm restoration / TA / TIES); apply through an outer step;
re-branch. This is the only home where OrthoMerge's math is non-degenerate, it plugs in *above*
KDFlow's combine point (sidestepping the non-domain-pure micro-batch problem entirely, and most of
the FSDP surgery), it reuses Phase-1 single-teacher branches for free, and it keeps rollouts
on-policy per branch. Honest accounting: its conceptual neighbors exist (ColD Fusion/DIMAT/MERIT),
so the claim is empirical; and the make-or-break ablation is **plain delta averaging (FedAvg)** —
if that matches the geometric merge, the win belongs to domain-pure local phases, not geometry.

**The reframed paper** (higher expected value than the method paper): *"When does multi-teacher
on-policy distillation need more than a weighted sum?"* — built on the axis this team uniquely
controls: teacher distance (near-base RLVR teachers → continue-pretrained specialists), with the
conflict instrumentation as the diagnostic contribution and the outer-loop merge as the
intervention arm. The already-in-hand negative result ("RLVR teachers are distributionally
near-base; naive sums suffice at this scale") touches a live debate and is publishable *with* the
scaled-up diagnosis; it is not publishable as a bare null.

**Revised 4-week sequence** (replaces brief §7 Phases 0–2; pre-registered kill criteria):

1. **Fix the measurement layer first**: teacher-native chat templates/system prompts everywhere;
   cap math-teacher-scored sequences at 4,096 or swap in an Instruct-base math RL checkpoint;
   regenerate the KL screen for the *current* roster (never actually committed).
2. **Phase-1 branches double as the experiment**: single-teacher OPD per domain on KDFlow (also
   regenerates ceilings on the new stack, at matched per-domain budget). From these branches the
   **delta-interference analysis is free**: pairwise cos(Δθᵢ, Δθⱼ) per matrix, interference ratio,
   cross-domain transfer matrix.
3. **One-day offline merge bake-off** (tests the entire toolkit before any in-loop build): merge
   the per-domain deltas with plain average / TA / TIES / OrthoMerge-style decoupled merge;
   evaluate. If geometric merging cannot beat plain averaging *offline*, it will not do so
   in-loop — hard kill for the geometry arm.
4. **Forced-conflict positive control** (validates the instruments): shared low-rank adapter +
   imbalanced mixture + higher lr on the most format-conflicting pair; confirm the metrics fire
   and a real see-saw appears; then run identical instrumentation on the natural setting.
5. **Gates**: build Phase 4 only if (a) the tuned baseline's min-over-domains deficit vs
   matched-budget ceilings exceeds ~3pp, AND (b) delta-interference is negative beyond the
   split-half null, AND (c) the offline bake-off shows geometry > plain averaging. Otherwise
   pivot to the reframed empirical paper with the diagnosis in hand.

**Three questions for Weiyang** (before building anything):

1. FDA's anchor construction requires one parameter space — teachers are 7B, student 1.5B. Which
   variant do you intend: merge-teachers-at-7B-then-distill, anchors over *student-space* K-step
   deltas, or anchors reinterpreted as discrepancy-mined probe prompts?
2. At per-step scale, Lie-algebra/spherical aggregation reduces to the Euclidean average (O(η²)
   corrections), so the geometry only has content on accumulated deltas — do you agree the method
   should live in an outer loop (K-step branches, merge, re-branch), and what K?
3. Given our 5-teacher naive baseline improved all five domains, what teacher/data regime does
   your experience say produces genuine destructive interference for a small student — or is the
   toolkit's value here the *diagnosis* rather than the fix?

## Answers to brief §9

1. **Which architecture?** None of α/β/γ as written. δ (outer-loop student-space branch-merge),
   with the FDA-probe-prompt idea as the diagnostic/anti-forgetting arm.
2. **Division of labor** FDA-as-representation / AMO-as-orthogonalizer is the right reading of the
   papers, but AMO belongs (if anywhere) as a *whitening step on deltas before consensus*, and it
   must be ablated against AMO-on-the-naive-sum or it confounds everything.
3. **FDA × on-policy interface**: anchors cannot replace rollouts (type error); as mined probe
   prompts they *augment* rollouts for suppressed domains only.
4. **Where does aggregation live?** Above KDFlow's combine point, at the round boundary — not
   inside the micro-batch loop.
5. **Magnitude side**: per-domain normalization is right in spirit but is absorbed by AdamW+clip
   in-loop; on outer-loop deltas, normalization + TA (and TIES as ablation) is meaningful. ×c is
   a no-op in-loop; on deltas it is exactly OrthoMerge's regime.

---

# Addendum (2026-07-10, after user clarification): the intended method is OFT-in-the-OPD-loop — this changes the assessment

**Clarification received:** the idea is to migrate OFT / orthogonal methods from *weight merging*
to *multi-teacher OPD training time*. Weiyang is on board, reports a prior small experiment
applying OFT to OPD **worked**, and suggests FDA can help.

## What this changes

**Finding 2 (per-step collapse) no longer kills the intended method.** That finding applies to
aggregating raw per-step gradients. Per-domain **OFT rotations trained inside the OPD loop are
O(1) objects far from identity** — the verifier explicitly scoped them out of the collapse: on
accumulated rotations, so(d) averaging ≠ Euclidean averaging, and the ×c correction has real
content. This is the "iterated OrthoMerge trainer" variant the steelman lens flagged as *the only
in-loop object where OrthoMerge's math applies natively*. The clarified project = that variant.

**The motivation shifts from "resolve conflict" to "isolate, then consolidate."** With per-domain
rotations R_d, each domain's OPD loss updates only its own R_d — there is *no cross-domain
gradient interference during accumulation at all*; conflict is deferred to the merge, where the
geometry is principled. This sidesteps the "see-saw never observed" problem as a *motivation*
(the method doesn't need per-step conflict to exist), though it sharpens the *bar*: naive mixed
full-parameter OPD already gets 5/5 domains positive, so the OFT-merge method must match the mean
and beat the min with strictly less expressivity.

**FDA now type-checks.** Per-domain OFT branches of the *student* define genuine student-space
task objects (R_d, or W_d − W₀). FDA anchors can be synthesized in student space against the
student's own branches — coherent uses: (i) joint adaptation data for the merged student after
each merge (data-free complement to on-policy rollouts), (ii) replay/anti-forgetting data for
suppressed domains between merges.

## The architecture, made concrete

Per layer: freeze W₀ (student), per-domain R_d via Cayley–Neumann (OFTv2, block-diagonal).
Each domain's rollouts scored by its teacher (reverse-KL) update only R_d. Every K steps:
inverse-Cayley → Q_d ∈ so(d) → average → ×c = Σ‖Q_d‖/‖ΣQ_d‖ → Cayley → R_merged; re-branch
(fold R_merged into W₀, reset R_d ≈ I). Final model = merged student.

**Two design forks that determine novelty and difficulty — get Weiyang's intent:**
1. **Merge cadence.** Merge once at the end = OrthoMerge applied to OPD-trained experts (novelty
   shrinks to "experts trained by OPD"; also the experts drift apart, exactly post-hoc merging's
   weakness). **Iterated merge/re-branch (K ~ tens of steps) is the genuinely new in-loop object**
   and keeps branches close (better merge quality, closer to on-policy for the merged model).
   Neighbors to cite/differentiate: ColD Fusion/DIMAT/MERIT (iterated train-merge), CaMOPD.
2. **Rollout policy.** Branch-local rollouts (W_d) are cleanly on-policy per branch but the merged
   model is off-policy at merge points; rolling out from the merged model each round restores
   on-policyness at the cost of engine weight-swaps. KDFlow engineering: this lives *above* the
   combine point (round boundary), avoiding the non-domain-pure micro-batch surgery entirely.

## What still stands from the main assessment

1. **Capacity/headroom is now the #1 risk** (was Finding 7): rotations are spectrum-preserving;
   a 1.5B student absorbing 4–5 domains through R_d only may underfit vs full-param OPD. The
   *decisive first experiment* is therefore **Gate A′: single-teacher OFT-OPD vs full-param OPD
   parity** on one domain. Ask Weiyang for the prior experiment's details (scale,
   single-vs-multi-teacher, what metric improved, vs what baseline) — if it already shows parity
   at comparable scale, Gate A′ is pre-cleared.
2. **All measurement fixes**: teacher-native templates/system prompts; the math teacher's 4k
   context overrun; paired per-problem evals; matched-budget ceilings; regenerate baselines on
   KDFlow (engine-migration hole).
3. **The honest ablation set** (what makes any win attributable): (i) naive mixed full-param OPD
   with tuned weights; (ii) per-domain LoRA/full branches merged with TA/TIES — is *orthogonality*
   doing the work, or just branch-merge?; (iii) OFT branches merged with plain averaging of Q_d —
   is ×c/so(d) doing the work?; (iv) CaMOPD-style alternating updates (cheapest competitor).
4. **min-over-domains stats and pre-registered gates** as before.

## Revised minimal path

1. Gate A′: 1 domain, OFT-OPD vs full-param OPD (capacity check) — on the fixed measurement layer.
2. 2-domain head-to-head: iterated OFT-branch-merge vs tuned naive mixed OPD vs LoRA-branch-TIES,
   paired evals, K swept coarsely.
3. Offline pre-test stays valid and even more natural: train per-domain OFT branches once, merge
   offline with so(d)+×c vs plain-average vs TA — one day, tests the merge math before any
   iterated-loop engineering.
4. FDA arm only after (2) shows the merge preserves per-domain gains: student-space anchors for
   post-merge joint adaptation.

---

# Addendum 2 (2026-07-10): Weiyang's decomposition recipe, made precise

**Guidance relayed:** orthogonalize the per-domain signal (OFT, Muon, or his tool that finds the
nearest orthogonal matrix), then decompose into direction (polar coordinates, aggregated on the
manifold) and magnitude (aggregated linearly, TA-style).

**What this is:** the nearest orthogonal matrix to A (Frobenius) is its polar factor
Q = UVᵀ from A = USVᵀ — matrix "polar coordinates" A = Q·H (Q = angle, H = VSVᵀ ⪰ 0 = radius).
Muon/NS computes Q iteratively; the Procrustes/SVD solve computes it exactly; OFT parameterizes it
directly. This recipe is OrthoMerge's **Orthogonal-Residual Decoupling** transplanted to per-domain
OPD deltas at merge points. Three concrete instantiations:

- **V1 — OFT branches (direction by construction).** Per-domain R_d trained in-loop; no magnitude
  channel exists (pure rotation). Merge: so(d) average + ×c. Cleanest math; capacity risk
  (spectrum-preserving only).
- **V2 — full-param branches + ORD (recommended; matches the description).** Per domain, solve
  R_d = polar(W_d·W₀ᵀ) (nearest rotation taking W₀ toward W_d), residual E_d = W_d − R_d·W₀.
  Direction channel: so(d)-average the R_d with ×c. Magnitude channel: TA/TIES on the E_d.
  Recompose W_new = R_merged·W₀ + Σλ_d·E_d. Keeps full expressivity (residual carries content),
  uses the Procrustes tool directly, and R_d is guaranteed near-identity (see landmine below).
- **V3 — polar of the delta (Muon-flavored).** Q_d = NS(ΔW_d), H_d = magnitude. Note polar() is
  scale-invariant — polar(cA) = polar(A) — so this is genuinely non-degenerate even for small
  deltas (the per-step collapse critique does NOT apply). Recompose ΔW = Q_merged·H̄ with
  H̄ = Σλ_d H_d (SPD cone is convex; linear averaging valid). BUT see landmine.

**Landmine — the near-identity requirement.** so(d)/Cayley averaging is only defined for
rotations near I with det = +1. OFT rotations (V1) and ORD rotations of nearby checkpoints (V2)
satisfy this automatically. **polar(ΔW) does not**: the polar factor of an update matrix is
generically FAR from I and can have det = −1 (not even in SO(d) — no skew logarithm exists).
V3's directions cannot be Lie-algebra-averaged as-is; they'd need a Karcher mean on the orthogonal
group (expensive, non-unique between components). This is a decisive argument for **V2 (or V1)
over V3** as the primary build — raise with Weiyang.

**Other fine print:**
- Rectangular matrices (MLP up/down, lm_head): keep R square on the output side (R·W₀, like OFT)
  so the group structure survives; polar factors of rectangular deltas are only semi-orthogonal
  (Stiefel, no group).
- "Magnitude = TA" must be pinned: scalar norms vs SPD factor H vs residual E are three different
  methods. Scalar-only discards the spectral shape (likely hurts); TA-on-residual is the
  OrthoMerge-sanctioned reading; SPD-average is the clean V3 form.
- Noise: polar amplifies the tail of a noisy delta's spectrum (all directions → weight 1).
  Merge at K-step round boundaries with K chosen so ‖ΔW‖ clears the split-half noise floor
  (reuse the Addendum-1 diagnostics to pick K).
- Compute is a non-issue at round cadence: exact SVD polar on 1.5B-scale matrices is seconds/GPU,
  NS is faster; only per-step cadence would be costly.
- ×c lives on the rotation average (undoing cancellation in so(d)); λ_d/normalization lives on
  the magnitude channel — consistent with brief §4's "normalize, don't ×c" note, each channel
  gets its own treatment.

**Updated offline bake-off arms** (still the right first experiment; one day): plain delta
average / TA / TIES / V1 (if OFT branches) / V2-ORD / V3-polar — plus direction-only and
magnitude-only ablations of V2 to attribute any win to a channel.

**For Weiyang:** (1) which tool did he mean — the OrthoMerge release's Procrustes solve, or the
Muon/NS kernel? (2) confirm V2 (ORD on branch weights) vs V3 (polar of deltas) — the det/-far-from-I
issue makes them materially different; (3) in his successful OFT+OPD experiment, was the merge
so(d)+×c or something simpler?

## Honest limitations of this assessment

Iter2's all-positive result is one 80-step, single-seed run — long-horizon see-saw is not ruled
out, and the reviewers themselves note 80 steps is uninformative about near-saturation
interference. AMO/OrthoMerge claims rest on abstract-level paraphrases (full texts unfetchable
here). The format-conflict minimal-pair recommendation (math vs tool-JSON rather than math vs
medical) survived review only as UNCERTAIN. And the panel did not fully converge on the pivoted
project's headline — that choice (method-with-empirical-wedge vs diagnosis-first study) is a
strategic call for the humans, with the collaborator's expectations in the room.
