# MOPD × Orthogonal Aggregation — The Complete Project Guide

**How to read this document.** Part I tells the whole story in plain language —
no equations, no prior knowledge beyond "a language model predicts the next
word." Part II rebuilds every idea with the actual math, always through small
worked examples you can check with a calculator. Part III is hands-on: run the
pipeline on your laptop in five seconds, then find your way around the repo.
A glossary and FAQ close the document. Jargon is **bolded** where it is first
explained, and every bolded term is also in the glossary.

---

# Part I — The Big Picture

## 1. What this project is about, in one paragraph

We have several large "specialist" language models — one great at math, one at
medicine, one at coding — and we want to pour all of their skills into a single
small model. The standard way to combine their teaching signals is to simply
*add them up*. We are testing a mathematically more careful way of combining
them, borrowed from our collaborator Weiyang Liu's work on "orthogonal"
methods: split every teaching signal into a *direction* and a *strength*,
combine directions the way you'd combine compass bearings, and combine
strengths the way you'd average numbers. The project's job is to find out —
with honest measurements — whether that extra care actually produces a better
student model than the simple sum everyone uses today.

## 2. Why would anyone want this?

Training a language model to be excellent at *one* thing is comparatively
easy: you optimize for math without caring whether medical advice gets worse.
Companies therefore often train several **specialist** models (in this project
we call them **teachers**). But serving five models is expensive and clumsy —
users want one assistant. So there is a compression step:

> take N specialist teachers → produce one small generalist **student**.

Our teachers are 7-billion-parameter models from the Qwen2.5 family (a math
specialist, a medical specialist, etc.). Our student is a much smaller
1.5-billion-parameter model. The question is how to get the skills across.

## 3. What "distillation" means

A language model doesn't just output a word — internally, at every position it
produces a score for *every word in its vocabulary* (~150,000 of them), which
becomes a probability distribution: "next word is 'the': 41%, 'a': 22%, …".

**Knowledge distillation** trains the student to match the teacher's
*probability distributions*, not just the teacher's final answers. That is a
far richer signal than "the right answer was C": the student learns how
confident to be, which alternatives are plausible, and which are nonsense —
thousands of numbers per word instead of one.

The name comes from the image of distilling a big model's knowledge down into
a small one. The teacher is **frozen** (never changes); only the student learns.

## 4. On-policy vs. off-policy: whose text do we practice on?

There are two ways to run distillation, and the difference matters a lot.

- **Off-policy**: the student studies *the teacher's* texts. Like learning to
  drive by watching videos of a perfect driver — you never learn how to
  recover from your own mistakes, because the perfect driver never makes them.
  The mistake situations you'll face are missing from the training data. (The
  formal name for this problem is **exposure bias**.)
- **On-policy**: the student *writes its own answer first*, and the teacher
  then grades every word of that answer: "at this point, here's how likely I
  would have been to write each word you could have chosen." Like a driving
  instructor sitting next to you while *you* drive — feedback lands exactly on
  the situations your own behavior gets you into.

This project uses **on-policy distillation (OPD)** throughout. One training
step looks like this:

```
1. Take a batch of prompts (math problems, medical questions, ...)
2. STUDENT writes an answer to each prompt              ("rollout")
3. TEACHER reads each student answer and, for every word position,
   outputs its own probability distribution over next words
4. Loss = how far the student's distributions are from the teacher's
   (measured with "reverse KL" — explained gently in §5, precisely in Part II)
5. Nudge the student's weights to shrink that gap        (one "update")
```

## 5. The grading rule, in words

The distance in step 4 is the **reverse KL divergence**. The only intuition
you need at this stage: reverse KL is a *strict* grader about nonsense. If the
student puts even a little probability on words the teacher considers absurd,
the penalty is enormous; if the student plays it safe and covers only part of
what the teacher would accept, the penalty is mild. So the student learns to
stay inside the teacher's notion of "sensible" — which is what you want from a
small model — rather than trying to imitate everything the teacher could ever
say (a small model doesn't have room for that). Part II §2 makes this exact
with a three-word example you can check by hand, including the failure mode
this strictness creates and how we guard against it.

## 6. Many teachers: routing

With several teachers, each prompt is **routed** to the right one: the math
teacher grades math prompts only, the medical teacher grades medical prompts
only. (This mirrors how frontier labs do it; no teacher ever grades another
domain's rollouts.) Every batch contains a mix of domains, so after grading we
hold several teaching signals — one per domain — and they all want to update
the **same shared student weights**.

**The combination step is the entire subject of this project.** Today's
standard: multiply each domain's signal by a hand-tuned importance weight and
add them up. We call this the **weighted Euclidean sum** — "Euclidean" just
means the signals are treated as plain arrows that you add tip-to-tail.

## 7. The suspected problem: the see-saw

Adding arrows can destroy information. Imagine math's update says "move
northeast" and medicine's says "move northwest." Adding them gives "move
north" — and *both* domain-specific parts cancel each other. Scale that up to
a billion-dimensional weight space and the worry becomes:

- updates partially cancel where domains disagree (**cancellation**);
- the domain with the louder signal wins the tug-of-war (**imbalance**);
- one domain's score climbs while another's quietly falls (**the see-saw**).

This is a *suspected* problem — read §9 before assuming it is real.

## 8. The proposed fix: directions and strengths

Here's the everyday version of the mathematical idea (Part II §8–§12 does it
properly).

Think of each domain's update as a compass bearing plus a distance: "go 3 km
northeast." Now combine "3 km northeast" and "3 km northwest" two ways:

- **Add the arrows** (what everyone does): you get "2.1 km north" — the
  direction is fine but you lost 30% of the distance to cancellation. In
  higher dimensions, with more teachers, more is lost.
- **Combine bearings and distances separately** (our approach): average the
  *bearings* on the compass dial (northeast + northwest → north), average the
  *distances* as plain numbers (3 km + 3 km → 3 km), and reassemble: "3 km
  north." Nothing cancels, because directions were combined on the curved
  space where directions actually live (the compass circle), and strengths —
  which are just numbers, and numbers can't point against each other — were
  combined separately.

For neural networks, "the compass circle" becomes the space of **rotations**
(orthogonal matrices), and there is a clean recipe — the **polar
decomposition** — that splits any update into a rotation part (direction) and
a stretch part (magnitude). Weiyang's group built an entire toolkit around
this: **OFT** trains models through pure rotations; **Muon/AMO** extracts the
rotation part of an update quickly; **OrthoMerge** combines several finished
specialists' rotations on the proper curved space, with a correction factor
(`×c`) that restores the strength lost to cancellation. OrthoMerge does this
*after* training, to merge finished models. **Our project moves that machinery
into the training loop itself** — combining the teachers' influences while the
student is still learning, which no lab currently does.

## 9. What we've already learned (read this — it shapes everything)

This project has history, and the history taught us three lessons that make
this a *measure-first* project rather than a build-first project.

**Lesson 1: check your teachers.** An early experiment collapsed
spectacularly — the student unlearned math within 40 steps, ending at 0%
accuracy. Weeks of debugging pointed at exotic causes before the real one
surfaced: the downloaded math teacher checkpoint was *broken* — it literally
output only space characters, and its training log showed it had collapsed
during its own RL training. Distilling from a broken teacher is how you get a
broken student. Since then, every teacher goes through a health check
(generate text on a few prompts, verify it's coherent) before it is ever used.

**Lesson 2: the see-saw is not guaranteed.** When we later ran a clean
5-teacher experiment with healthy teachers and the *plain Euclidean sum* —
the method we're trying to beat — **all five domains improved simultaneously**
(math +2.3 points, medical +7.5, coding +1.2, tool-use +14.3, search +4.3).
No see-saw. It may appear with longer training, more specialized teachers, or
different domains — but we have *never observed it yet*, and a method that
fixes a problem that doesn't occur is worthless. That is why the current
sweep leads with *instruments* that can detect and quantify conflict, and why
the plan has explicit "stop" gates if conflict turns out to be absent.

**Lesson 3: measure on the right objects.** Single training steps are almost
pure noise (Part II §5), teachers must be prompted in their own native format
or they look artificially bland, and one teacher (the math specialist) can
only read 4,096 words at a time — feed it more and its grades past that point
are garbage. All three measurement traps are now fixed in the pipeline.

## 10. What is running right now, and what happens next

With a large GPU allocation available this week, we're running a wide sweep
(details in `sweep/RUNBOOK.md`) that produces, for each domain:

1. a **branch**: a copy of the student trained on that domain alone — this
   gives both the "how good could the student get with this teacher"
   ceiling *and* the raw material for merging;
2. **merged students**: every way of combining the branches — the plain
   average, the standard weighted sum, and the geometric methods (with each
   ingredient switchable on/off so we can attribute any win);
3. **diagnostics**: direct measurements of how much the domains' updates
   actually conflict, with a statistical noise floor so we don't fool
   ourselves;
4. **baselines**: the conventional all-domains-at-once training, at several
   importance-weight settings — the opponent to beat.

Everything gets evaluated per-domain with paired statistics. When we meet
Weiyang next week, the data will answer: does conflict exist here? do the
geometric merges beat the simple ones? is the rotation-only variant (OFT)
expressive enough for a small student? — whichever way he wants to take the
method, the relevant evidence will already exist.

## 11. Try it yourself in five seconds (no GPU needed)

```bash
pip install torch safetensors     # CPU versions are fine
python3 sweep/demo_synthetic.py
```

This builds a tiny fake student + two "domain branches" whose updates
genuinely conflict, then runs the *real* merging code (all eight operators)
and the *real* conflict diagnostics on them, printing the same report the
full-scale sweep produces. The file invites you to dial the conflict up and
down and watch the numbers respond — the fastest way to build intuition for
Part II.

---

# Part II — Technical Foundations

Notation used throughout:

```
x                    a prompt;  y = (y_1 ... y_T) a generated response
π_s, π_t             student / teacher next-token distributions
W, W0                a weight matrix; its pretrained ("base") value
Δ or delta           an update or accumulated change:  Δ = W_trained − W0
g                    a gradient (the per-step update direction)
‖A‖                  Frobenius norm (square root of the sum of squared entries)
A^T                  transpose;  I  the identity matrix
d                    number of parameters (~1.5 × 10^9 for our student)
N                    number of domains/teachers (4–5 here)
```

## §1. The distillation objective

A language model maps a context to **logits** — one raw score per vocabulary
item — which softmax turns into a probability distribution:

```
π(v | context) = exp(logit_v) / Σ_u exp(logit_u)
```

Distillation minimizes a divergence between the student's and teacher's
distributions at every token position of the training text. Which text, and
which divergence, are the two design choices that define the method.

## §2. KL divergence: forward vs. reverse, and why it matters

The **Kullback–Leibler divergence** between distributions P and Q is

```
KL(P ‖ Q) = Σ_v P(v) · log( P(v) / Q(v) )      (≥ 0;  = 0 iff P = Q)
```

It is asymmetric — `KL(P‖Q) ≠ KL(Q‖P)` — and the choice of *which way round*
changes the learned behavior qualitatively. On-policy distillation uses
**reverse KL**: `KL(π_s ‖ π_t)` — student first, evaluated on the student's
own rollouts.

**Worked example.** Three possible next words; the teacher is torn between
two good options and considers the third nonsense:

```
teacher π_t = (0.4995, 0.4995, 0.001)
```

Compare two imperfect students:

```
student B "picks one mode":        π_s = (0.998, 0.001, 0.001)
student C "spreads 20% on junk":   π_s = (0.4,   0.4,   0.2  )

              reverse KL(π_s‖π_t)      forward KL(π_t‖π_s)
student B          0.68                     2.76
student C          0.88                     0.22
```

(Check one entry of B's reverse KL: `0.998·ln(0.998/0.4995) = 0.998·0.692 =
0.69`; the other terms are tiny.)

Read the table: **reverse KL punishes junk hardest** (C's 0.2 on the nonsense
word costs it `0.2·ln(0.2/0.001) = 1.06` all by itself) and is lenient about
dropping one of the teacher's modes (B gets the *better* reverse-KL score).
Forward KL is the exact opposite: it forces the student to cover every teacher
mode (B is heavily punished for the missing mode) and barely minds junk. This
is the **mode-seeking** (reverse) vs. **mode-covering** (forward) distinction.

For a small student, mode-seeking is usually right: a 1.5B model cannot
represent everything a 7B teacher can, and we'd rather it do a subset
confidently than everything badly — and its errors (junk mass) are suppressed
hard, which fights hallucination.

**The failure mode.** Strictness has a dark side: on a teacher whose
distribution the student *can't* sensibly approach, the cheapest way to
reduce reverse KL can be degenerate text — endless repetition with collapsing
entropy ("The answer is 3. The answer is 3. …"). Our project history contains
a spectacular instance (Part I §9, the broken teacher, whose distribution was
"far" from every sensible model). Guardrails: teacher health checks before
training; monitoring entropy, output length, and the fraction of responses
hitting the length cap during training; and per-domain evals that would catch
an accuracy/loss divergence.

We set `kd_ratio = 1.0` (pure KD loss, no supervised mixing) so that the
aggregation mechanism is the only experimental variable.

## §3. One training step, precisely (as KDFlow implements it)

```
prompts ──► student generates rollouts (sglang engines serve the student)
        ──► rollouts are routed:  each sample's hidden states go to ITS
            domain teacher, which produces teacher logits per token
        ──► per-token loss = reverse KL(student ‖ teacher), summed over
            the batch, normalized by token count
        ──► loss.backward() ACCUMULATES gradients across micro-batches
        ──► AdamW optimizer step  (lr 2e-6, gradient clipped to norm 1.0)
```

The line to stare at is the accumulation: every domain's token losses feed
**one** gradient buffer, so the update applied to the student is *exactly the
plain sum of per-domain gradients*. That sum is the baseline mechanism.
KDFlow's multi-teacher path supports only this; our patches add the two knobs
the baseline needs to be a *fair* opponent:

- `--teacher_loss_weights {"math": 2.0, ...}` — per-domain weights inside the
  loss (the "tuned weighted sum" every lab runs);
- `--teacher_max_len {"math": 4096}` — zero the loss on tokens beyond a
  teacher's trained context window. Why this exists: the math teacher
  (Qwen2.5-Math-7B-Instruct) was trained with 4,096-token **RoPE** positions;
  ask it to grade token 5,000 and you get logits from a positional geometry it
  has never seen — noise dressed as a grade. Sequences here can reach 6,144
  tokens, so without the cap every long math rollout injects garbage into the
  loss (and into any teacher-distance measurement).

## §4. Do the domains actually fight? A 2-D worked example, then the real test

Take one shared parameter space with just two dimensions and two domains:

```
g_math = ( 1.0,  0.6 )        shared part: x-axis    domain part: y-axis
g_med  = ( 1.0, −0.6 )
sum    = ( 2.0,  0.0 )
```

The shared component reinforces; the domain-specific components annihilate.
`‖g_math‖ = ‖g_med‖ = 1.166`, so the norms sum to 2.33, but `‖sum‖ = 2.0` —
and 100% of the domain-specific signal is gone. If the norms are imbalanced
(`g_med = (0.5, −0.3)`), the sum is `(1.5, +0.3)`: the surviving y points
toward math — medicine is dragged along math's preference. That is the
see-saw mechanism in miniature.

**But beware the high-dimensional null.** In `d ≈ 1.5×10⁹` dimensions, two
*independent random* directions are nearly orthogonal: their cosine
similarity concentrates around `±1/√d ≈ ±0.00003`. Try it: two random unit
vectors in 1,000 dimensions typically have `|cos| ≈ 0.03`, and their sum has
norm `≈ √2 = 1.41`, not 2. So **norm shrinkage of a sum proves nothing**: for
N equal-strength independent teachers, `‖Σg‖ = √N·‖g‖` while `Σ‖g‖ = N·‖g‖`,
so the ratio `ρ := ‖Σg‖ / Σ‖g‖ ≈ 1/√N` *with zero conflict*. Conflict is a
claim about **negative inner products**, judged against that null:

```
cos(Δ_a, Δ_b) < 0   beyond what noise explains   ⟺   real interference
ρ  compared to  1/√N  (NOT to 1)                 ⟺   more/less than "independent"
```

Our diagnostics (§14) implement exactly this, with a measured noise floor.

## §5. Why per-step geometry is a dead end (the argument in full)

It is tempting to apply the geometric machinery to each *training step's*
gradients. Three facts, in sequence, kill this:

1. **Per-step updates are microscopic.** With lr `2×10⁻⁶` and clipped
   gradients, one AdamW step moves the weights by roughly one part in 10⁴.
   Any "rotation" representing such a step sits a hair's width from the
   identity matrix.
2. **Near the identity, curved and flat averaging agree.** For rotations
   `R_i = I + ηA_i + O(η²)` with tiny η, averaging them on the curved space
   (via logarithms/Cayley, §11) equals plain averaging up to `O(η²)` — here a
   relative difference of ~10⁻⁸. The geometry has *literally nothing to act
   on*; every geometric scheme collapses to the weighted Euclidean sum it was
   supposed to beat, leaving at most an overall scalar.
3. **The scalar doesn't survive the optimizer.** AdamW normalizes each
   parameter's update by its running gradient scale, so a constant rescaling
   of all gradients cancels out of the update (and gradient clipping caps
   whatever remains). The ×c-style correction, applied per-step, is an
   elaborate no-op.

Additionally, a *single* step's per-domain gradient (a few dozen samples) is
dominated by sampling noise — its direction barely correlates with the same
domain's average direction (§14 measures this as "reliability").

**What escapes all of this: O(1) objects.** Accumulate a domain's training
over K steps into a branch delta `Δ_d = W_d − W0`, or train an explicit
rotation adapter (OFT) — these are *large* displacements with high
signal-to-noise, the curved-vs-flat distinction is real for them, and ×c has
genuine content. Hence the project's architecture:

> **branch** (train per-domain for K steps) → **decompose** (direction +
> magnitude) → **merge** (geometry for directions, arithmetic for magnitudes)
> → optionally **re-branch and repeat**.

## §6. A primer on orthogonal matrices

A square matrix `R` is **orthogonal** iff `RᵀR = I` — its columns are
mutually perpendicular unit vectors. Consequences that make orthogonal
matrices "safe":

- they preserve lengths: `‖Rx‖ = ‖x‖` for every x;
- they preserve angles between vectors;
- `det(R) = +1` (**rotation**) or `−1` (**reflection** — a flip). The
  rotations form the group **SO(n)**; you can travel smoothly from any
  rotation to the identity, but you cannot smoothly turn a reflection into a
  rotation (there's no "half a mirror flip") — this distinction becomes a
  real engineering constraint in §13.

Intuition for why they matter here: applying a rotation to a weight matrix
re-mixes its rows/columns *without changing any magnitudes* — it can move
knowledge around without amplifying or erasing it. That norm-preservation is
the formal version of "no cancellation," and it is why Weiyang's finetuning
line (OFT) constrains updates to be rotations: they provably cannot blow up
or collapse the pretrained feature geometry.

## §7. Polar decomposition: every matrix is a rotation times a stretch

The complex-number fact `z = e^{iθ} · r` (angle times radius) generalizes to
matrices. For any matrix `A` with **SVD** `A = U S Vᵀ` (S diagonal ≥ 0):

```
A = Q · H        Q = U Vᵀ   (orthogonal — the "angle"/direction)
                 H = V S Vᵀ  (symmetric positive semi-definite — the "radius"/stretch)
```

**Worked 2×2 example.** The matrix

```
A = [ 1.732  −0.25  ]
    [ 1.0     0.433 ]
```

is exactly "stretch x by 2, shrink y to 0.5, then rotate by 30°":

```
Q = [ cos30° −sin30° ] = [ 0.866 −0.5   ]        H = [ 2   0   ]
    [ sin30°  cos30° ]   [ 0.5    0.866 ]            [ 0   0.5 ]
```

(Multiply Q·H and check you recover A.) The polar decomposition finds this
split for *any* matrix.

Two properties we lean on constantly:

- **Nearest orthogonal matrix.** Q = UVᵀ is the closest orthogonal matrix to
  A in Frobenius distance — the solution of the **orthogonal Procrustes
  problem**. When Weiyang says "the tool that automatically finds the nearest
  orthogonal matrix," this is the object it computes.
- **Scale invariance.** `polar(cA) = polar(A)` for any c > 0: direction
  extraction ignores how *far* the update went — exactly the
  direction/magnitude split we wanted, done by linear algebra instead of
  metaphor.

## §8. Three ways to get the direction

**(a) SVD — exact.** Compute `U S Vᵀ`, return `UVᵀ`. On matrices of our size
(≤ 8960×1536) this takes seconds on a GPU and is what the merge library uses
by default (`method="svd"`).

**(b) Newton–Schulz iterations — fast, approximate (this is Muon).** Start
from `X = A/‖A‖` and repeat a fixed polynomial map

```
X ← a·X + (b·XXᵀ + c·(XXᵀ)²) X        (a, b, c) = (3.4445, −4.775, 2.0315)
```

five-ish times. Each pass pushes A's singular values toward 1 while leaving
its singular *directions* untouched — annealing the stretch out of the matrix
until only the rotation remains. The Muon optimizer applies exactly this to
gradients. Trade-off (we measured it): singular values land in roughly
[0.7, 1.3], not at 1.0 — great for an optimizer's inner loop, unnecessary
for our offline merges, where exact SVD is cheap. Both are implemented; the
sweep uses SVD.

**(c) OFT — direction by construction.** Instead of extracting a rotation
from an unconstrained update after the fact, *parameterize* the update as a
rotation from the start: freeze `W0`, train `W = R·W0` where R is kept
orthogonal throughout via the Cayley map (§11) and block-diagonal structure
keeps it cheap. Then "the direction of the branch" needs no extraction — R
*is* it. This is our V1 variant, and Weiyang reports a small OFT-on-OPD
experiment already worked. Open question it must answer (Gate A′, §15): pure
rotations preserve all magnitudes by design, so can a 1.5B student absorb a
whole domain through rotations alone?

## §9. V2 — Orthogonal-Residual Decoupling (the recommended decomposition)

Given a branch `W_d` and the base `W0` (same shape), find the rotation that
best explains the change, and keep whatever's left as a residual:

```
R_d = polar( W_d · W0ᵀ )                 (the Procrustes solution: the
E_d = W_d − R_d · W0                      nearest rotation carrying W0
                                          toward W_d)
⟹  W_d = R_d · W0 + E_d   exactly.
```

- **direction channel** = R_d (merged on the curved space, §11);
- **magnitude channel** = E_d (merged with plain arithmetic, §12).

Because branches start at W0 and move modestly, `R_d` is guaranteed close to
the identity with det = +1 — safely inside the domain where the direction
math of §11 is valid. That guarantee is V2's decisive advantage over V3
(§13).

**Rectangular matrices.** so(n) machinery needs a *square* R. For a
non-square W (e.g. the 151936×1536 output layer), we put R on the smaller
side: `W ≈ R·W0` (R is out×out) if out ≤ in, else `W ≈ W0·R` (in×in) — both
are Procrustes problems, and this mirrors how OFT attaches its rotations.
Vectors (biases, layer norms) skip the geometry and merge arithmetically.

## §10. Why you can't just average rotation matrices

Try averaging `R(+30°)` and `R(−30°)` entrywise:

```
mean = [ cos30°   0    ]  =  0.866 · I
       [ 0      cos30° ]
```

That is **not a rotation** (det = 0.75): it shrinks every vector by 13.4%.
Entrywise averaging leaves the curved space of rotations and pays for it in
lost magnitude — the compass problem again, now in matrix form. The remedy is
to average not the rotations but their *coordinates on the curved space*:

## §11. The Cayley transform, so(n), and the ×c correction

A **skew-symmetric** matrix satisfies `Qᵀ = −Q` (zero diagonal, mirrored
entries with flipped sign). The set of these, called **so(n)**, is a flat
vector space — adding and averaging inside it is legitimate. The **Cayley
transform** is a two-way bridge between skew matrices and rotations near the
identity:

```
to a rotation:      R = (I − Q)(I + Q)⁻¹        (always lands in SO(n))
back to skew:       Q = (I − R)(I + R)⁻¹        (defined when R has no −1 eigenvalue)
```

In 2-D this is beautifully concrete: `Q = [[0, −t], [t, 0]]` maps to the
rotation by angle `2·arctan(t)` — the single number t *is* the angle in
disguise (t = tan(angle/2)). Averaging t-values averages angles, which is
exactly what entrywise averaging of the R's failed to do.

**The merge rule (OrthoMerge's, verbatim in our code):**

```
Q_d  = CayleyLog(R_d)                          d = 1..N   (one per domain)
Q̄   = Σ_d w_d Q_d                             (weighted average in so(n))
c    = ( Σ_d w_d ‖Q_d‖ ) / ‖ Q̄ ‖              (norm-restoration factor)
R*   = Cayley( c · Q̄ )                        (back to a genuine rotation)
```

**What ×c does — compass numbers.** Bearings northeast and northwest as unit
arrows: `(0.707, 0.707)` and `(−0.707, 0.707)`. Their average is
`(0, 0.707)` — right direction (north), 29% too short. Here
`c = (1+1)/1.414 = 1.414`, and `c · (0, 0.707) = (0, 1.0)`: full-strength
north. ×c restores the energy that cancellation between disagreeing
directions destroyed — in OrthoMerge's ablations this single factor is the
largest contributor to its wins.

**When ×c is aggressive.** If two domains rotate nearly *oppositely*
(+30° and −18°, say), their average is a faint +6°, and ×c inflates that
faint consensus back to average strength (+24°) — trusting a consensus that
barely exists. Whether that helps or hurts is an empirical question, which is
why the sweep runs `v2_ord` and `v2_ord_noxc` side by side.

## §12. The magnitude channel: Task Arithmetic and TIES

Residuals (and any non-geometric parameters) are combined with ordinary
vector arithmetic — safe for magnitudes, since scalars can't point against
each other:

**Task Arithmetic (TA):** `E* = Σ_d λ_d E_d`. The λ's are per-domain knobs —
this is where imbalance between teachers is handled (a **normalization** job:
losses/updates from different teachers have different characteristic scales,
so per-domain normalization equalizes voices *before* combining — a different
job from ×c, which repairs the *direction average*; don't conflate them).

**TIES** adds interference-awareness in three steps — worked example on one
coordinate with two domains proposing `+1.0` and `−0.4`:

```
1. TRIM  : zero each domain's small-magnitude entries (keep top-k%)
2. ELECT : majority sign per coordinate:  sign(+1.0 − 0.4) = +
3. MERGE : average only the entries agreeing with the elected sign
           → +1.0        (plain TA would give +0.3: 70% cancelled)
```

Where domains agree (`+0.5, +0.7 → +0.6`), TIES ≈ TA. Where they collide,
TIES lets the majority win outright instead of splitting the difference —
a *discrete* anti-cancellation mechanism, and the natural head-to-head
opponent for the *continuous* geometric one.

## §13. The three variants, and V3's landmine

| | V1: OFT branches | V2: ORD (recommended) | V3: polar of the delta |
|---|---|---|---|
| direction | trained rotation R_d itself | `polar(W_d·W0ᵀ)` | `polar(Δ_d)` |
| magnitude | — (pure rotation) | residual E_d | SPD factor H_d |
| near identity? | ✔ by construction | ✔ (branches start at W0) | ✘ generically far |
| det = +1? | ✔ | ✔ | **not guaranteed** |
| main risk | capacity (Gate A′) | none known | see below |

**The landmine.** V3 extracts the polar factor of the *update itself*. An
update matrix is not "base plus a small twist" — its polar factor is an
essentially arbitrary orthogonal matrix: typically *far* from the identity,
and possibly a **reflection** (det = −1). Two consequences: the Cayley log
doesn't exist for det = −1 (no skew coordinates — you cannot smoothly reach a
reflection from the identity), and even for det = +1 rotations far from I,
averaging their logs is unreliable (the bridge distorts far from its
anchor). Our implementation guards this: V3 falls back to an extrinsic
average (average, then re-project to the nearest orthogonal matrix), logs
every det < 0 occurrence, and is included in the sweep *to quantify the
problem with data*, not because we trust it. In the synthetic demo the
warning fires immediately — as predicted. This is a named agenda item for the
Weiyang meeting: confirm V2-not-V3 (or learn what we're missing).

## §14. Measuring conflict properly

Everything here exists because of two facts: single-step gradients are noise,
and norm shrinkage is the *default* in high dimensions (§4). The instrument
(implemented in `sweep/delta_diagnostics.py`):

- **Objects:** accumulated K-step branch deltas `Δ_d` — not per-step
  gradients.
- **Pairwise cosines** `cos(Δ_a, Δ_b)`, reported per parameter group
  (embeddings / output layer / attention / MLP / norms) and globally. Why
  per-group: if format-level conflict exists (long chain-of-thought math vs.
  terse tool-call JSON), it should localize in the *format-carrying*
  parameters — output layer rows for end-of-turn and template tokens,
  embeddings — rather than mid-network MLPs. Localization is diagnosis.
- **ρ = ‖ΣΔ‖ / Σ‖Δ‖** compared against **1/√N** (the independence null), not
  against 1.
- **Split-half reliability.** Train the same domain twice on disjoint halves
  of its data; `r_d = cos(Δ_half A, Δ_half B)` measures how much of the delta
  is signal. If `r_d ≈ 0`, the branch delta is noise at this K and *no
  cross-domain statement is meaningful* — increase K first. (This is
  psychometrics' split-half reliability, applied to weight updates.)
- **Disattenuated alignment** `A_ab = cos(Δ_a, Δ_b) / √(r_a·r_b)` — the
  conflict estimate corrected for measurement noise. Example: observed
  cos = −0.09 with reliabilities 0.3 and 0.3 → true alignment ≈ −0.30.
  Moderate-looking raw numbers can hide substantial true conflict, and vice
  versa; this formula is what separates the two.

**Pre-registered readings** (agreed *before* seeing results, so we can't
rationalize afterwards): cosines within the null and ρ ≈ 1/√N → no measurable
conflict (the geometry has no target — stop or pivot); negative cosines
beyond the null, localized somewhere interpretable → conflict is real —
proceed and cite the measurement.

## §15. Experimental design: how we avoid fooling ourselves

- **Ceilings at matched budget.** "Domain d recovered 93% of its ceiling"
  only means something if the ceiling — the single-teacher branch — was
  trained with the same per-domain data/steps the multi-domain run gave that
  domain. Otherwise "interference" and "less data" are confounded.
- **min-over-domains, decided by paired statistics.** Our headline metric is
  the *worst* domain's recovery (a generalist that's 95% on four domains and
  60% on the fifth is a failure). But min is a noisy statistic, and our
  noisiest eval (HumanEval+, 164 problems) has a single-run standard error of
  `√(0.25/164) ≈ 3.9` points — twice the effect we're hunting. Fixes, all
  implemented: bigger eval sets for the noisy domains; **paired**
  per-problem comparison (method and baseline answer the *same* problems;
  analyze the per-problem differences — variance from problem difficulty
  cancels); bootstrap confidence intervals + permutation tests
  (`sweep/paired_compare.py`); no win is claimed whose CI contains zero.
- **Gate A′ (capacity).** For math and medical we train each branch twice:
  full-parameter and OFT (rotation-only). If OFT lands within ~1–2 points of
  full-parameter on its own domain, rotations are expressive enough for V1;
  if not, that is a hard fact about spectrum-preserving updates at 1.5B scale
  that reshapes the method — and the meeting agenda.
- **The ablation grid is the argument.** Any observed win must name its
  load-bearing ingredient: vs `plain_avg` (is it just branch-training?), vs
  `ta` (just weighting?), vs `ties` (is discrete conflict-resolution
  enough?), `v2_ord` vs `v2_ord_noxc` (is it ×c?), vs `v2_dir_only` /
  `v2_mag_only` (which channel?). One column of numbers answers every "but
  couldn't it just be…" in the eventual paper review.
- **Teacher hygiene** (lessons paid for in GPU-hours): health-check every
  checkpoint; prompt teachers in their **native chat template** (a
  specialist prompted off-template looks deceptively like a generic model —
  which contaminated an earlier "teachers are all the same" measurement);
  never let a teacher grade past its trained context window (§3).

## §16. The open forks (= the Weiyang meeting agenda)

1. **V2 vs V3** — the det/far-from-identity argument says V2; his "orthogonalize
   with Muon" phrasing hints V3. Bring the guard logs from the sweep.
2. **Merge cadence** — merge branches once at the end (closest to OrthoMerge,
   least novel) vs. iterated K-step branch/merge/re-branch (genuinely
   in-loop; keeps branches close, so merges stay in the safe near-identity
   regime, and keeps the merged model near the rollout distribution).
   Snapshots at K = 20/40/60/80 let us compare offline before building the
   loop.
3. **Gate A′ outcome** — if rotation-only capacity binds, V1 is out and V2's
   residual channel becomes the load-bearing part.
4. **FDA's role** — his suggestion; with branches living in *student* space,
   FDA-style anchors (synthetic inputs whose induced gradient reproduces a
   branch's delta) become well-defined; candidate uses: post-merge
   adaptation data, or targeted replay for whichever domain the merge
   suppressed. Not in the current sweep; needs his intended design.

---

# Part III — Hands-On Guide

## Run the demo (laptop, no GPU, ~5 s)

```bash
pip install torch safetensors
python3 sweep/demo_synthetic.py
```

You get: eight merged checkpoints under `output/demo_synthetic/merged/` (one
per operator) and a diagnostics report. Then open `sweep/demo_synthetic.py`,
change the `-0.7` conflict coefficient, and watch `cos` and `rho` respond —
five minutes of this teaches §4 and §14 better than any text.

## Run the tests

```bash
python3 test_merge_lib.py        # 38 checks: polar, Cayley, ORD, every
                                 # operator, ×c behavior, TIES, diagnostics
```

## Repo map

| Piece | Location |
|---|---|
| Merge library: polar/Cayley/ORD/operators/diagnostics/IO | `mopd_merge/{linalg,decompose,operators,diagnostics,ckpt_io}.py` |
| Unit tests | `test_merge_lib.py` |
| Laptop demo | `sweep/demo_synthetic.py` |
| Merge bake-off CLI (real checkpoints) | `sweep/offline_merge_bakeoff.py` |
| Conflict diagnostics CLI | `sweep/delta_diagnostics.py` |
| Paired statistics | `sweep/paired_compare.py` |
| OFT/LoRA adapter → full checkpoint | `sweep/materialize_peft.py` |
| Training-data builder (+ split halves) | `sweep/build_kdflow_data.py` |
| Branch / multi-teacher launchers | `sweep/train_{branch,multiteacher}.sh` |
| Teacher roster / context caps | `sweep/configs/` |
| Cluster execution plan + decision gates | `sweep/RUNBOOK.md` |
| KDFlow patches (weights, caps, OFT) | `KDFlow/kdflow/…` (see RUNBOOK "sharp edges") |
| Research log: assessments, collapse proof, landmines | `NOTES_independent_assessment.md` |
| KDFlow internals map | `NOTES_kdflow_combine_point.md` |

## Reading a diagnostics report

```json
"groups": { "_global": {
    "pairwise_cos": {"math|medical": -0.18},   // negative = opposing pulls
    "rho": 0.61,                               // vs null 1/sqrt(2)=0.707: below null → conflict
    "rho_null_orthogonal": 0.707,
    "norms": {"math": 4.1, "medical": 3.8} },  // imbalance check
  "lm_head": { ... }, "mlp": { ... } }         // WHERE the conflict lives
"split_half_reliability": {"math": 0.42}       // <~0.2 → increase K before concluding
"disattenuated_alignment": {"math|medical": -0.44}   // the corrected conflict number
```

---

# Glossary

- **branch** — a copy of the student trained on one domain only, from a shared snapshot.
- **Cayley transform** — the map `R = (I−Q)(I+Q)⁻¹` between skew-symmetric matrices and rotations near the identity; our bridge into "angle coordinates."
- **ceiling** — a domain's single-teacher branch score at matched budget; the denominator of "recovery."
- **delta (Δ) / task vector** — trained weights minus base weights.
- **distillation** — training a student model to match a teacher's output *distributions*.
- **exposure bias** — errors compounding at inference because training never showed the model its own mistakes; the problem on-policy training removes.
- **FDA (functional dual anchors)** — representing a model-change as synthetic *inputs* whose induced gradient reproduces it (Weiyang's paper; a possible later arm).
- **forward/reverse KL** — the two orientations of KL divergence; mode-covering vs mode-seeking (§2).
- **KDFlow** — the open-source on-policy distillation engine we build on (vendored in `KDFlow/`).
- **logits** — a model's raw pre-softmax scores, one per vocabulary token.
- **min-over-domains** — worst domain's fraction of its ceiling; our headline metric.
- **mode-seeking** — concentrating on part of the teacher's distribution confidently rather than covering all of it thinly; reverse KL's behavior.
- **Muon / Newton–Schulz** — fast polynomial iterations that push a matrix's singular values to 1, approximating the polar factor; Muon is the optimizer built on them.
- **OFT (orthogonal finetuning)** — finetuning where the update is constrained to be a rotation: `W = R·W0`, R orthogonal.
- **on-policy distillation (OPD)** — the student generates; the teacher grades the student's own tokens.
- **ORD (orthogonal-residual decoupling)** — `W_d = R_d·W0 + E_d` with R_d the Procrustes rotation; our V2.
- **orthogonal matrix** — `RᵀR = I`; preserves lengths and angles; rotation if det=+1, reflection if det=−1.
- **OrthoMerge** — Weiyang's method merging specialists' rotations in so(n) with the ×c correction; the template we move in-loop.
- **polar decomposition** — `A = Q·H`: orthogonal direction × symmetric stretch; matrix version of `z = e^{iθ}r`.
- **Procrustes problem** — find the orthogonal matrix best aligning one matrix to another; solved by the polar factor of their product.
- **reliability (split-half)** — cosine between deltas from two halves of the same domain's data; the noise floor for conflict claims.
- **RoPE / context window** — positional encoding; beyond its trained length a model's outputs are extrapolation garbage (why the math teacher is capped at 4,096).
- **rollout** — a response the student generates during training.
- **routing** — sending each prompt to its domain's teacher.
- **see-saw** — one domain improving while another regresses, from conflicting shared-parameter updates.
- **skew-symmetric / so(n)** — `Qᵀ=−Q`; the flat "angle space" where rotations near I can be averaged.
- **SVD** — `A = USVᵀ`; supplies both the polar factor (UVᵀ) and the stretch (S).
- **TA (task arithmetic)** — weighted addition of deltas.
- **TIES** — trim → elect sign → merge agreeing entries; discrete anti-cancellation merging.
- **×c correction** — rescaling the averaged skew matrix so its norm equals the average input norm; restores energy lost to cancellation.

# FAQ

**Q: Why not just train the student on all five domains' data directly?**
Because the teachers *are* the curriculum: they were RL-trained/specialized in
ways raw data doesn't capture, and distillation transfers full distributions
(thousands of numbers per token), which is far more sample-efficient than
next-token supervision. Data mixing also has its own see-saw problem — it
just moves the question.

**Q: Why is the student smaller than the teachers?**
That's the product shape: specialist quality at deployable cost. It also
sharpens the science — a 1.5B student *cannot* absorb everything from five 7B
teachers, so how the combination spends its limited capacity really matters.

**Q: Isn't this just model merging?**
Merging (TA/TIES/OrthoMerge) combines *finished* models' weights, post hoc.
We interleave training and geometric combination — branches stay close, merges
happen in the regime where the math is valid, and rollouts stay on-policy.
Post-hoc merging is, however, exactly the right *cheap pre-test*, which is
what the offline bake-off is.

**Q: Why not use gradient-surgery methods like PCGrad?**
Per-step gradients are the wrong object (§5): noisy, microscopic, and the
literature (Kurin et al. 2022; Xin et al. 2022) finds tuned plain weighting
matches gradient surgery at scale. We operate on accumulated deltas, where
signal-to-noise is high and geometry is non-degenerate.

**Q: What if the diagnostics find no conflict at all?**
Then the honest headline changes: "when does multi-teacher OPD need more than
a weighted sum?" — with the first careful measurement apparatus for the
question, teacher-distance as the controlled axis, and the geometric toolkit
evaluated fairly. A pre-registered null with a mechanism is publishable; a
method fixing a non-problem is not.

**Q: The 5-teacher run improved everything already — why continue?**
That run was 80 steps with mildly-specialized teachers scored off-template —
gentle conditions. The sweep tests harder conditions (genuinely specialized
teachers, native templates, proper caps, longer accumulation) *and* builds
the instruments to detect conflict if it appears. Either outcome is
informative; only building-the-method-first-and-hoping is not.

**Q: Why reverse KL rather than forward?**
Mode-seeking suits small students (§2): confident subset > thin everything,
and junk mass is punished hardest, which suppresses hallucination. Forward KL
(and mixtures like JSD) are one config flag away if a teacher proves too
sharp to approach.

**Q: What are the biggest known risks, honestly?**
(1) No conflict exists at this scale → the method has no target (that's what
the gates are for). (2) Rotation-only capacity is insufficient (Gate A′
answers this in week one). (3) The geometric merge loses to plain averaging
offline → in-loop engineering unjustified (the bake-off answers this before
we build it). Each risk has a cheap, early, pre-registered test — that is the
design philosophy of the whole sweep.

# References

GKD — Agarwal et al., 2023 (on-policy KD) · MiniLLM — Gu et al., 2024
(reverse-KL KD) · OFT — Qiu et al., NeurIPS 2023 · OFTv2 — arXiv 2506.19847 ·
POET — arXiv 2506.08001 · OrthoMerge — arXiv 2602.05943 · FDA — arXiv
2510.21223 · AMO/Muon — arXiv 2605.17806; Jordan et al. 2024 · Task
Arithmetic — Ilharco et al., 2022 · TIES — Yadav et al., 2023 · gradient-
surgery skepticism — Kurin et al. 2022 (2201.04122), Xin et al. 2022
(2209.11379) · CaMOPD — arXiv 2605.27115 · KDFlow —
github.com/songmzhang/KDFlow.
