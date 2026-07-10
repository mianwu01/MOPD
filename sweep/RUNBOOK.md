# MOPD Wide-Sweep Runbook (pre-Weiyang-meeting)

Goal: by meeting time, have in hand — per-domain branch checkpoints (full-param
AND OFT), the **offline merge bake-off** across every geometry variant, the
**delta-interference diagnostics**, the tuned multi-teacher baseline, and
Gate-A′ (OFT capacity parity). Whatever fork Weiyang picks (V1/V2/V3, merge
cadence), the data will already exist.

**What was verified where:** the merge library (38/38 unit tests), the bake-off
+ diagnostics end-to-end (synthetic checkpoints), the paired-stats tool, and the
KDFlow loss-weighting/context-cap logic (CPU smoke tests) were all verified in
the dev container. Training launches, data builds, and evals need the GPU
cluster — treat first launches as smoke runs (`--limit 200`, SAVE_STEPS=5).

---

## Stage 0 — environment + teacher screening (½ day, 8 GPUs)

```bash
pip install -e ./KDFlow   # brings torch/sglang per KDFlow's pins; H100 = sm90, cu128 wheels fine
ray start --head --num-gpus 8
```

1. **Teacher health check** (degenerate-checkpoint screen — the old yukang failure):
   `python3 teacher_health_check.py` over every model in `sweep/configs/teachers_5domain.json`.
2. **KL screen with native templates**: re-run the Control-A-style screen but with each
   teacher's OWN chat template / system prompt (the old screen scored teachers
   off-template — part of why they looked like vanilla 7B). Commit the numbers.
3. Verify the roster facts: `math` (Qwen2.5-Math-7B-Instruct) has a **4,096-token
   trained window** — the cap in `sweep/configs/teacher_max_len.json` plus
   `MAX_LEN=4096` in the launchers enforces it. Confirm `medical`
   (HuatuoGPT-o1-7B) is Qwen2.5-7B-Instruct-based and vocab-compatible.

## Stage 1 — data (½ day, CPU)

```bash
for d in math medical code search tool; do
  python3 sweep/build_kdflow_data.py --preset $d --out data/kdflow/$d.jsonl --limit 6000
done
# split halves for the reliability probe (math + medical suffice)
python3 sweep/build_kdflow_data.py --preset math    --out data/kdflow/math.jsonl    --limit 6000 --split_half
python3 sweep/build_kdflow_data.py --preset medical --out data/kdflow/medical.jsonl --limit 6000 --split_half
cat data/kdflow/{math,medical,code,search,tool}.jsonl | shuf > data/kdflow/mixed_5domain.jsonl
```
Spot-check 20 rows per file (presets are written from dataset docs, untested on
the live hub — field names may need a one-line fix).

## Stage 2 — branch fleet (1–2 days; the big parallel block)

Each job = 8 GPUs, ~half-day for 80 steps at bs128. With 40+ H100s run 5+ jobs
concurrently. `SAVE_STEPS=20` gives K-step snapshots at 20/40/60/80 for
merge-cadence sweeps.

| Job | Command (env-driven) | Purpose |
|---|---|---|
| 5 × full-param branches | `DOMAIN=$d TEACHER=<path> DATA=data/kdflow/$d.jsonl bash sweep/train_branch.sh` | ceilings + bake-off inputs + diagnostics |
| 2 × OFT branches (math, medical) | `... OFT_BLOCK=64 RUN_TAG=oft bash sweep/train_branch.sh` | **Gate A′** (capacity parity) + V1 inputs |
| 2 × 2 split-half branches (math, medical) | `DATA=data/kdflow/math.halfA.jsonl RUN_TAG=halfA ...` | split-half reliability (noise floor for Gate B) |
| lr probe (math only) | `LR=1e-6 RUN_TAG=fp_lr1e-6 ...` | sensitivity |

Math jobs keep `MAX_LEN=4096` (teacher window). If any branch shows collapse
signatures (entropy→0, length cap, repetition), stop it and log — that is data.

## Stage 3 — multi-teacher baselines (1 day, parallel with Stage 2 tail)

```bash
DATA=data/kdflow/mixed_5domain.jsonl RUN_TAG=uniform bash sweep/train_multiteacher.sh
# weighted variants (the "tuned" opponent) — write JSON grids, e.g.:
echo '{"math":2.0,"medical":1.0,"code":1.0,"search":1.0,"tool":1.0}' > sweep/configs/w_math2.json
DATA=... WEIGHTS_JSON=sweep/configs/w_math2.json RUN_TAG=wmath2 bash sweep/train_multiteacher.sh
```
Grid suggestion: uniform, math×2, math×0.5, inverse-loss-norm (set from the
per-domain losses of the uniform run's first 10 steps). 4 runs × 8 GPUs.

## Stage 4 — offline merge bake-off (hours, 1 GPU or CPU)

```bash
# (OFT branches first:  python3 sweep/materialize_peft.py --base <student> --adapter <ckpt> --out <full>)
python3 sweep/offline_merge_bakeoff.py \
  --base <student_dir> \
  --branch math=<...>/epoch_1_global_step_80 --branch medical=<...> \
  --branch code=<...> --branch search=<...> --branch tool=<...> \
  --out_dir output/merged_step80
# repeat with global_step_20/40 snapshots for the merge-cadence axis
```
Produces one HF checkpoint per operator:
`plain_avg, ta, ties, v2_ord, v2_ord_noxc, v2_dir_only, v2_mag_only, v3_polar`.

## Stage 5 — diagnostics (minutes, CPU)

```bash
python3 sweep/delta_diagnostics.py --base <student_dir> \
  --branch math=<...> --branch medical=<...> --branch code=<...> \
  --branch search=<...> --branch tool=<...> \
  --half math=<halfA_ckpt>,<halfB_ckpt> --half medical=<halfA>,<halfB> \
  --out output/diagnostics_step80.json
```
Pre-registered readings: pairwise cos ≈ 0 within the split-half null and
rho ≈ 1/√5 ≈ 0.447 → **no measurable conflict** (Gate-B stop for the conflict
story; the merge question remains). cos ≪ 0 beyond null → conflict exists;
note WHERE (per-group report: lm_head/embed vs mlp/attn).

## Stage 6 — eval everything, paired (1 day, parallel)

Eval every artifact on the per-domain suites (MATH-500, MedQA, MBPP+/HumanEval+,
NQ/HotpotQA subset, tool suite): base student, 5 single-teacher branches
(= matched-budget ceilings), OFT branches (Gate A′ vs their full-param twins),
multi-teacher baselines, all 8 merged models. Emit per-problem JSONL
(`{"id":…,"correct":…}`), then:

```bash
python3 sweep/paired_compare.py --a results/v2_ord_math500.jsonl --b results/ta_math500.jsonl
```
Decision rules (pre-registered):
- **Matched-budget ceilings (weak-see-saw accounting)**: the multi-teacher run
  (bs 320, ~64/domain/step × 80 = ~5,120 samples/domain) is compared against
  each branch's **step-40** snapshot (128 × 40 = 5,120) — same per-domain
  budget. Recovery_d = multi_d / branch_d@40. Branch step-80 (10,240) is
  reported as the 2×-budget ceiling. A "weak see-saw" claim = recovery
  significantly < 1 under paired stats; comparing multi against a full-budget
  or multi-epoch solo run (e.g. a v10g-style 4-epoch number) conflates
  interference with data dilution and is not evidence.
- **Gate A′**: OFT branch within ~1–2pp of full-param branch on its own domain,
  else rotation-only capacity is the binding constraint → tell Weiyang.
- **Bake-off**: if no geometric op beats `plain_avg`/`ta` (paired, 95%) on
  min-over-domains, in-loop geometry engineering is not justified.
- **Baseline check**: if the tuned multi-teacher baseline already recovers ≥ 97%
  of every ceiling, there is nothing for ANY merge method to recover at this scale.

## GPU budget summary

| Block | Jobs × GPUs | Wall |
|---|---|---|
| Stage 0–1 | 8 | ½ day |
| Stage 2 branch fleet | 10 × 8 (staggered on 40–48) | 1–2 days |
| Stage 3 baselines | 4 × 8 | 1 day |
| Stage 4–5 merge + diagnostics | ≤ 8 | hours |
| Stage 6 eval | any | 1 day |

## Known sharp edges

- KDFlow on-policy asserts `kd_ratio == 1.0` (pure KD) — launchers comply.
- Multi-teacher path requires identical vocab across student/teachers; KDFlow
  asserts this at startup. Qwen2.5 family: student vocab 151936 vs teacher
  152064 — KDFlow truncates logits to the min vocab inside `chunked_loss`.
- PEFT (OFT) checkpoints save as adapter `.bin` — materialize before merging.
- `--teacher_loss_weights` / `--teacher_max_len` are OUR additions (see
  `KDFlow/kdflow/{arguments,algorithms,loss}` diffs); token weights zero the
  loss beyond a teacher's window but the `avg_token_num` denominator is
  unchanged (capped samples are slightly downweighted — intended).
- v3_polar logs `det<0` warnings by design — that operator carries a known
  theoretical landmine (see NOTES_independent_assessment.md, Addendum 2) and is
  in the sweep to quantify it, not because we trust it.
