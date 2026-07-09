# KDFlow on-policy multi-teacher: the gradient combine point (where our mods go)

Verified by reading the cloned repo at `MOPD/KDFlow` (HEAD from songmzhang/KDFlow, 2026-07).
This is the Phase-0 deliverable: the exact place the per-domain (routed) signal becomes one
optimizer step. Two mods hang off it: (1) per-teacher weighting baseline [Phase 3];
(2) direction/magnitude-decoupled aggregation [Phase 4].

## Entry + loop
- CLI: `kdflow/cli/train_kd_on_policy.py` (main at ~L166 builds `OnPolicyKDTrainer`).
- Sampler: `train_kd_on_policy.py:116` — plain `DistributedSampler(shuffle=True, drop_last=True)`
  over ALL prompts. **No per-domain quota** → each domain's share = its share of the data.
  (Phase-0 balance knob = how much data per domain; a stratified sampler is a later add.)
- Trainer loop: `kdflow/trainer/on_policy_kd_trainer.py:148-254`.
  - (a) rollout: `trainer.py:172` `self.rollout(prompt_batch, ...)`
  - (b) teacher forward (hidden states): `trainer.py:181` `self.teacher.forward(rollout_samples)`

## The combine point (THE anchor)
`kdflow/ray/train/student_actor.py:286-327` — per-micro-batch loop:
- L309 `loss_info = self.kd_algorithm.training_step(micro_batch)`  ← loss compute
- L313 `loss = loss_info["loss"]`
- L314 `self.strategy.backward(loss, self.student, self.optim)`     ← backward (scaled by 1/accum)
- L321-325 grad-norm record
- L327 `self.strategy.optimizer_step(self.optim, self.student, self.scheduler)` ← step + zero_grad
- FSDP strategy: `kdflow/backend/fsdp/fsdp_strategy.py:347-369`
  - `backward()` divides loss by `accumulated_gradient` then `loss.backward()`
  - `optimizer_step()` steps + `zero_grad()` only when the accumulation boundary is reached.

**Combine semantics = plain Euclidean sum via gradient accumulation.** Each micro-batch's
grads accumulate into one FSDP shard; a single `optimizer.step()` applies the blended update.
No per-domain separation anywhere. This is exactly the baseline the project wants to beat.

## Multi-teacher routing
- Config load + restriction: `kdflow/arguments/__init__.py:96-113`. Loads
  `{routing_key: teacher_path}`; **raises unless `kd_algorithm == vanilla_kd`** (L109-113).
- Routing field carried through data: `kdflow/datasets/prompts_dataset.py` (~L115-200) reads
  `--teacher_routing_key` field per sample, keeps it through collate.
- Split/route/re-merge: `kdflow/ray/train/multi_teacher_group.py:56-88`
  - splits global batch by `teacher_routing_key`, sends each key's samples to its TeacherActorGroup,
  - **re-concatenates teacher hidden states back into original batch order** → a micro-batch can be
    MIXED-domain; teacher hiddens are per-sample.
- Per-domain logits in loss: `kdflow/algorithms/vanilla_kd.py:24-52,81-89` —
  `compute_multi_teacher_logits()` routes hiddens by key to each teacher's `lm_head` (CUDA streams),
  combines into one logits tensor → one `training_step` loss per micro-batch.

### ⚠️ Important nuance for Phase 4
The handoff assumed per-domain grads "naturally arrive as separate per-domain micro-batch
gradients." In this code, micro-batches are **NOT domain-pure by default** — routing is per-sample
and re-merged, so one micro-batch's single loss already blends domains. To obtain clean per-domain
gradients for geometry-aware aggregation we must either:
  (i) enforce domain-pure micro-batches (a stratified/grouped sampler + no cross-key mixing), or
  (ii) compute per-domain masked losses and separate backward passes per domain before combining.
The handoff's named helpers `_sort_opd_batch_by_data_source` / `_route_micro_batch` were NOT found
in this HEAD (possibly older/off-policy path) — flag & re-verify before relying on them.

## Losses available (`kdflow/loss/`)
`kl` (fkl, default), `rkl` (reverse-KL), `jsd`, `srkl` (skewed rkl), `skl`, `akl`, `tvd`, `top1_ce`,
`hrl`. NOTE multi-teacher is restricted to `vanilla_kd` algorithm, but `--kd_loss_fn` still selects
the divergence. rkl is the far-teacher collapse risk → jsd/srkl are the mitigations.

## Key args + defaults
- rollout: `rollout_batch_size`=32, `n_samples_per_prompt`=1 (`arguments/rollout_args.py`)
- training: `train_batch_size`=128, `micro_train_batch_size`=1, `num_epochs`=1 (`arguments/training_args.py`)
- distill: `kd_ratio`=0.5 → loss=`(1-kd_ratio)*nll + kd_ratio*kd`; `kd_temperature`=1.0;
  `multi_teacher_config`=None (`arguments/distillation_args.py`)
- Example recipe overrides: train_batch_size 128, micro 8, lr 2e-6, kd_ratio 1.0, kd_loss_fn rkl,
  enable_sleep True (colocate), rollout_num_engines 8.
