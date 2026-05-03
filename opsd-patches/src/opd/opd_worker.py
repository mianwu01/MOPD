"""
OPD Worker: extends verl's ActorRolloutRefWorker with divergence-based training.

A separate teacher model (ref) and trainable student model (actor) see the same
input sequences. The teacher produces better distributions naturally (bigger/stronger).

Training step:
  1. Forward teacher (ref model, frozen) on teacher_input_ids -> teacher logits
  2. Forward student (actor model, trainable) on student_input_ids -> student logits
  3. Compute token-wise divergence along response positions
  4. Backward and step optimizer
"""

import contextlib
import logging

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.protocol import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.attention_utils import index_first_axis, rearrange, unpad_input
from verl.utils.device import get_device_id
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.torch_functional import entropy_from_logits_with_chunking
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from .losses import LOSS_FN_MAP, compute_teacher_token_stats

logger = logging.getLogger(__name__)


def _shift_loss_mask_right_per_sequence(loss_mask_rmpad: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Shift loss-mask labels within each unpadded sequence only.

    The padded path uses ``loss_mask[:, 1:]`` so each logit at position ``t``
    is trained against whether token ``t + 1`` belongs to the response. After
    unpadding, we must preserve that same next-token alignment without letting
    the final token of one sample spill into the first token of the next.
    """
    shifted = torch.zeros_like(loss_mask_rmpad)
    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        start = int(seq_start.item())
        end = int(seq_end.item())
        if end - start > 1:
            shifted[start : end - 1] = loss_mask_rmpad[start + 1 : end]
    return shifted


def _shift_ids_within_sequences(ids_rmpad: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Get next-token target IDs within each unpadded sequence.

    For each sequence, target[t] = ids[t+1]. The last token of each sequence
    gets 0 (won't be selected by loss_mask anyway).
    """
    shifted = torch.zeros_like(ids_rmpad)
    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        s, e = int(seq_start.item()), int(seq_end.item())
        if e - s > 1:
            shifted[s : e - 1] = ids_rmpad[s + 1 : e]
    return shifted


class OPDWorker(AsyncActorRolloutRefWorker):
    """Worker with on-policy distillation training on top of verl's actor+rollout+ref.

    Inherits actor_module_fsdp (student) and ref_module_fsdp (teacher) from
    the HybridEngine ActorRolloutRef worker. Both teacher and student see the
    same input sequences (no privileged information).
    """

    def __init__(self, config, role: str, **kwargs):
        # The parent trainer's init_workers() creates hybrid-engine workers with
        # role="actor_rollout", which sets _is_ref=False and skips building
        # ref_module_fsdp.  OPD needs the colocated ref model as the teacher,
        # so we promote the role to "actor_rollout_ref".
        if role == "actor_rollout":
            role = "actor_rollout_ref"
        super().__init__(config=config, role=role, **kwargs)
        # Multi-teacher state, populated lazily in init_model() once the primary
        # ref_module_fsdp is built. Single-teacher OPD leaves these unset / empty.
        self.extra_ref_modules: dict = {}
        self._teacher_modules: dict = {}
        self._teacher_for_data_source: dict[str, str] = {}
        self._default_teacher_name: str | None = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        self._build_extra_teachers()

    @contextlib.contextmanager
    def _preserve_tokenizer(self):
        # `_build_model_optimizer` overwrites self.tokenizer / self.processor for
        # each model it loads. We need the student's (== primary teacher's, since
        # they share the Qwen2.5 family vocab) to remain authoritative after init.
        primary_tokenizer = self.tokenizer
        primary_processor = getattr(self, "processor", None)
        try:
            yield
        finally:
            self.tokenizer = primary_tokenizer
            if primary_processor is not None:
                self.processor = primary_processor

    def _assert_vocab_compat(self, model_path: str, name: str) -> None:
        # Crash early if a teacher's vocab is *smaller* than the student's: we
        # cannot truncate teacher to match (would lose probability mass on tokens
        # the student CAN predict). Equal-or-larger is fine — losses._align_vocab
        # truncates teacher down to student's V at loss-compute time.
        from transformers import AutoConfig
        teacher_cfg = AutoConfig.from_pretrained(
            model_path, trust_remote_code=self.config.model.get("trust_remote_code", False),
        )
        student_v = self.actor_model_config.vocab_size
        teacher_v = teacher_cfg.vocab_size
        if teacher_v < student_v:
            raise ValueError(
                f"[OPD-MT] teacher '{name}' has vocab_size={teacher_v} < student vocab={student_v}. "
                f"Cannot align losses — pick a teacher whose vocab covers the student's."
            )
        if teacher_v != student_v:
            logger.warning(
                "[OPD-MT] teacher '%s' vocab=%d > student vocab=%d. losses._align_vocab will "
                "truncate teacher logits to student's V at compute time.",
                name, teacher_v, student_v,
            )

    def _build_extra_teachers(self):
        """Stage 2: build N-1 extra ref FSDP modules for `opd_mt.teachers[1:]`.

        verl passes `config.actor_rollout_ref` as the worker's `self.config`, so
        multi-teacher config must live UNDER actor_rollout_ref. We use
        `actor_rollout_ref.opd_mt.teachers` (worker-visible) — distinct from the
        trainer-side `opd.*` block (loss_type, reward_fns, etc).

        Layout assumed: `opd_mt.teachers[0].path` already wired into
        `actor_rollout_ref.ref.model.path`, so verl built it as `ref_module_fsdp`
        in super().init_model(). We append the rest here.
        """
        opd_mt_cfg = self.config.get("opd_mt", None)
        if opd_mt_cfg is None:
            return
        teachers_cfg = opd_mt_cfg.get("teachers", None)
        if teachers_cfg is None:
            return
        teachers_cfg = list(teachers_cfg)
        if len(teachers_cfg) == 0:
            return
        if not getattr(self, "_is_ref", False) or not hasattr(self, "ref_module_fsdp"):
            raise RuntimeError(
                "opd.teachers requires the OPDWorker to own ref_module_fsdp "
                "(role 'actor_rollout_ref'). Got role without ref slot."
            )

        primary_path = teachers_cfg[0].get("path")
        wired_ref_path = self.config.ref.get("model", {}).get("path") or self.config.model.path
        if primary_path and primary_path != wired_ref_path:
            logger.warning(
                "[OPD-MT] opd_mt.teachers[0].path=%s differs from actor_rollout_ref.ref.model.path=%s. "
                "verl will distill samples for that teacher from the wired path, not opd_mt.teachers[0].",
                primary_path, wired_ref_path,
            )
        print(f"[OPD-MT] rank={getattr(self, 'rank', '?')} multi-teacher config seen: "
              f"{len(teachers_cfg)} teachers, primary={teachers_cfg[0]['name']}", flush=True)

        # Vocab compat check for ALL teachers (incl. primary) before building.
        self._assert_vocab_compat(primary_path or wired_ref_path, teachers_cfg[0]["name"])
        for entry in teachers_cfg[1:]:
            self._assert_vocab_compat(entry["path"], entry["name"])

        from verl.utils.config import omega_conf_to_dataclass

        override_model_config = OmegaConf.to_container(
            OmegaConf.create(self.config.model.get("override_config", {}))
        )
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        use_shm = self.config.model.get("use_shm", False)
        ref_offload = self.config.ref.fsdp_config.get("param_offload", False)

        with self._preserve_tokenizer():
            for entry in teachers_cfg[1:]:
                name = entry["name"]
                path = entry["path"]
                local_path = copy_to_local(path, use_shm=use_shm)
                logger.info("[OPD-MT] building extra teacher %s from %s", name, local_path)
                # role='ref' + optim_config=None ⇒ no optimizer is built (verl
                # fsdp_workers.py line 647 gates on role=='actor' AND optim
                # config). FSDP wrapped model only.
                mod = self._build_model_optimizer(
                    model_path=local_path,
                    fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                    optim_config=None,
                    override_model_config=override_model_config,
                    use_remove_padding=use_remove_padding,
                    use_fused_kernels=use_fused_kernels,
                    trust_remote_code=self.config.model.get("trust_remote_code", False),
                    use_liger=self.config.model.get("use_liger", False),
                    role="ref",
                )[0]
                if ref_offload:
                    offload_fsdp_model_to_cpu(mod)
                self.extra_ref_modules[name] = mod

        primary_name = teachers_cfg[0]["name"]
        self._teacher_modules = {primary_name: self.ref_module_fsdp, **self.extra_ref_modules}
        self._teacher_for_data_source = {
            ds: t["name"] for t in teachers_cfg for ds in t.get("data_sources", []) or []
        }
        self._default_teacher_name = opd_mt_cfg.get("default_teacher") or primary_name
        logger.info(
            "[OPD-MT] multi-teacher routing active. teachers=%s, data_source map=%s, default=%s",
            list(self._teacher_modules), self._teacher_for_data_source, self._default_teacher_name,
        )

    def _route_micro_batch(self, micro_batch: DataProto) -> str | None:
        # Single-teacher path: special marker, update_opd uses ref_module_fsdp.
        if not self._teacher_modules:
            return "__primary__"
        ds_arr = micro_batch.non_tensor_batch.get("data_source")
        if ds_arr is None or len(ds_arr) == 0:
            return self._default_teacher_name

        # Map every row to its teacher. Drop padded rows (valid_row_mask=False) from
        # the routing decision — they inherit the trailing source from padding.
        valid_mask = micro_batch.batch.get("valid_row_mask")
        if valid_mask is not None:
            valid_idx = valid_mask.bool().nonzero(as_tuple=True)[0].tolist()
            valid_sources = [str(ds_arr[i]) for i in valid_idx]
        else:
            valid_sources = [str(s) for s in ds_arr]
        if not valid_sources:
            return self._default_teacher_name

        teachers = {self._teacher_for_data_source.get(s, self._default_teacher_name)
                    for s in valid_sources}
        if len(teachers) == 1:
            return next(iter(teachers))
        # Boundary mb: rows split across teachers after upstream sort. Drop the
        # whole mb instead of majority-voting (would pollute minority rows with
        # the wrong teacher's logits). At ubs=2 the loss is at most ubs-1 = 1
        # row per boundary, ≤ 1 boundary mb per step after stable sort.
        return None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_opd(self, data: DataProto) -> DataProto:
        """One OPD training step: divergence between teacher and student.

        Uses a two-phase approach to avoid holding both models on GPU simultaneously:
          Phase 1: Load teacher (ref) → run all teacher forwards → cache logits on CPU → offload teacher
          Phase 2: Load student (actor) + optimizer → train with cached teacher logits → offload

        Input DataProto batch keys:
          teacher_input_ids, teacher_attention_mask, teacher_position_ids, teacher_loss_mask
          student_input_ids, student_attention_mask, student_position_ids, student_loss_mask

        Meta info:
          opd_loss_type: "reverse_kl" | "forward_kl" | "jsd"
          opd_beta: JSD interpolation (only used for jsd)
          opd_chunk_size: tokens per chunk for loss computation
        """
        assert self._is_actor, "update_opd requires actor role"

        def _mem(tag):
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info("[OPD-MEM] %s: allocated=%.2f GB, reserved=%.2f GB", tag, alloc, reserved)

        ref_needs_offload = self.config.ref.fsdp_config.get("param_offload", False)

        data = data.to("cpu")
        loss_type = data.meta_info.get("opd_loss_type", "reverse_kl")
        beta = data.meta_info.get("opd_beta", 0.5)
        chunk_size = data.meta_info.get("opd_chunk_size", 512)

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        micro_batch_size = self.config.actor.get(
            "ppo_micro_batch_size_per_gpu",
            self.config.actor.get("micro_batch_size_per_gpu", 2),
        )
        device = get_device_id()

        batch_size = data.batch["student_input_ids"].shape[0]
        if batch_size == 0:
            return DataProto(meta_info={"metrics": {"opd/loss": 0.0, "opd/num_tokens": 0}})

        micro_batches = list(data.split(micro_batch_size))
        logger.info("[OPD-MEM] batch_size=%d, micro_batches=%d, micro_batch_size=%d, ref_needs_offload=%s",
                     batch_size, len(micro_batches), micro_batch_size, ref_needs_offload)
        _mem("before-ulysses")

        with self.ulysses_sharding_manager:
            # ------------------------------------------------------------------
            # Phase 1: Teacher forward — only ref model on GPU
            # ------------------------------------------------------------------
            forward_fn = self._forward_logits_unpadded if use_remove_padding else self._forward_logits_padded
            # Each cache slot: (teacher_logits_cpu, is_valid, teacher_name).
            # teacher_name lets phase 2 break out per-teacher loss/entropy.
            teacher_logits_cache: list = [(None, False, None)] * len(micro_batches)

            # Group micro_batches by routed teacher (single-teacher path uses the
            # special "__primary__" key and resolves to ref_module_fsdp).
            # _route_micro_batch returns None for cross-domain boundary mb's,
            # which we drop from phase 1 → cache stays (None, False) → phase 2 skips.
            buckets: dict[str, list[int]] = {}
            n_dropped_boundary_mb = 0
            for i, micro_batch in enumerate(micro_batches):
                tname = self._route_micro_batch(micro_batch)
                if tname is None:
                    n_dropped_boundary_mb += 1
                    continue
                buckets.setdefault(tname, []).append(i)
            logger.info("[OPD-MEM] phase1 buckets=%s, boundary_mbs_dropped=%d",
                        {k: len(v) for k, v in buckets.items()}, n_dropped_boundary_mb)

            # Iterate in stable, rank-consistent order. self._teacher_modules
            # is built identically on all ranks at init_model() time. This
            # protects against FSDP all-gather deadlock that happens if rank A
            # loads teacher X while rank B loads teacher Y at the same time.
            ordered_names = list(self._teacher_modules) if self._teacher_modules else ["__primary__"]
            for teacher_name in ordered_names:
                mb_indices = buckets.get(teacher_name, [])
                if not mb_indices:
                    # Empty bucket on this rank — but FSDP load_fsdp_model_to_gpu
                    # is collective. With chunk-interleave at trainer side, every
                    # rank should have ≥1 mb per teacher; this branch is a defensive
                    # noop only when the dataset is severely unbalanced.
                    continue
                if teacher_name == "__primary__":
                    teacher_module = self.ref_module_fsdp
                else:
                    teacher_module = self._teacher_modules[teacher_name]

                _mem(f"phase1-{teacher_name}-before-load")
                if ref_needs_offload:
                    load_fsdp_model_to_gpu(teacher_module)
                teacher_module.eval()
                _mem(f"phase1-{teacher_name}-after-load")

                for i in mb_indices:
                    micro_batch = micro_batches[i].to(device)
                    valid_row_mask = micro_batch.batch.get("valid_row_mask")
                    if valid_row_mask is not None:
                        valid_row_mask = valid_row_mask.bool()
                        if not valid_row_mask.any():
                            continue

                    t_input_ids = micro_batch.batch["teacher_input_ids"]
                    t_attention_mask = micro_batch.batch["teacher_attention_mask"]
                    t_position_ids = micro_batch.batch["teacher_position_ids"]
                    t_loss_mask = micro_batch.batch["teacher_loss_mask"]

                    if valid_row_mask is not None:
                        t_input_ids = t_input_ids[valid_row_mask]
                        t_attention_mask = t_attention_mask[valid_row_mask]
                        t_position_ids = t_position_ids[valid_row_mask]
                        t_loss_mask = t_loss_mask[valid_row_mask]

                    logger.info("[OPD-MEM] phase1 mb[%d] teacher=%s: teacher_ids shape=%s, response_tokens=%d",
                                i, teacher_name, list(t_input_ids.shape), int(t_loss_mask[:, 1:].sum().item()))

                    with torch.no_grad():
                        teacher_logits = forward_fn(
                            teacher_module, t_input_ids, t_attention_mask, t_position_ids, t_loss_mask
                        )
                    teacher_logits_cache[i] = (teacher_logits.to("cpu"), True, teacher_name)
                    del teacher_logits

                _mem(f"phase1-{teacher_name}-before-offload")
                if ref_needs_offload:
                    offload_fsdp_model_to_cpu(teacher_module)
                torch.cuda.empty_cache()
                _mem(f"phase1-{teacher_name}-after-offload")

            # ------------------------------------------------------------------
            # Phase 2: Student forward + loss + backward — actor + optimizer on GPU
            # ------------------------------------------------------------------
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
            _mem("phase2-after-actor-load")
            if self._is_offload_optimizer:
                load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=device)
            _mem("phase2-after-optimizer-load")

            use_sample_weights = "sample_weights" in data.batch
            metrics = self._opd_training_step(
                micro_batches, teacher_logits_cache,
                loss_type=loss_type, beta=beta, chunk_size=chunk_size,
                use_remove_padding=use_remove_padding, device=device, batch_size=batch_size,
                use_sample_weights=use_sample_weights,
            )

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["opd/lr"] = lr.item() if torch.is_tensor(lr) else lr
            if metrics["opd/num_tokens"] > 0:
                self.actor_lr_scheduler.step()
            else:
                metrics["opd/skipped_step"] = 1.0

            metrics["perf/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        return output

    def _opd_training_step(
        self,
        micro_batches: list,
        teacher_logits_cache: list,
        loss_type: str = "reverse_kl",
        beta: float = 0.5,
        chunk_size: int = 512,
        use_remove_padding: bool = False,
        device: int = 0,
        batch_size: int = 0,
        use_sample_weights: bool = False,
    ) -> dict:
        """Core OPD training with pre-computed teacher logits.

        Teacher logits are already cached on CPU from phase 1. This phase only
        has the student (actor) model + optimizer on GPU, avoiding OOM from
        holding both models simultaneously.

        For each micro-batch:
          1. Move cached teacher logits to GPU
          2. Student forward (trainable actor) on student_input_ids -> logits
          3. Compute divergence loss (chunk-wise for memory efficiency)
          4. Backward with gradient accumulation scaling
        """
        self.actor_module_fsdp.train()

        active_micro_batches = sum(1 for entry in teacher_logits_cache if entry[1])
        grad_accum = max(1, active_micro_batches)

        loss_fn = LOSS_FN_MAP[loss_type]
        forward_fn = self._forward_logits_unpadded if use_remove_padding else self._forward_logits_padded

        self.actor_optimizer.zero_grad()
        total_loss = 0.0
        total_entropy_sum = 0.0
        total_tokens = 0
        total_rows = 0
        # Per-teacher accumulators for multi-teacher metrics. {teacher: [loss_sum, tok_count, mb_count]}
        # Pre-populate with all configured teachers so each DP rank emits the
        # same key set — verl's reduce_metrics asserts identical keys across ranks.
        per_teacher_stats: dict[str, list[float]] = {
            tname: [0.0, 0, 0] for tname in self._teacher_modules
        }

        def _mem2(tag):
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info("[OPD-MEM] %s: allocated=%.2f GB, reserved=%.2f GB", tag, alloc, reserved)

        for i, (micro_batch, (cached_teacher_logits, is_valid, teacher_name)) in enumerate(
            zip(micro_batches, teacher_logits_cache, strict=True)
        ):
            if not is_valid:
                continue

            micro_batch = micro_batch.to(device)

            valid_row_mask = micro_batch.batch.get("valid_row_mask")
            if valid_row_mask is not None:
                valid_row_mask = valid_row_mask.bool()

            s_input_ids = micro_batch.batch["student_input_ids"]
            s_attention_mask = micro_batch.batch["student_attention_mask"]
            s_position_ids = micro_batch.batch["student_position_ids"]
            s_loss_mask = micro_batch.batch["student_loss_mask"]

            if valid_row_mask is not None:
                s_input_ids = s_input_ids[valid_row_mask]
                s_attention_mask = s_attention_mask[valid_row_mask]
                s_position_ids = s_position_ids[valid_row_mask]
                s_loss_mask = s_loss_mask[valid_row_mask]

            _mem2(f"phase2-mb{i}-before-teacher-to-gpu")
            teacher_logits = cached_teacher_logits.to(device)
            logger.info("[OPD-MEM] phase2 micro_batch[%d]: student_ids shape=%s, cached_teacher shape=%s",
                        i, list(s_input_ids.shape), list(teacher_logits.shape))
            _mem2(f"phase2-mb{i}-after-teacher-to-gpu")

            student_logits = forward_fn(
                self.actor_module_fsdp, s_input_ids, s_attention_mask, s_position_ids, s_loss_mask
            )
            logger.info("[OPD-MEM] phase2 micro_batch[%d]: student_logits shape=%s", i, list(student_logits.shape))
            _mem2(f"phase2-mb{i}-after-student-fwd")

            if teacher_logits.shape[0] != student_logits.shape[0]:
                raise RuntimeError(
                    "Teacher and student response token counts diverged. "
                    "This usually means prompt truncation dropped response tokens."
                )
            if teacher_logits.shape[0] == 0:
                del teacher_logits
                continue

            _mem2(f"phase2-mb{i}-before-loss")

            # Per-sample reward weighting: compute loss per sample, weight, then average
            mb_sample_weights = None
            if use_sample_weights and "sample_weights" in micro_batch.batch:
                mb_sample_weights = micro_batch.batch["sample_weights"]
                if valid_row_mask is not None:
                    mb_sample_weights = mb_sample_weights[valid_row_mask]
                mb_sample_weights = mb_sample_weights.to(device)

            if mb_sample_weights is not None:
                # Compute per-sample loss using loss_mask to identify sample boundaries
                # s_loss_mask: (n_valid_rows, seq_len), teacher/student_logits: (N_response_tokens, V)
                # We need to map response tokens back to samples
                per_sample_token_counts = s_loss_mask[:, 1:].sum(dim=1).long()  # tokens per sample
                sample_losses = []
                token_offset = 0
                for si in range(per_sample_token_counts.shape[0]):
                    n_tok = per_sample_token_counts[si].item()
                    if n_tok == 0:
                        sample_losses.append(torch.tensor(0.0, device=device))
                        continue
                    t_slice = teacher_logits[token_offset:token_offset + n_tok]
                    s_slice = student_logits[token_offset:token_offset + n_tok]
                    if loss_type == "jsd":
                        sl, _ = loss_fn(t_slice, s_slice, beta=beta, chunk_size=chunk_size)
                    else:
                        sl, _ = loss_fn(t_slice, s_slice, chunk_size=chunk_size)
                    sample_losses.append(sl)
                    token_offset += n_tok
                sample_losses = torch.stack(sample_losses)
                loss = (sample_losses * mb_sample_weights).sum() / mb_sample_weights.sum()
                n_tokens = int(per_sample_token_counts.sum().item())
            else:
                if loss_type == "jsd":
                    loss, n_tokens = loss_fn(teacher_logits, student_logits, beta=beta, chunk_size=chunk_size)
                else:
                    loss, n_tokens = loss_fn(teacher_logits, student_logits, chunk_size=chunk_size)

            del teacher_logits
            _mem2(f"phase2-mb{i}-after-loss-del-teacher")

            # Compute per-token entropy of the student policy (no grad needed)
            with torch.no_grad():
                token_entropy = entropy_from_logits_with_chunking(student_logits.float(), chunk_size=chunk_size)
                total_entropy_sum += token_entropy.sum().item()

            scaled_loss = loss / grad_accum
            scaled_loss.backward()
            _mem2(f"phase2-mb{i}-after-backward")

            total_loss += loss.detach().item()
            total_tokens += n_tokens
            total_rows += s_input_ids.shape[0]

            if teacher_name is not None and teacher_name in per_teacher_stats:
                stats = per_teacher_stats[teacher_name]
                stats[0] += loss.detach().item()
                stats[1] += int(n_tokens)
                stats[2] += 1

        if total_tokens == 0:
            self.actor_optimizer.zero_grad()
            out = {
                "opd/loss": 0.0,
                "opd/entropy": 0.0,
                "opd/grad_norm": 0.0,
                "opd/num_tokens": 0,
                "opd/batch_size": int(batch_size),
                "opd/valid_rows": int(total_rows),
            }
            # Emit zero-valued per-teacher keys so reduce_metrics across ranks works.
            for tname in per_teacher_stats:
                out[f"opd/loss/{tname}"] = 0.0
                out[f"opd/num_tokens/{tname}"] = 0
                out[f"opd/num_micro_batches/{tname}"] = 0
            return out

        # Gradient clipping and optimizer step
        grad_clip = self.config.actor.get("grad_clip", 1.0)
        if isinstance(self.actor_module_fsdp, FSDP):
            grad_norm = self.actor_module_fsdp.clip_grad_norm_(max_norm=grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module_fsdp.parameters(), max_norm=grad_clip
            )

        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()

        if torch.isfinite(grad_norm):
            self.actor_optimizer.step()
        else:
            logger.warning("Non-finite grad_norm (%.4f), skipping step", grad_norm.item())
            self.actor_optimizer.zero_grad()

        out = {
            "opd/loss": total_loss / max(1, active_micro_batches),
            "opd/entropy": total_entropy_sum / max(1, total_tokens),
            "opd/grad_norm": grad_norm.detach().item(),
            "opd/num_tokens": int(total_tokens),
            "opd/batch_size": batch_size,
            "opd/valid_rows": int(total_rows),
        }
        # Per-teacher loss/token breakdown — multi-teacher diagnostics. In
        # single-teacher mode (teacher_name=='__primary__') this still emits a
        # single bucket; harmless and useful for sanity-checking.
        for tname, (lsum, ntok, nmb) in per_teacher_stats.items():
            out[f"opd/loss/{tname}"] = lsum / max(1, nmb)
            out[f"opd/num_tokens/{tname}"] = int(ntok)
            out[f"opd/num_micro_batches/{tname}"] = int(nmb)
        return out

    def _extract_response_target_ids_padded(self, input_ids, loss_mask):
        """Extract next-token target IDs at response positions (padded path)."""
        target_ids = input_ids[:, 1:]
        shift_mask = loss_mask[:, 1:]
        flat_target = target_ids.reshape(-1)
        flat_mask = shift_mask.reshape(-1)
        return flat_target[flat_mask.nonzero(as_tuple=True)[0]]

    def _extract_response_target_ids_unpadded(self, input_ids, attention_mask, loss_mask):
        """Extract next-token target IDs at response positions (unpadded path)."""
        _, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        ids_rmpad = input_ids.reshape(-1)[indices]
        target_ids_rmpad = _shift_ids_within_sequences(ids_rmpad, cu_seqlens)
        loss_mask_rmpad = loss_mask.reshape(-1)[indices]
        shifted_mask = _shift_loss_mask_right_per_sequence(loss_mask_rmpad, cu_seqlens)
        return target_ids_rmpad[shifted_mask.nonzero(as_tuple=True)[0]]

    def _forward_logits_padded(self, model, input_ids, attention_mask, position_ids, loss_mask):
        """Forward pass returning response-position logits (padded path).

        Returns: (N_response_tokens, vocab_size)
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = outputs.logits
            del outputs

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_loss_mask = loss_mask[:, 1:]
        del logits

        B, S, V = shift_logits.shape
        flat_logits = shift_logits.reshape(B * S, V)
        flat_mask = shift_loss_mask.reshape(B * S)
        del shift_logits

        response_indices = flat_mask.nonzero(as_tuple=True)[0]
        return flat_logits[response_indices]

    def _forward_logits_unpadded(self, model, input_ids, attention_mask, position_ids, loss_mask):
        """Forward pass returning response-position logits (unpadded/flash_attn path).

        Returns: (N_response_tokens, vocab_size)
        """
        input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        use_ulysses = self.ulysses_sequence_parallel_size > 1
        if use_ulysses:
            input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_rmpad,
                position_ids_rmpad=position_ids_rmpad,
                sp_size=self.ulysses_sequence_parallel_size,
            )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                use_cache=False,
            )
            logits_rmpad = outputs.logits.squeeze(0)
            del outputs

        if use_ulysses:
            logits_rmpad = gather_outputs_and_unpad(
                logits_rmpad,
                gather_dim=0,
                unpad_dim=0,
                padding_size=pad_size,
            )

        loss_mask_flat = loss_mask.reshape(-1)
        loss_mask_rmpad = loss_mask_flat[indices]

        shifted_loss_mask = _shift_loss_mask_right_per_sequence(loss_mask_rmpad, cu_seqlens)
        response_indices = shifted_loss_mask.nonzero(as_tuple=True)[0]
        return logits_rmpad[response_indices]
