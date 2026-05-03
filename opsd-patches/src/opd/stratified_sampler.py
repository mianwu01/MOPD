"""Stratified-by-data_source sampler for OPSD multi-teacher OPD.

Goal: every batch of size BS contains exactly per_source[s] rows for each
data_source s. With 5 teachers × per_source=64 and bs=320, every batch is
{math:64, medical:64, code:64, tool:64, search:64} → 0% drop, 0 sampling
variance, perfect routing balance.

Usage (in verl data config):
  data:
    train_batch_size: 320
    shuffle: true   # ignored when sampler.class_path set
    sampler:
      class_path: opd.stratified_sampler
      class_name: StratifiedDataSourceSampler
      per_source:
        math_dapo: 64
        medical_o1: 64
        code_kodcode: 64
        tool_star: 64
        search_nqhq: 64
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator, Sized

from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractSampler


class StratifiedDataSourceSampler(AbstractSampler):
    """Yield indices in stratified batches: each batch has fixed per-source counts.

    Sampler emits indices linearly; DataLoader groups every `train_batch_size`
    consecutive indices into a batch. For correctness, sum(per_source.values())
    MUST equal train_batch_size.
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.dataset = data_source
        self.data_config = data_config

        # Read stratification config
        sampler_cfg = data_config.get("sampler", {}) or {}
        per_source_cfg = sampler_cfg.get("per_source", None)
        if per_source_cfg is None:
            raise ValueError(
                "StratifiedDataSourceSampler requires data.sampler.per_source: "
                "{<data_source>: <count_per_batch>, ...}"
            )
        # Convert OmegaConf to plain dict
        try:
            self.per_source = dict(per_source_cfg)
            self.per_source = {str(k): int(v) for k, v in self.per_source.items()}
        except Exception as e:
            raise ValueError(f"per_source must be a {{str: int}} mapping; got {per_source_cfg}") from e

        # Verify sum equals batch size
        bs = data_config.get("train_batch_size") or sum(self.per_source.values())
        sum_pc = sum(self.per_source.values())
        if sum_pc != bs:
            raise ValueError(
                f"sum(per_source values)={sum_pc} != train_batch_size={bs}. "
                f"Each batch must consume exactly bs rows."
            )

        # Bucket indices by data_source
        self.buckets: dict[str, list[int]] = defaultdict(list)
        for i in range(len(self.dataset)):
            row = self.dataset[i]
            ds = self._extract_data_source(row)
            self.buckets[ds].append(i)

        # Sanity-check we have data for every source the user specified
        missing = [s for s in self.per_source if s not in self.buckets]
        if missing:
            raise ValueError(
                f"per_source references data_sources not present in dataset: {missing}. "
                f"Available: {sorted(self.buckets.keys())}"
            )

        # Compute number of full batches we can emit per epoch
        # = min over teachers of (bucket_size / per_source[teacher])
        self.batches_per_epoch = min(
            len(self.buckets[s]) // self.per_source[s] for s in self.per_source
        )
        if self.batches_per_epoch == 0:
            raise ValueError(
                "No full stratified batch possible: "
                f"per_source={self.per_source}, bucket sizes={ {s: len(v) for s, v in self.buckets.items()} }"
            )

        # Seeding
        self.base_seed = int(data_config.get("seed") or 0)
        self.epoch = 0

        import logging
        logging.getLogger(__name__).info(
            "[StratifiedSampler] %d sources, %d batches/epoch (bs=%d). per_source=%s",
            len(self.per_source), self.batches_per_epoch, bs, dict(self.per_source),
        )

    @staticmethod
    def _extract_data_source(row) -> str:
        """Get data_source string from a dataset row (handles dict / dataclass / DataItem)."""
        if isinstance(row, dict):
            ds = row.get("data_source")
        else:
            ds = getattr(row, "data_source", None)
        if ds is None:
            # Some verl rows store it under 'extra_info' or 'non_tensor_batch'
            if isinstance(row, dict):
                ds = (row.get("extra_info") or {}).get("data_source")
        if ds is None:
            raise KeyError("dataset row missing 'data_source' field")
        if hasattr(ds, "item"):
            ds = ds.item()
        return str(ds)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.base_seed + self.epoch)
        # Shuffle each bucket independently
        shuffled = {s: list(idxs) for s, idxs in self.buckets.items()}
        for s in shuffled:
            rng.shuffle(shuffled[s])
        cursors = {s: 0 for s in self.buckets}

        for _ in range(self.batches_per_epoch):
            batch_indices: list[int] = []
            # Stable order over per_source keys (sorted by name) for reproducibility
            for s in sorted(self.per_source):
                k = self.per_source[s]
                batch_indices.extend(shuffled[s][cursors[s] : cursors[s] + k])
                cursors[s] += k
            for idx in batch_indices:
                yield idx
        self.epoch += 1

    def __len__(self) -> int:
        bs = sum(self.per_source.values())
        return self.batches_per_epoch * bs
