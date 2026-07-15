"""``build_mrts_loader`` — the offline-pretraining input pipeline.

Wraps :class:`MRTSSequenceDataset` in a ``torch.utils.data.DataLoader` configured
for throughput: worker processes prefetch and decode contiguous windows off disk
while the GPU trains, batches are pinned for async host->device copies, and each
worker gets its own h5py handle (the shared handle from ``__init__`` is closed;
``_worker_init`` clears the lazy handle so every worker reopens the file itself —
h5py handles are not fork-safe).

The default collate is exactly right here: each item is a ``dict[str, Tensor]`` of
``[T, ...]`` fields, which the default collate stacks into ``[B, T, ...]`` — the
shape the tokenizer / dynamics losses consume. Pretraining is step-based, so
:func:`cycle` turns the epoch loader into an endless batch stream and
:func:`to_device` moves a batch (non-blocking when pinned).

Usage::

    loader = build_mrts_loader(path, task="tokenizer", seq_len=16, batch_size=32)
    for step, batch in zip(range(n_steps), cycle(loader)):
        batch = to_device(batch, device)
        loss, metrics, z = tokenizer_loss(model, batch["obs"], batch["mask"])
"""

from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler

from .mrts_dataset import MRTSSequenceDataset


class H5ChunkShuffleSampler(Sampler):
    """Shuffle HDF5 storage blocks while keeping reads local within each block.

    Fully random row sampling is pathological for gzip-chunked stores: requesting
    one row decompresses the entire chunk. We randomize chunk order every epoch
    and rows inside each chunk, retaining stochasticity while allowing h5py's raw
    chunk cache to serve the other rows without decompression and disk rereads.
    """

    def __init__(self, dataset, chunk_rows: int):
        self.dataset = dataset
        starts = dataset._win_start
        chunk = starts // max(int(chunk_rows), 1)
        cuts = np.flatnonzero(chunk[1:] != chunk[:-1]) + 1
        self.bounds = np.concatenate(([0], cuts, [len(starts)]))

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        block_order = torch.randperm(len(self.bounds) - 1).tolist()
        for block in block_order:
            lo, hi = int(self.bounds[block]), int(self.bounds[block + 1])
            for offset in torch.randperm(hi - lo).tolist():
                yield lo + offset


class H5FixedChunkSampler(Sampler):
    """A repeatable, storage-local validation sample spanning the corpus.

    Validation should compare the same examples at every checkpoint, while HDF5
    still needs reads from the same compressed chunk to remain adjacent. Choose
    a fixed set of eligible chunks once and emit exactly one full batch from
    each. Rebuilding with the same seed produces the same sample.
    """

    def __init__(
        self,
        dataset,
        chunk_rows: int,
        batch_size: int,
        num_batches: int,
        seed: int = 0,
    ):
        starts = dataset._win_start
        chunk = starts // max(int(chunk_rows), 1)
        cuts = np.flatnonzero(chunk[1:] != chunk[:-1]) + 1
        bounds = np.concatenate(([0], cuts, [len(starts)]))
        eligible = np.flatnonzero(np.diff(bounds) >= int(batch_size))
        if not len(eligible):
            raise ValueError(
                "fixed-chunk validation found no HDF5 chunk with a full batch"
            )
        count = min(max(1, int(num_batches)), len(eligible))
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(eligible, size=count, replace=False))
        indices = []
        for block in selected:
            lo, hi = int(bounds[block]), int(bounds[block + 1])
            offsets = rng.choice(hi - lo, size=int(batch_size), replace=False)
            indices.extend((lo + np.sort(offsets)).tolist())
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        yield from self.indices


class H5PairedBatchSampler(Sampler):
    """Storage-local batches with a configured fraction of paired rows.

    Counterfactual branches are sparse enough that an ordinary batch of 32 has
    only a handful of causal interventions. This sampler oversamples rows whose
    stored ``counterfactual_valid`` flag is true while retaining HDF5 chunk
    locality. Sampling is with replacement when a chunk has fewer paired rows
    than requested; training is step-based, so epoch cardinality is not a data
    weighting contract.
    """

    def __init__(self, dataset, chunk_rows: int, batch_size: int, paired_fraction: float):
        import h5py

        fraction = float(paired_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("paired_fraction must be in (0, 1]")
        if "counterfactual_valid" not in dataset.fields:
            raise ValueError(
                "paired batch sampling requires the counterfactual_valid field"
            )
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.paired_per_batch = max(1, round(self.batch_size * fraction))
        self.paired_per_batch = min(self.paired_per_batch, self.batch_size)
        starts = dataset._win_start
        chunk = starts // max(int(chunk_rows), 1)
        cuts = np.flatnonzero(chunk[1:] != chunk[:-1]) + 1
        self.bounds = np.concatenate(([0], cuts, [len(starts)]))
        with h5py.File(dataset.path, "r", locking=dataset._locking) as f:
            # The flag column is only one byte per transition. Reading it once
            # and indexing in NumPy is dramatically faster than millions of
            # h5py point selections for the window-start array.
            self.paired = f["counterfactual_valid"][:].astype(bool)[starts]
        self.global_paired = np.flatnonzero(self.paired)
        self.global_unpaired = np.flatnonzero(~self.paired)
        if not len(self.global_paired):
            raise ValueError("paired batch sampling found no paired rows")
        self.num_batches = sum(
            int(np.ceil((int(hi) - int(lo)) / self.batch_size))
            for lo, hi in zip(self.bounds[:-1], self.bounds[1:])
        )

    def __len__(self):
        return self.num_batches

    @staticmethod
    def _draw(pool: np.ndarray, count: int) -> list[int]:
        if count <= 0:
            return []
        if not len(pool):
            return []
        if len(pool) >= count:
            order = torch.randperm(len(pool))[:count].numpy()
        else:
            order = torch.randint(len(pool), (count,)).numpy()
        return pool[order].tolist()

    def __iter__(self):
        for block in torch.randperm(len(self.bounds) - 1).tolist():
            lo, hi = int(self.bounds[block]), int(self.bounds[block + 1])
            local = np.arange(lo, hi)
            paired = local[self.paired[lo:hi]]
            unpaired = local[~self.paired[lo:hi]]
            if not len(paired):
                paired = self.global_paired
            if not len(unpaired):
                unpaired = self.global_unpaired
            batches = int(np.ceil((hi - lo) / self.batch_size))
            for _ in range(batches):
                indices = self._draw(paired, self.paired_per_batch)
                indices += self._draw(
                    unpaired, self.batch_size - self.paired_per_batch
                )
                if len(indices) != self.batch_size:
                    raise RuntimeError("could not construct a full paired batch")
                # h5py serves a sorted local batch much more cheaply; default
                # collation does not depend on within-batch row order.
                yield sorted(indices)


def _worker_init(worker_id):
    """Reset the (fork-inherited) h5py handle so each worker opens its own."""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._f = None


def build_mrts_loader(
    path,
    *,
    task,
    seq_len,
    batch_size,
    num_workers=4,
    shuffle=True,
    stride=1,
    map_ids=None,
    opponent_ids=None,
    policy_ids=None,
    drop_last=True,
    pin_memory=None,
    prefetch_factor=2,
    persistent_workers=None,
    locking=True,
    val_frac=0.0,
    split="train",
    split_seed=0,
    chunk_shuffle=True,
    fixed_chunk_batches=None,
    fixed_chunk_seed=0,
    paired_batch_fraction=None,
    h5_cache_mb=64,
):
    """Build a ``DataLoader`` of ``[B, T, ...]`` window batches for ``task``.

    ``task`` is ``"tokenizer"`` / ``"dynamics"`` / ``"all"`` (which fields to load).
    ``val_frac`` / ``split`` / ``split_seed`` carve a trajectory-level held-out
    split (same seed+frac from two loaders -> complementary sets).
    ``pin_memory`` / ``persistent_workers`` default to sensible values (pin iff CUDA
    is available; persist workers iff there are any). The dataset is reachable as
    ``loader.dataset`` for its ``obs_shape`` / ``action_nvec`` / attrs.
    """
    ds = MRTSSequenceDataset(
        path,
        seq_len=seq_len,
        task=task,
        stride=stride,
        map_ids=map_ids,
        opponent_ids=opponent_ids,
        policy_ids=policy_ids,
        locking=locking,
        val_frac=val_frac,
        split=split,
        split_seed=split_seed,
        h5_cache_mb=h5_cache_mb,
    )
    if len(ds) == 0:
        raise ValueError(
            f"no trajectory in {path!r} yields a window of seq_len={seq_len} "
            f"(after map/opponent filtering)"
        )

    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    persistent_workers = (
        (num_workers > 0) if persistent_workers is None else persistent_workers
    )
    kw = {}
    if num_workers > 0:
        kw["prefetch_factor"] = prefetch_factor
        kw["persistent_workers"] = persistent_workers
    sampler = None
    if paired_batch_fraction is not None:
        if fixed_chunk_batches:
            raise ValueError(
                "paired_batch_fraction and fixed_chunk_batches are mutually exclusive"
            )
        batch_sampler = H5PairedBatchSampler(
            ds, ds.chunk_rows, batch_size, paired_batch_fraction
        )
        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            worker_init_fn=_worker_init,
            pin_memory=pin_memory,
            **kw,
        )
    if fixed_chunk_batches:
        sampler = H5FixedChunkSampler(
            ds,
            ds.chunk_rows,
            batch_size,
            fixed_chunk_batches,
            fixed_chunk_seed,
        )
        shuffle = False
    elif shuffle and chunk_shuffle:
        sampler = H5ChunkShuffleSampler(ds, ds.chunk_rows)
        shuffle = False
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=_worker_init,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kw,
    )


def cycle(loader):
    """Endless batch stream over ``loader`` (reshuffles each epoch)."""
    while True:
        for batch in loader:
            yield batch


def to_device(batch, device, non_blocking=True):
    """Move every tensor field of a batch dict to ``device``."""
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}
