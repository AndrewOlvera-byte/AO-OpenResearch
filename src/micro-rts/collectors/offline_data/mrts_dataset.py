"""``MRTSSequenceDataset`` — a map-style torch ``Dataset`` over a collected store.

The offline HDF5 store (see :class:`HDF5Writer`) lays out each lane's timeline as a
single contiguous row-range (a *trajectory*). :class:`H5SequenceDataset` draws a
whole random batch in one call — great for a replay-style RL loop, but it can't be
handed to a multi-worker ``DataLoader``. For the two **offline pretraining** runs
(tokenizer, then dynamics with a frozen tokenizer) we want the standard PyTorch
input pipeline: worker processes prefetching, pinning, and shuffling contiguous
windows while the GPU trains.

So this is a plain map-style ``Dataset``:

- **one window per item** — every contiguous ``seq_len`` window that fits inside a
  trajectory is enumerated up front (``stride`` controls overlap). Windows never
  cross a trajectory boundary, so a window is one contiguous slice read and never
  mixes two lanes/episodes. ``__getitem__`` returns a dict of ``[T, ...]`` tensors;
  the default collate stacks them into ``[B, T, ...]``.
- **per-worker H5 handle** — h5py handles are not fork-safe, so the file is opened
  lazily on first access *inside each worker* (the loader's ``worker_init_fn``
  resets the handle after fork). The lightweight trajectory index / attrs are read
  once in ``__init__`` and the handle is closed again, so nothing open is inherited.
- **task toggle** — decoding only what a phase needs keeps the (expensive) IO down:
    * ``"tokenizer"`` -> ``{obs, mask}``            (reconstruction + mask-BCE),
    * ``"dynamics"``  -> ``{obs, action, reward, cont, is_first}`` (frozen-tokenizer
      world-model training; skips the heavy ``H*W x mask_width`` mask planes),
    * ``"all"``       -> every field (inspection / validation).

Decoding casts the compact on-disk dtypes back to what the models expect: obs
``u8 -> float32`` (one-hot planes), action ``u8 -> long``, mask ``u8 -> bool``,
reward ``f32``, ``cont = 1 - done`` (float), ``is_first`` bool.
"""

from __future__ import annotations

import json

import numpy as np
import torch

# Fields to decode for each training phase. ``obs`` is always needed (the tokenizer
# encodes it); everything else is phase-specific so a run never pays IO for planes
# it will not use (the mask alone is H*W x mask_width uint8 — the store's biggest
# per-step field after obs).
_TASK_FIELDS = {
    "tokenizer": ("obs", "mask"),
    "dynamics": ("obs", "action", "opponent_action", "reward", "done", "is_first"),
    "all": (
        "obs",
        "action",
        "opponent_action",
        "mask",
        "reward",
        "raw_rewards",
        "done",
        "is_first",
    ),
    "structured_tokenizer": ("state", "globals", "obs", "mask"),
    # structured_flow_loss currently consumes exactly these six fields. Loading
    # counterfactual arrivals and bookkeeping here more than doubles decompression.
    "structured_dynamics": (
        "state",
        "globals",
        "next_state",
        "next_globals",
        "action",
        "opponent_action",
    ),
    # Paired causal training intentionally pays the extra IO: cloned-engine
    # arrivals are part of the objective, not merely an evaluation probe.
    "structured_dynamics_paired": (
        "state",
        "globals",
        "next_state",
        "next_globals",
        "action",
        "opponent_action",
        "counterfactual_action",
        "counterfactual_opponent_action",
        "counterfactual_next_state",
        "counterfactual_next_globals",
        "counterfactual_valid",
    ),
    # Dual forward/inverse action-tokenizer SSL uses the same factual and
    # cloned-branch transition contract as paired dynamics, but has its own task
    # name so entrypoint IO requirements remain explicit and testable.
    "structured_action_tokenizer": (
        "state",
        "globals",
        "next_state",
        "next_globals",
        "action",
        "opponent_action",
        "counterfactual_action",
        "counterfactual_opponent_action",
        "counterfactual_next_state",
        "counterfactual_next_globals",
        "counterfactual_valid",
    ),
    "structured_dynamics_eval": (
        "state",
        "globals",
        "next_state",
        "next_globals",
        "action",
        "opponent_action",
        "counterfactual_action",
        "counterfactual_opponent_action",
        "counterfactual_next_state",
        "counterfactual_next_globals",
        "counterfactual_valid",
    ),
}


class MRTSSequenceDataset(torch.utils.data.Dataset):
    MIN_FORMAT = 3  # v3: opponent_action + policy provenance (NEXT_PLAN.md)

    def __init__(
        self,
        path,
        *,
        seq_len,
        task="dynamics",
        stride=1,
        map_ids=None,
        opponent_ids=None,
        policy_ids=None,
        locking=True,
        val_frac=0.0,
        split="train",
        split_seed=0,
        h5_cache_mb=64,
    ):
        import h5py

        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        if task not in _TASK_FIELDS:
            raise ValueError(
                f"task must be one of {sorted(_TASK_FIELDS)}, got {task!r}"
            )
        self.path = str(path)
        self.seq_len = int(seq_len)
        self.task = task
        self.stride = max(1, int(stride))
        self.fields = _TASK_FIELDS[task]
        self._locking = locking
        self._h5_cache_bytes = int(h5_cache_mb) * 1024 * 1024
        self._f = None  # lazy, per-worker (see the module docstring)

        # Read the (small) index + metadata once, then close: nothing stays open to
        # be inherited across the DataLoader's fork.
        with h5py.File(self.path, "r", locking=locking) as f:
            self.attrs = dict(f.attrs)
            fmt = int(self.attrs.get("format_version", 0))
            if fmt < self.MIN_FORMAT:
                raise ValueError(
                    f"{self.path}: format_version={fmt} < {self.MIN_FORMAT} — "
                    "pre-v3 stores lack opponent_action; recollect with "
                    "collect_mrts_data.py on the patched jar."
                )
            if task.startswith("structured_") and fmt < 4:
                raise ValueError(
                    f"{self.path}: task={task!r} requires HDF5 format v4 complete "
                    "state; recollect with collect_mrts_data.py --full-state"
                )
            g = f["traj"]
            start = g["start"][:]
            length = g["length"][:]
            map_id = g["map_id"][:]
            opp_id = g["opponent_id"][:]
            policy_id = g["policy_id"][:]
            self.obs_channels = int(f["obs"].shape[1])
            chunk_source = f["state"] if task.startswith("structured_") else f["obs"]
            self.chunk_rows = int(chunk_source.chunks[0]) if chunk_source.chunks else 1
            self.state_shape = tuple(f["state"].shape[1:]) if "state" in f else None
            self.globals_shape = (
                tuple(f["globals"].shape[1:]) if "globals" in f else None
            )
            # Counterfactual fields are optional within v4. Keep the standard
            # transition/evaluation tasks compatible with corpora collected at
            # fraction 0, but fail early when paired training explicitly needs
            # those fields.
            derived = {"cont", "is_first"}
            requested_fields = self.fields
            self.fields = tuple(
                name for name in self.fields if name in f or name in derived
            )
            if task in (
                "structured_dynamics_paired",
                "structured_action_tokenizer",
            ):
                missing = sorted(set(requested_fields) - set(self.fields))
                if missing:
                    raise ValueError(
                        f"{self.path}: task={task!r} requires a corpus collected "
                        "with --counterfactual-frac > 0; missing fields "
                        f"{missing}"
                    )

        # Stores collected with the terminal-frame patch carry the true terminal
        # arrival obs sparsely at done rows; __getitem__ splices them in.
        self.has_terminal_obs = bool(self.attrs.get("has_terminal_obs", False))
        self.maps = list(self.attrs.get("maps", []))
        self.opponents = list(self.attrs.get("opponents", []))
        self.policies = list(self.attrs.get("policies", []))
        self.action_nvec = [int(x) for x in self.attrs.get("action_nvec", [])]
        self.grid_hw = tuple(int(x) for x in self.attrs.get("grid_hw", ()))
        self.config = json.loads(self.attrs.get("config_json", "{}"))

        keep = np.ones(len(start), dtype=bool)
        if map_ids is not None:
            keep &= np.isin(map_id, np.asarray(list(map_ids)))
        if opponent_ids is not None:
            keep &= np.isin(opp_id, np.asarray(list(opponent_ids)))
        if policy_ids is not None:
            keep &= np.isin(policy_id, np.asarray(list(policy_ids)))

        # Trajectory-level held-out split: a seeded permutation of the kept
        # trajectories, first ceil(val_frac*N) go to "val". Same (seed, frac)
        # from two processes yields complementary, non-overlapping splits.
        if val_frac > 0.0:
            kept_idx = np.nonzero(keep)[0]
            perm = np.random.default_rng(split_seed).permutation(len(kept_idx))
            n_val = int(np.ceil(val_frac * len(kept_idx)))
            chosen = perm[:n_val] if split == "val" else perm[n_val:]
            drop = np.ones(len(start), dtype=bool)
            drop[kept_idx[chosen]] = False
            keep &= ~drop

        # Enumerate every window start (absolute flat row index). A window stays
        # inside one trajectory, so ``[start, start + seq_len)`` is contiguous and
        # single-lane. ``traj_idx`` records which trajectory each window came from
        # (handy for grouping / debugging).
        starts, traj_of = [], []
        for tj in np.nonzero(keep)[0]:
            s, L = int(start[tj]), int(length[tj])
            if L < self.seq_len:
                continue
            offs = np.arange(0, L - self.seq_len + 1, self.stride, dtype=np.int64)
            starts.append(s + offs)
            traj_of.append(np.full(len(offs), tj, dtype=np.int64))
        self._win_start = np.concatenate(starts) if starts else np.empty(0, np.int64)
        self.traj_idx = np.concatenate(traj_of) if traj_of else np.empty(0, np.int64)

    # --- introspection ---------------------------------------------------
    @property
    def obs_shape(self):
        return (self.obs_channels, *self.grid_hw)

    @property
    def num_trajectories(self):
        return len(np.unique(self.traj_idx))

    def __len__(self):
        return int(self._win_start.shape[0])

    # --- data access -----------------------------------------------------
    def _file(self):
        if self._f is None:
            import h5py

            self._f = h5py.File(
                self.path,
                "r",
                locking=self._locking,
                rdcc_nbytes=self._h5_cache_bytes,
                rdcc_nslots=10007,
            )
        return self._f

    def __getitem__(self, idx):
        f = self._file()
        s = int(self._win_start[idx])
        e = s + self.seq_len

        obs = f["obs"][s:e].astype(np.float32) if "obs" in self.fields else None
        mask = f["mask"][s:e] if "mask" in self.fields else None
        needs_reset_alignment = self.has_terminal_obs and (
            obs is not None or mask is not None
        )
        is_first = (
            f["is_first"][s:e].astype(bool)
            if "is_first" in self.fields or needs_reset_alignment
            else None
        )

        # Terminal-frame substitution (stores with ``has_terminal_obs``): a reset
        # row at slot t hides the terminal arrival of the t-1 transition. Splice
        # the stored terminal obs in at slot t — it becomes a REAL arrival (its
        # action-shift already carries the terminal action, and cont/reward
        # targets at slot t are the terminal transition's) — and push is_first to
        # t+1, where the reset's unknown action is replaced by the learned
        # embedding as usual. The reset frame itself drops out of the window.
        if needs_reset_alignment:
            sub = np.nonzero(is_first[1:])[0] + 1
            if sub.size:
                term = f["terminal_obs"][s:e]
                if obs is not None:
                    obs[sub] = term[sub - 1].astype(np.float32)
                if mask is not None:
                    mask = mask.copy()
                    mask[sub] = 0  # game over: nothing is legal
                is_first = is_first.copy()
                is_first[sub] = False
                nxt = sub + 1
                is_first[nxt[nxt < is_first.shape[0]]] = True

        out = {}
        if obs is not None:
            out["obs"] = torch.from_numpy(obs)
        if "action" in self.fields:
            out["action"] = torch.from_numpy(f["action"][s:e].astype(np.int64))
        if "opponent_action" in self.fields:
            out["opponent_action"] = torch.from_numpy(
                f["opponent_action"][s:e].astype(np.int64)
            )
        if mask is not None:
            out["mask"] = torch.from_numpy(mask).bool()
        if "reward" in self.fields:
            out["reward"] = torch.from_numpy(f["reward"][s:e].astype(np.float32))
        if "raw_rewards" in self.fields:
            out["raw_rewards"] = torch.from_numpy(
                f["raw_rewards"][s:e].astype(np.float32)
            )
        if "done" in self.fields:
            # The world model consumes continue (1 - done), never done directly.
            out["cont"] = torch.from_numpy(
                (~f["done"][s:e].astype(bool)).astype(np.float32)
            )
        if "is_first" in self.fields:
            out["is_first"] = torch.from_numpy(is_first)
        for name in ("state", "next_state", "globals", "next_globals"):
            if name in self.fields:
                out[name] = torch.from_numpy(f[name][s:e].astype(np.int64))
        for name in ("counterfactual_action", "counterfactual_opponent_action"):
            if name in self.fields:
                out[name] = torch.from_numpy(f[name][s:e].astype(np.int64))
        for name in ("counterfactual_next_state", "counterfactual_next_globals"):
            if name in self.fields:
                out[name] = torch.from_numpy(f[name][s:e].astype(np.int64))
        if "counterfactual_valid" in self.fields:
            out["counterfactual_valid"] = torch.from_numpy(
                f["counterfactual_valid"][s:e].astype(bool)
            )
        return out

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
