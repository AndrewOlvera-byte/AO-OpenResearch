"""``H5SequenceDataset`` — fast contiguous-window loader over a collected store.

Mirrors :meth:`SequenceReplayBuffer.sample`: draws ``batch`` sequences of length
``seq_len`` as a TensorDict ``[B, T, ...]`` ready for the tokenizer / dynamics
losses. The on-disk layout (see :class:`HDF5Writer`) stores each lane's timeline
as one contiguous row-range (a *trajectory*), so a window is a single contiguous
slice read — no strided gather. Decoding casts the compact on-disk dtypes back to
what the models expect:

    obs   u8 -> float32   (one-hot planes),
    action u8 -> long     (gridnet components),
    mask   u8 -> bool,
    reward f32,  done u8/bool -> cont = 1 - done (float),  is_first bool.

Trajectories shorter than ``seq_len`` are skipped when sampling. ``map_id`` /
``opponent_id`` filters let a training run pick a subset of the collected
map/opponent permutations.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from tensordict import TensorDict


class H5SequenceDataset:
    MIN_FORMAT = 3  # v3: opponent_action + policy provenance (NEXT_PLAN.md)

    def __init__(self, path, *, device="cpu", map_ids=None, opponent_ids=None,
                 policy_ids=None):
        import h5py

        self.path = str(path)
        self.device = device
        self._f = h5py.File(self.path, "r")
        self.attrs = dict(self._f.attrs)
        fmt = int(self.attrs.get("format_version", 0))
        if fmt < self.MIN_FORMAT:
            raise ValueError(
                f"{self.path}: format_version={fmt} < {self.MIN_FORMAT} — pre-v3 "
                "stores lack opponent_action; joint-action dynamics training on "
                "them would silently see zero opponent actions. Recollect with "
                "collect_mrts_data.py on the patched jar.")
        self.maps = list(self.attrs.get("maps", []))
        self.opponents = list(self.attrs.get("opponents", []))
        self.policies = list(self.attrs.get("policies", []))
        self.config = json.loads(self.attrs.get("config_json", "{}"))

        g = self._f["traj"]
        self._start = g["start"][:]
        self._length = g["length"][:]
        self._map_id = g["map_id"][:]
        self._opp_id = g["opponent_id"][:]
        self._policy_id = g["policy_id"][:]
        self._action_noise = g["action_noise"][:]
        keep = np.ones(len(self._start), dtype=bool)
        if map_ids is not None:
            keep &= np.isin(self._map_id, np.asarray(list(map_ids)))
        if opponent_ids is not None:
            keep &= np.isin(self._opp_id, np.asarray(list(opponent_ids)))
        if policy_ids is not None:
            keep &= np.isin(self._policy_id, np.asarray(list(policy_ids)))
        self._keep = np.nonzero(keep)[0]

    # --- introspection ---------------------------------------------------
    @property
    def num_trajectories(self) -> int:
        return len(self._keep)

    @property
    def num_steps(self) -> int:
        return int(self.attrs.get("num_steps", self._f["reward"].shape[0]))

    def can_sample(self, seq_len: int) -> bool:
        return bool(np.any(self._length[self._keep] >= seq_len))

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- sampling --------------------------------------------------------
    def sample(self, batch: int, seq_len: int) -> TensorDict:
        """Draw ``batch`` random ``[T]`` windows -> TensorDict ``[B, T, ...]``."""
        eligible = self._keep[self._length[self._keep] >= seq_len]
        if len(eligible) == 0:
            raise ValueError(f"no trajectory has length >= {seq_len}")
        pick = eligible[np.random.randint(0, len(eligible), size=batch)]
        rows = np.empty((batch, seq_len), dtype=np.int64)
        for i, tj in enumerate(pick):
            s, L = int(self._start[tj]), int(self._length[tj])
            off = np.random.randint(0, L - seq_len + 1)
            rows[i] = s + off + np.arange(seq_len)
        return self._gather(rows)

    def _read(self, name, rows) -> np.ndarray:
        # h5py wants sorted unique indices for fast fancy-indexing; a window is
        # contiguous, so read the flat span [min,max] once and reshape.
        flat = rows.reshape(-1)
        lo, hi = int(flat.min()), int(flat.max())
        block = self._f[name][lo:hi + 1]
        return block[flat - lo].reshape(*rows.shape, *block.shape[1:])

    def _gather(self, rows) -> TensorDict:
        B, T = rows.shape
        obs = torch.from_numpy(self._read("obs", rows)).float()
        action = torch.from_numpy(self._read("action", rows).astype(np.int64))
        opp_action = torch.from_numpy(
            self._read("opponent_action", rows).astype(np.int64))
        mask = torch.from_numpy(self._read("mask", rows)).bool()
        reward = torch.from_numpy(self._read("reward", rows).astype(np.float32))
        raw = torch.from_numpy(self._read("raw_rewards", rows).astype(np.float32))
        done = torch.from_numpy(self._read("done", rows)).bool()
        is_first = torch.from_numpy(self._read("is_first", rows)).bool()
        td = TensorDict(
            {
                "obs": obs,
                "action": action,
                "opponent_action": opp_action,
                "mask": mask,
                "reward": reward,
                "raw_rewards": raw,
                "cont": (~done).float(),
                "is_first": is_first,
            },
            batch_size=[B, T],
        )
        return td.to(self.device)
