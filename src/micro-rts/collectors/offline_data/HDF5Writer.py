"""``HDF5Writer`` — asynchronous, compact, sequence-friendly rollout store.

Design goals (from the collection spec):

- **Asynchronous** — the collector's hot loop is the JVM env step; it must never
  block on gzip/IO. So the writer owns a background thread and an
  :class:`queue.Queue`; the collector only ``enqueue``s per-step numpy batches
  (shape ``[N, ...]``) and returns immediately. The writer thread compresses and
  appends. h5py is not thread-safe, so **only the writer thread touches the file**.

- **Compact** — the raster obs/mask are the expensive part. We exploit that they
  are binary and the action components are small:
    * ``obs``   -> ``uint8`` one-hot planes (bit-exact, 1/4 of float32),
    * ``mask``  -> ``uint8`` 0/1,
    * ``action``-> ``uint8`` (every gridnet component < 256),
  all with gzip. One-hot/binary planes compress ~5-10x, so the store is small
  while staying lossless; the loader casts back to float/long/bool.

- **Fast to load in sequences** — the dynamics world model trains on contiguous
  ``(B, T)`` windows *per trajectory*. Collection is step-major (all ``N`` lanes
  at once), which would scatter a single lane across the file with stride ``N``.
  So a **segment** (a block of ``T`` steps over ``N`` lanes for one map/opponent)
  is buffered in RAM and written **lane-major**: lane ``l``'s whole timeline is a
  single contiguous row-range. Chunking is aligned to that range, so reading a
  window is one contiguous read. A ``traj`` index (start/len + map/opponent id)
  records each lane-block; ``is_first`` still marks episode resets *inside* a
  block (gym auto-reset), so windows never need to cross trajectories.

Datasets (resizable on axis 0 = flat row index, gzip, chunked):
  ``obs``[S,C,H,W] u8, ``action``[S,H*W,7] u8, ``opponent_action``[S,H*W,7] u8,
  ``mask``[S,H*W,MW] u8, ``reward``[S] f32, ``raw_rewards``[S,6] f32,
  ``done``[S] bool, ``is_first``[S] bool.
Trajectory index: ``traj/start``[K] i64, ``traj/length``[K] i64,
  ``traj/map_id``[K] i32, ``traj/opponent_id``[K] i32, ``traj/policy_id``[K] i32,
  ``traj/action_noise``[K] f32.
Legends/meta live in root attrs (see :meth:`_init_file`).

Format v3 (world-model v3 data contract, NEXT_PLAN.md): every step stores BOTH
players' gridnet actions — ``opponent_action`` uses the identical ``(H*W, 7)``
layout — plus per-trajectory provenance of the player-1 controller
(``policy_id`` into the ``policies`` legend, ``action_noise`` = the ε of an
ε-greedy wrapper, 0 otherwise). Joint-action dynamics training cannot fall back
silently: readers hard-error on pre-v3 files.
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any

import numpy as np


# Per-field on-disk dtype. Binary/small-int fields shrink to uint8 losslessly.
_DTYPES = {
    "obs": np.uint8,
    "action": np.uint8,
    "opponent_action": np.uint8,
    "mask": np.uint8,
    "reward": np.float32,
    "raw_rewards": np.float32,
    "done": np.bool_,
    "is_first": np.bool_,
    # Optional (``store_terminal_obs=True``): the true terminal arrival frame of
    # the transition taken at each row — zeros everywhere except done rows, so
    # gzip stores it at ~no cost. Autoresetting envs swallow this frame from the
    # regular stream; without it neither the continue head nor the terminal
    # (win) reward can be supervised (see the dataset's substitution logic).
    "terminal_obs": np.uint8,
    # World-model v2 structured transition fields. int32 preserves sentinels,
    # absolute ticks, and monotonically increasing engine unit IDs.
    "state": np.int32,
    "next_state": np.int32,
    "globals": np.int32,
    "next_globals": np.int32,
    "counterfactual_action": np.uint8,
    "counterfactual_opponent_action": np.uint8,
    "counterfactual_obs": np.uint8,
    "counterfactual_next_state": np.int32,
    "counterfactual_next_globals": np.int32,
    "counterfactual_valid": np.bool_,
}
_STEP_FIELDS = (
    "obs",
    "action",
    "opponent_action",
    "mask",
    "reward",
    "raw_rewards",
    "done",
    "is_first",
)


class HDF5Writer:
    def __init__(
        self,
        path,
        *,
        obs_shape,
        action_shape,
        mask_shape,
        action_nvec,
        grid_hw,
        reward_weight,
        maps,
        opponents,
        policies=("unknown",),
        gzip=4,
        chunk_rows=256,
        config=None,
        git_sha="",
        queue_size=64,
        store_terminal_obs=False,
        store_full_state=False,
        state_shape=None,
        globals_shape=(8,),
        store_counterfactual=False,
        store_counterfactual_obs=False,
        episode_aware=False,
        split="",
        collection_seed=0,
        manifest_hash="",
    ):
        import h5py  # local import: only the writer process/thread needs it

        self._h5py = h5py
        self.path = str(path)
        self.obs_shape = tuple(int(x) for x in obs_shape)
        self.action_shape = tuple(int(x) for x in action_shape)
        self.mask_shape = tuple(int(x) for x in mask_shape)
        self.gzip = gzip
        self.chunk_rows = int(chunk_rows)
        self.store_terminal_obs = bool(store_terminal_obs)
        self.store_full_state = bool(store_full_state)
        self.state_shape = tuple(int(x) for x in state_shape) if state_shape else None
        self.globals_shape = tuple(int(x) for x in globals_shape)
        self.store_counterfactual = bool(store_counterfactual)
        self.store_counterfactual_obs = bool(store_counterfactual_obs)
        self.episode_aware = bool(episode_aware)
        self.split = str(split)
        self.collection_seed = int(collection_seed)
        self.manifest_hash = str(manifest_hash)
        if self.store_counterfactual_obs and not self.store_counterfactual:
            raise ValueError("counterfactual observations require counterfactual storage")
        if self.store_full_state and self.state_shape is None:
            raise ValueError("store_full_state=True requires state_shape")
        structured = (
            ("state", "next_state", "globals", "next_globals")
            if self.store_full_state
            else ()
        )
        counterfactual = (
            (
                "counterfactual_action",
                "counterfactual_opponent_action",
                *(("counterfactual_obs",) if self.store_counterfactual_obs else ()),
                "counterfactual_next_state",
                "counterfactual_next_globals",
                "counterfactual_valid",
            )
            if self.store_counterfactual
            else ()
        )
        if self.store_counterfactual and not self.store_full_state:
            raise ValueError("counterfactual storage requires full state")
        self._fields = (
            _STEP_FIELDS
            + (("terminal_obs",) if store_terminal_obs else ())
            + structured
            + counterfactual
        )

        self._meta = dict(
            format_version=4 if self.store_full_state else 3,
            has_terminal_obs=self.store_terminal_obs,
            has_full_state=self.store_full_state,
            state_schema_version=1 if self.store_full_state else 0,
            state_fields=[
                "terrain",
                "present",
                "unit_id",
                "owner",
                "unit_type",
                "hp",
                "carried",
                "assignment",
                "action_type",
                "direction",
                "target_x",
                "target_y",
                "produced_type",
                "start_tick",
                "eta",
                "remaining",
            ]
            if self.store_full_state
            else [],
            global_fields=[
                "tick",
                "self_resources",
                "opponent_resources",
                "self_reserved",
                "opponent_reserved",
                "reserved_positions",
                "winner",
                "gameover",
            ]
            if self.store_full_state
            else [],
            has_counterfactual=self.store_counterfactual,
            has_counterfactual_obs=self.store_counterfactual_obs,
            action_nvec=list(int(x) for x in action_nvec),
            grid_hw=list(int(x) for x in grid_hw),
            reward_weight=list(float(x) for x in reward_weight),
            maps=list(maps),
            opponents=list(opponents),
            policies=list(policies),
            git_sha=git_sha,
            config_json=json.dumps(config or {}, sort_keys=True),
            episode_aware=self.episode_aware,
            split=self.split,
            collection_seed=self.collection_seed,
            manifest_hash=self.manifest_hash,
        )

        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        # Counters owned by the writer thread.
        self._rows = 0
        self._ntraj = 0
        self._dropped_partial_rows = 0
        self._episode_serial = 0
        self._episode_buffers: list[list[dict[str, np.ndarray]]] | None = None
        # Segment accumulation (writer thread only): list of per-step batches.
        self._seg: list[dict[str, np.ndarray]] = []
        self._thread = threading.Thread(
            target=self._run, name="hdf5-writer", daemon=True
        )
        self._thread.start()

    # --- producer side (collector thread) --------------------------------
    def add_batch(self, batch: dict[str, Any]) -> None:
        """Enqueue one step's data. ``batch[field]`` is array-like ``[N, ...]``."""
        self._raise_if_failed()
        np_batch = {}
        for k in self._fields:
            if k == "terminal_obs" and k not in batch:
                # Sparse field: rows without a terminal arrival store zeros.
                np_batch[k] = np.zeros(
                    (np_batch["obs"].shape[0], *self.obs_shape), _DTYPES[k]
                )
                continue
            v = batch[k]
            arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            np_batch[k] = arr.astype(_DTYPES[k], copy=False)
        self._q.put(("batch", np_batch))

    def end_segment(
        self, *, map_id: int, opponent_id, policy_id=0, action_noise=0.0, seat=0
    ) -> None:
        """Flush the buffered steps as a lane-major set of trajectories.

        ``opponent_id`` is either a scalar (all lanes share an opponent) or a
        per-lane array of length ``N`` (lanes play different bots in one env), so
        one env can cover several bot permutations at once. ``policy_id`` indexes
        the ``policies`` legend (the player-1 controller of this block) and
        ``action_noise`` records the ε of an ε-greedy wrapper — both scalar or
        per-lane, stored per trajectory for provenance filtering.
        """
        self._raise_if_failed()
        opp = np.asarray(opponent_id, dtype=np.int32)
        pol = np.asarray(policy_id, dtype=np.int32)
        eps = np.asarray(action_noise, dtype=np.float32)
        seats = np.asarray(seat, dtype=np.int8)
        self._q.put(("end_segment", (int(map_id), opp, pol, eps, seats)))

    def end_stream(self) -> None:
        """Finish one env/policy stream, dropping only non-terminal episode tails."""
        self._raise_if_failed()
        self._q.put(("end_stream", None))

    def close(self) -> None:
        """Flush the last segment, finalize metadata, join the writer thread."""
        self._q.put(("close", None))
        self._thread.join()
        self._raise_if_failed()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("HDF5 writer thread failed") from self._error

    # --- consumer side (writer thread) -----------------------------------
    def _run(self) -> None:
        try:
            with self._h5py.File(self.path, "w") as f:
                self._init_file(f)
                while True:
                    kind, payload = self._q.get()
                    if kind == "batch":
                        self._seg.append(payload)
                    elif kind == "end_segment":
                        self._write_segment(f, *payload)
                    elif kind == "end_stream":
                        self._drop_partial_episodes()
                    elif kind == "close":
                        if self._seg:
                            # Untagged trailing steps (defensive; collector always
                            # ends segments explicitly). Tag as unknown (-1).
                            n = self._seg[0]["reward"].shape[0]
                            self._write_segment(
                                f,
                                -1,
                                np.full(n, -1, np.int32),
                                np.full(n, -1, np.int32),
                                np.zeros(n, np.float32),
                                np.zeros(n, np.int8),
                            )
                        self._drop_partial_episodes()
                        self._finalize(f)
                        return
        except BaseException as exc:  # surface to the producer
            self._error = exc

    def _init_file(self, f) -> None:
        comp = dict(compression="gzip", compression_opts=self.gzip)
        shapes = {
            "obs": self.obs_shape,
            "action": self.action_shape,
            "opponent_action": self.action_shape,
            "mask": self.mask_shape,
            "reward": (),
            "raw_rewards": (6,),
            "done": (),
            "is_first": (),
            "terminal_obs": self.obs_shape,
            "state": self.state_shape,
            "next_state": self.state_shape,
            "globals": self.globals_shape,
            "next_globals": self.globals_shape,
            "counterfactual_action": self.action_shape,
            "counterfactual_opponent_action": self.action_shape,
            "counterfactual_obs": self.obs_shape,
            "counterfactual_next_state": self.state_shape,
            "counterfactual_next_globals": self.globals_shape,
            "counterfactual_valid": (),
        }
        for name in self._fields:
            tail = shapes[name]
            f.create_dataset(
                name,
                shape=(0, *tail),
                maxshape=(None, *tail),
                dtype=_DTYPES[name],
                chunks=(self.chunk_rows, *tail),
                **comp,
            )
        g = f.create_group("traj")
        for name, dt in (
            ("start", np.int64),
            ("length", np.int64),
            ("map_id", np.int32),
            ("opponent_id", np.int32),
            ("policy_id", np.int32),
            ("action_noise", np.float32),
            ("episode_id", np.int64),
            ("collection_seed", np.int64),
            ("seat", np.int8),
            ("terminal_outcome", np.int8),
        ):
            g.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt)
        for k, v in self._meta.items():
            f.attrs[k] = v

    def _write_segment(
        self, f, map_id: int, opponent_id, policy_id, action_noise, seat
    ) -> None:
        if not self._seg:
            return
        steps = self._seg
        self._seg = []
        T = len(steps)
        N = steps[0]["reward"].shape[0]
        opp = np.broadcast_to(np.asarray(opponent_id, np.int32), (N,)).copy()
        pol = np.broadcast_to(np.asarray(policy_id, np.int32), (N,)).copy()
        eps = np.broadcast_to(np.asarray(action_noise, np.float32), (N,)).copy()
        seats = np.broadcast_to(np.asarray(seat, np.int8), (N,)).copy()

        if self.episode_aware:
            self._write_episode_segment(f, steps, map_id, opp, pol, eps, seats)
            return

        for name in self._fields:
            # (T, N, *tail) -> (N, T, *tail) -> (N*T, *tail): lane-major so each
            # lane's timeline is a contiguous row-range on disk.
            stacked = np.stack([s[name] for s in steps], axis=0)
            tail = stacked.shape[2:]
            lane_major = np.swapaxes(stacked, 0, 1).reshape(N * T, *tail)
            ds = f[name]
            old = ds.shape[0]
            ds.resize(old + N * T, axis=0)
            ds[old:] = lane_major

        base = self._rows
        starts = base + np.arange(N, dtype=np.int64) * T
        self._append_traj(
            f,
            starts,
            np.full(N, T, np.int64),
            np.full(N, map_id, np.int32),
            opp,
            pol,
            eps,
            np.arange(self._ntraj, self._ntraj + N, dtype=np.int64),
            np.full(N, self.collection_seed, np.int64),
            seats,
            np.zeros(N, np.int8),
        )
        self._rows += N * T
        self._ntraj += N

    def _append_traj(
        self,
        f,
        start,
        length,
        map_id,
        opponent_id,
        policy_id,
        action_noise,
        episode_id,
        collection_seed,
        seat,
        terminal_outcome,
    ) -> None:
        g = f["traj"]
        k = g["start"].shape[0]
        n = start.shape[0]
        for name, val in (
            ("start", start),
            ("length", length),
            ("map_id", map_id),
            ("opponent_id", opponent_id),
            ("policy_id", policy_id),
            ("action_noise", action_noise),
            ("episode_id", episode_id),
            ("collection_seed", collection_seed),
            ("seat", seat),
            ("terminal_outcome", terminal_outcome),
        ):
            g[name].resize(k + n, axis=0)
            g[name][k:] = val

    def _write_episode_segment(
        self, f, steps, map_id, opponent_id, policy_id, action_noise, seat
    ) -> None:
        """Accumulate lane streams and append only complete episodes contiguously."""
        T = len(steps)
        N = steps[0]["reward"].shape[0]
        if self._episode_buffers is None:
            self._episode_buffers = [[] for _ in range(N)]
        elif len(self._episode_buffers) != N:
            raise ValueError("num_envs changed before end_stream()")

        for lane in range(N):
            buf = self._episode_buffers[lane]
            for t in range(T):
                row = {name: steps[t][name][lane] for name in self._fields}
                if not buf and not bool(row["is_first"]):
                    # A stream must start at reset. Ignore a defensive partial
                    # prefix rather than admitting an incomplete episode.
                    self._dropped_partial_rows += 1
                    continue
                buf.append(row)
                if bool(row["done"]):
                    self._append_complete_episode(
                        f,
                        buf,
                        map_id=map_id,
                        opponent_id=int(opponent_id[lane]),
                        policy_id=int(policy_id[lane]),
                        action_noise=float(action_noise[lane]),
                        seat=int(seat[lane]),
                    )
                    self._episode_buffers[lane] = []
                    buf = self._episode_buffers[lane]

    def _append_complete_episode(
        self,
        f,
        rows,
        *,
        map_id,
        opponent_id,
        policy_id,
        action_noise,
        seat,
    ) -> None:
        if not rows or not bool(rows[0]["is_first"]) or not bool(rows[-1]["done"]):
            raise ValueError("attempted to append an incomplete episode")
        length = len(rows)
        start = self._rows
        for name in self._fields:
            values = np.stack([row[name] for row in rows], axis=0)
            ds = f[name]
            ds.resize(start + length, axis=0)
            ds[start:] = values

        episode_id = (self.collection_seed << 32) | self._episode_serial
        outcome = int(np.sign(float(rows[-1]["raw_rewards"][0])))
        self._append_traj(
            f,
            np.asarray([start], np.int64),
            np.asarray([length], np.int64),
            np.asarray([map_id], np.int32),
            np.asarray([opponent_id], np.int32),
            np.asarray([policy_id], np.int32),
            np.asarray([action_noise], np.float32),
            np.asarray([episode_id], np.int64),
            np.asarray([self.collection_seed], np.int64),
            np.asarray([seat], np.int8),
            np.asarray([outcome], np.int8),
        )
        self._rows += length
        self._ntraj += 1
        self._episode_serial += 1

    def _drop_partial_episodes(self) -> None:
        if self._episode_buffers is None:
            return
        self._dropped_partial_rows += sum(len(rows) for rows in self._episode_buffers)
        self._episode_buffers = None

    def _finalize(self, f) -> None:
        f.attrs["num_steps"] = int(self._rows)
        f.attrs["num_trajectories"] = int(self._ntraj)
        f.attrs["num_episodes"] = int(self._ntraj if self.episode_aware else 0)
        f.attrs["dropped_partial_rows"] = int(self._dropped_partial_rows)
