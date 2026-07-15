"""Offline data collection tests (HDF5 writer / dataset / policy / collector).

The data-plane logic (compact lossless storage, lane-major layout, contiguous
sequence sampling, is_first tagging, per-lane opponent tags) is exercised with a
tiny in-memory ``FakeVecEnv`` so no JVM is needed. One JVM-backed smoke test
(``test_microrts_collect_smoke``) validates the real MicroRTS obs/action/mask
actually fit the compact dtypes and round-trip.
"""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from collectors.offline_data import (
    EpsilonGreedyPolicy,
    H5SequenceDataset,
    HDF5Writer,
    MaskedRandomPolicy,
    OfflineCollector,
)
from environments.dream_env import DreamEnv

OBS = (6, 8, 8)
GRID = 8 * 8
NVEC = [GRID, 6, 4, 4, 4, 4, 7, 49]
N_COMP = 7
MASK_W = 1 + sum(NVEC[1:])


class FakeVecEnv:
    """obs encodes the global step counter; done every ``period`` steps.

    obs plane 0 carries the step index so sampled sequences can be checked for
    contiguity; masks are all-legal so the masked-random policy has choices.
    """

    def __init__(self, num_envs=4, period=4):
        self.num_envs = num_envs
        self.period = period
        self.obs_shape = OBS
        self.action_nvec = torch.as_tensor(NVEC)
        self.mask_shape = (GRID, MASK_W)
        self.gridnet = True
        self._t = 0
        self._pending = None

    def _pack(self, done):
        n = self.num_envs
        obs = torch.zeros(n, *OBS)
        obs[:, 0] = float(self._t)                      # step index in plane 0, cell 0
        return TensorDict({
            "obs": obs,
            "mask": torch.ones(n, GRID, MASK_W),
            "reward": torch.full((n,), 0.5),
            "raw_rewards": torch.arange(n * 6, dtype=torch.float32).reshape(n, 6),
            "done": torch.as_tensor(done),
            # Patched-jar shape/tick contract: the scripted opponent's gridnet
            # action for the step that produced this frame (marker = step index).
            "opponent_action": torch.full((n, GRID, N_COMP), self._t % 5,
                                          dtype=torch.long),
        }, batch_size=[n])

    def async_reset(self, seed=None):
        self._t = 0
        self._pending = self._pack([False] * self.num_envs)

    def send(self, actions):
        self._t += 1
        done = [(self._t % self.period == 0)] * self.num_envs
        self._pending = self._pack(done)

    def recv(self):
        out, self._pending = self._pending, None
        return out

    def reset(self):
        self.async_reset()
        return self.recv()

    def step(self, actions):
        self.send(actions)
        return self.recv()

    def close(self):
        pass


class TerminalFakeVecEnv(FakeVecEnv):
    """FakeVecEnv surfacing the terminal arrival frame like the patched jar:
    a ``terminal_obs`` key on every transition, zeros except done lanes."""

    def _pack(self, done):
        td = super()._pack(done)
        term = torch.zeros(self.num_envs, *OBS)
        for i, d in enumerate(done):
            if d:
                term[i, 0] = 200.0                      # arrival-frame marker
        td.set("terminal_obs", term)
        return td


def _writer(path, **kw):
    return HDF5Writer(
        path, obs_shape=OBS, action_shape=(GRID, N_COMP), mask_shape=(GRID, MASK_W),
        action_nvec=NVEC, grid_hw=OBS[1:], reward_weight=[1.0] * 6,
        maps=["mapA", "mapB"], opponents=["botX", "botY"],
        policies=["masked_random"], gzip=4, **kw,
    )


def _run_collection(path, num_envs=4, steps=12, seg=5, opponent_id=0, period=4):
    env = DreamEnv(FakeVecEnv(num_envs=num_envs, period=period))
    policy = MaskedRandomPolicy(torch.as_tensor(NVEC))
    writer = _writer(path, chunk_rows=8)
    coll = OfflineCollector(env, policy, writer, steps_per_segment=seg)
    coll.collect(steps, map_id=0, opponent_id=opponent_id)
    writer.close()
    return env, policy


# --- masked random policy -------------------------------------------------
def test_masked_random_policy_respects_mask():
    pol = MaskedRandomPolicy(torch.as_tensor(NVEC))
    n = 3
    mask = torch.zeros(n, GRID, MASK_W)
    # Cell 0: component 0 (size 6) legal only at index 2 -> the sampled action's
    # type component must be forced to 2 (a real legality check). Its other
    # components have all-zero masks: like the canonical GridNet path, those
    # degenerate rows sample uniformly and are ignored by the engine (identical to
    # online collection, so the offline data matches the training distribution).
    mask[:, 0, 0] = 1.0            # source-selectable flag
    mask[:, 0, 1 + 2] = 1.0        # component 0 -> only choice index 2 legal
    out = pol.step(torch.zeros(n, *OBS), mask)
    assert out["action"].shape == (n, GRID, N_COMP)
    assert torch.all(out["action"][:, 0, 0] == 2)          # forced legal choice
    # All components stay within their nvec ranges everywhere.
    cell_nvec = torch.as_tensor(NVEC[1:])
    assert torch.all(out["action"] < cell_nvec) and torch.all(out["action"] >= 0)


def test_epsilon_greedy_mixes_legal_random():
    """ε=0 is the base policy verbatim; ε=1 replaces every cell with a masked
    random legal action; intermediate ε swaps roughly that fraction of cells."""

    class ConstPolicy:
        def step(self, obs, mask=None, deterministic=False):
            n = obs.shape[0]
            a = torch.full((n, GRID, N_COMP), 3, dtype=torch.long)
            return TensorDict({"action": a}, batch_size=[n])

    torch.manual_seed(0)
    nvec = torch.as_tensor(NVEC)
    obs = torch.zeros(2, *OBS)
    # Legal mask: only choice 1 of component 0 is legal everywhere.
    mask = torch.zeros(2, GRID, MASK_W)
    mask[:, :, 0] = 1.0
    mask[:, :, 1 + 1] = 1.0

    base = ConstPolicy()
    assert (EpsilonGreedyPolicy(base, nvec, 0.0).step(obs, mask)["action"] == 3).all()
    full = EpsilonGreedyPolicy(base, nvec, 1.0).step(obs, mask)["action"]
    assert (full[..., 0] == 1).all(), "ε=1 must sample the only LEGAL type choice"
    part = EpsilonGreedyPolicy(base, nvec, 0.3).step(obs, mask)["action"]
    frac = (part[..., 0] != 3).float().mean()
    assert 0.15 < frac < 0.45, f"swap fraction {frac} far from ε=0.3"


# --- writer basic contract ------------------------------------------------
def test_writer_shapes_dtypes_and_metadata(tmp_path):
    import h5py

    path = tmp_path / "d.h5"
    _run_collection(path, num_envs=4, steps=12, seg=5)
    with h5py.File(path, "r") as f:
        assert f["obs"].dtype == np.uint8
        assert f["action"].dtype == np.uint8
        assert f["opponent_action"].dtype == np.uint8
        assert f["opponent_action"].shape[1:] == (GRID, N_COMP)
        assert f["mask"].dtype == np.uint8
        assert f["reward"].dtype == np.float32
        assert f["raw_rewards"].shape[1:] == (6,)
        assert f["obs"].shape[1:] == OBS
        assert f["action"].shape[1:] == (GRID, N_COMP)
        assert f["mask"].shape[1:] == (GRID, MASK_W)
        # 12 steps x 4 lanes = 48 transitions.
        assert f["obs"].shape[0] == 48
        assert f.attrs["num_steps"] == 48
        assert list(f.attrs["opponents"]) == ["botX", "botY"]
        assert list(f.attrs["policies"]) == ["masked_random"]
        assert f.attrs["format_version"] == 3
        # terminal_obs is opt-in; a plain collection stores none.
        assert not f.attrs["has_terminal_obs"]
        assert "terminal_obs" not in f
        # segments of 5,5,2 steps -> 3 flushes x 4 lanes = 12 trajectories.
        assert f["traj"]["start"].shape[0] == 12


def test_collector_stores_terminal_obs_at_done_rows(tmp_path):
    """Positive path of the collector's terminal_obs feature-detect: an env that
    surfaces the arrival frame gets it stored sparsely, exactly at done rows."""
    import h5py

    path = tmp_path / "d.h5"
    env = DreamEnv(TerminalFakeVecEnv(num_envs=2, period=4))
    policy = MaskedRandomPolicy(torch.as_tensor(NVEC))
    has_term = "terminal_obs" in env.reset().keys()     # the CLI's feature-detect
    assert has_term
    writer = _writer(path, chunk_rows=8, store_terminal_obs=has_term)
    coll = OfflineCollector(env, policy, writer, steps_per_segment=12)
    coll.collect(12, map_id=0, opponent_id=0)
    writer.close()

    with h5py.File(path, "r") as f:
        assert f.attrs["has_terminal_obs"]
        done = f["done"][:].astype(bool)
        term = f["terminal_obs"][:]
        assert done.sum() == 6                          # dones at t=4,8,12 x 2 lanes
        nz = term.reshape(len(term), -1).any(axis=1)
        assert (nz == done).all()
        assert (term[done][:, 0] == 200).all()          # the arrival marker survived


def test_writer_lane_major_contiguity(tmp_path):
    """Each trajectory (lane block) must hold one lane's consecutive timeline."""
    import h5py

    path = tmp_path / "d.h5"
    _run_collection(path, num_envs=3, steps=9, seg=9, period=100)  # one segment, no resets
    with h5py.File(path, "r") as f:
        starts, lengths = f["traj"]["start"][:], f["traj"]["length"][:]
        assert len(starts) == 3 and set(lengths) == {9}
        for s, L in zip(starts, lengths):
            # obs plane-0/cell-0 stored the step index -> must be 0..L-1 in order.
            steps = f["obs"][s:s + L, 0, 0, 0].astype(np.int64)
            assert steps.tolist() == list(range(L))


# --- dataset round-trip / sampling ---------------------------------------
def test_dataset_sampling_shapes_and_contiguity(tmp_path):
    path = tmp_path / "d.h5"
    _run_collection(path, num_envs=4, steps=16, seg=16, period=100)
    with H5SequenceDataset(path) as ds:
        assert ds.can_sample(4)
        batch = ds.sample(batch=6, seq_len=4)
        assert batch["obs"].shape == (6, 4, *OBS)
        assert batch["action"].shape == (6, 4, GRID, N_COMP)
        assert batch["mask"].shape == (6, 4, GRID, MASK_W)
        assert batch["obs"].dtype == torch.float32
        assert batch["action"].dtype == torch.int64
        assert batch["mask"].dtype == torch.bool
        assert batch["cont"].dtype == torch.float32
        # contiguity: obs plane-0/cell-0 step index increases by 1 within a window.
        idx = batch["obs"][..., 0, 0, 0]
        assert torch.all(idx[:, 1:] - idx[:, :-1] == 1)


def test_dataset_decode_is_lossless(tmp_path):
    """uint8 storage must reconstruct obs/mask/action bit-exactly."""
    path = tmp_path / "d.h5"
    _run_collection(path, num_envs=2, steps=6, seg=6, period=100)
    with H5SequenceDataset(path) as ds:
        b = ds.sample(batch=4, seq_len=3)
        # obs plane-0/cell-0 holds integer step indices; they must survive round-trip.
        assert b["obs"][..., 0, 0, 0].round().eq(b["obs"][..., 0, 0, 0]).all()
        assert set(np.unique(b["mask"].numpy()).tolist()) <= {0.0, 1.0, True, False}


def test_opponent_ids_are_per_lane(tmp_path):
    """Cycling bots across lanes must tag each trajectory with its lane's bot."""
    import h5py

    path = tmp_path / "d.h5"
    env = DreamEnv(FakeVecEnv(num_envs=4, period=100))
    writer = _writer(path, chunk_rows=8)
    coll = OfflineCollector(env, MaskedRandomPolicy(torch.as_tensor(NVEC)), writer,
                            steps_per_segment=8)
    lane_opp = np.array([0, 1, 0, 1], dtype=np.int32)      # botX, botY, botX, botY
    coll.collect(8, map_id=0, opponent_id=lane_opp)
    writer.close()
    with h5py.File(path, "r") as f:
        assert f["traj"]["opponent_id"][:].tolist() == [0, 1, 0, 1]
        assert set(f["traj"]["map_id"][:].tolist()) == {0}


def test_opponent_action_roundtrip_and_alignment(tmp_path):
    """The stored opponent_action must be the post-step (same-tick) one: at row
    t the FakeVecEnv marks it with the step index t+1 (the step that produced
    the next frame), matching the submitted-action alignment."""
    import h5py

    path = tmp_path / "d.h5"
    _run_collection(path, num_envs=2, steps=4, seg=4, period=100)
    with h5py.File(path, "r") as f:
        opp = f["opponent_action"][:]
        assert opp.shape[1:] == (GRID, N_COMP)
        starts, lengths = f["traj"]["start"][:], f["traj"]["length"][:]
        for s, L in zip(starts, lengths):
            markers = opp[s:s + L, 0, 0].astype(int).tolist()
            assert markers == [(t + 1) % 5 for t in range(L)]


def test_provenance_policy_id_and_noise(tmp_path):
    """policy_id / action_noise land per trajectory for corpus slicing."""
    import h5py

    path = tmp_path / "d.h5"
    env = DreamEnv(FakeVecEnv(num_envs=2, period=100))
    writer = _writer(path, chunk_rows=8)
    coll = OfflineCollector(env, MaskedRandomPolicy(torch.as_tensor(NVEC)), writer,
                            steps_per_segment=4)
    coll.collect(4, map_id=0, opponent_id=0, policy_id=2, action_noise=0.15)
    coll.collect(4, map_id=0, opponent_id=1, policy_id=0, action_noise=0.0)
    writer.close()
    with h5py.File(path, "r") as f:
        assert f["traj"]["policy_id"][:].tolist() == [2, 2, 0, 0]
        assert np.allclose(f["traj"]["action_noise"][:], [0.15, 0.15, 0.0, 0.0])
    # policy filter must slice the corpus.
    with H5SequenceDataset(path, policy_ids=[2]) as ds:
        assert ds.num_trajectories == 2


def test_selfplay_pairs_records_partner_action(tmp_path):
    """selfplay_pairs=True: lane i's opponent_action is lane i^1's own action."""
    import h5py

    class SPFakeVecEnv(FakeVecEnv):
        def _pack(self, done):
            td = super()._pack(done)
            del td["opponent_action"]        # selfplay env has no scripted bot
            return td

    path = tmp_path / "d.h5"
    env = DreamEnv(SPFakeVecEnv(num_envs=4, period=100))

    class CountingPolicy:
        """Deterministic per-lane marker actions so pairing is checkable."""

        def step(self, obs, mask=None, deterministic=False):
            n = obs.shape[0]
            a = torch.arange(n, dtype=torch.long).view(n, 1, 1) \
                .expand(n, GRID, N_COMP).clone()
            return TensorDict({"action": a}, batch_size=[n])

    writer = _writer(path, chunk_rows=8)
    coll = OfflineCollector(env, CountingPolicy(), writer,
                            steps_per_segment=4, selfplay_pairs=True)
    coll.collect(4, map_id=0, opponent_id=-1, policy_id=1)
    writer.close()
    with h5py.File(path, "r") as f:
        starts = f["traj"]["start"][:]
        act = f["action"][:]
        opp = f["opponent_action"][:]
        # Lane-major blocks in lane order 0..3: action marker == lane, opponent
        # marker == partner lane (0<->1, 2<->3).
        for lane, s in enumerate(starts):
            assert (act[s] == lane).all()
            assert (opp[s] == (lane ^ 1)).all()


def test_collector_requires_opponent_action_in_bot_mode(tmp_path):
    """A bot-mode env without the patched jar's field must fail loudly."""

    class NoOppFakeVecEnv(FakeVecEnv):
        def _pack(self, done):
            td = super()._pack(done)
            del td["opponent_action"]
            return td

    writer = _writer(tmp_path / "d.h5")
    with pytest.raises(RuntimeError, match="opponent_action"):
        OfflineCollector(DreamEnv(NoOppFakeVecEnv(num_envs=2)),
                         MaskedRandomPolicy(torch.as_tensor(NVEC)), writer)
    writer.close()


def test_is_first_marks_episode_starts(tmp_path):
    """DreamEnv is_first must land in the store where the env auto-resets."""
    import h5py

    path = tmp_path / "d.h5"
    _run_collection(path, num_envs=1, steps=8, seg=8, period=4)
    with h5py.File(path, "r") as f:
        is_first = f["is_first"][:].astype(bool)
        # step 0 is a reset; done fires after steps 4 and 8 -> is_first at the
        # obs following them (indices 0 and 4 in the stored, 0-indexed timeline).
        assert is_first[0] and is_first[4]
        assert is_first.sum() == 2


def test_writer_surfaces_thread_errors(tmp_path):
    """A bad batch must fail loudly (not silently drop) on close/next add."""
    path = tmp_path / "d.h5"
    writer = _writer(path)
    good = {k: v for k, v in _good_batch().items()}
    writer.add_batch(good)
    writer.end_segment(map_id=0, opponent_id=0)
    # Wrong obs shape on a second segment -> writer thread raises on flush.
    bad = _good_batch()
    bad["obs"] = torch.zeros(4, 3, 3, 3)                    # mismatched shape
    writer.add_batch(bad)
    with pytest.raises(RuntimeError):
        writer.end_segment(map_id=0, opponent_id=0)
        writer.close()


def test_inspect_dataset_tool_runs(tmp_path):
    """The inspect/validate CLI runs its loader + invariant checks end to end."""
    from collectors.offline_data import inspect_dataset

    path = tmp_path / "d.h5"
    _run_collection(path, num_envs=4, steps=16, seg=16, period=100)
    # FakeVecEnv obs stores step indices (not one-hot), so skip the real-data
    # invariants; this guards the tool's structure/loader path against bitrot.
    rc = inspect_dataset.main(
        [str(path), "--skip-invariants", "--skip-model",
         "--batch", "3", "--seq-len", "4", "--iters", "2"])
    assert rc == 0


def _good_batch(n=4):
    return {
        "obs": torch.zeros(n, *OBS),
        "action": torch.zeros(n, GRID, N_COMP, dtype=torch.long),
        "opponent_action": torch.zeros(n, GRID, N_COMP, dtype=torch.long),
        "mask": torch.ones(n, GRID, MASK_W),
        "reward": torch.zeros(n),
        "raw_rewards": torch.zeros(n, 6),
        "done": torch.zeros(n, dtype=torch.bool),
        "is_first": torch.zeros(n, dtype=torch.bool),
    }


# --- JVM smoke test -------------------------------------------------------
def test_microrts_collect_smoke(tmp_path):
    """Real MicroRTS obs/action/mask must fit the compact dtypes and round-trip;
    the patched jar's terminal frames must reach the store at done rows."""
    import h5py

    from environments.microrts_env import EnvConfig, MicroRTSVecEnv

    cfg = EnvConfig(num_envs=2, max_steps=64, mode="bot",
                    bots=("randomBiasedAI", "workerRushAI"), gridnet=True,
                    opponent_action=True, full_state=True)
    base = MicroRTSVecEnv(cfg)
    env = DreamEnv(base)
    # The patched microrts.jar (infra/microrts-jar-patch) is a build-time part of
    # the container image; the wrapper must detect and surface its terminal frames.
    assert base._has_terminal_obs, "microrts.jar lacks terminalObservation — run apply_patch.sh"
    n_comp = len(base.action_nvec) - 1
    path = tmp_path / "mrts.h5"
    writer = HDF5Writer(
        path, obs_shape=base.obs_shape, action_shape=(base._grid_cells, n_comp),
        mask_shape=base.mask_shape, action_nvec=base.action_nvec.tolist(),
        grid_hw=base.obs_shape[1:], reward_weight=cfg.reward_weight,
        maps=[cfg.map_path], opponents=list(cfg.bots),
        policies=["masked_random"], gzip=1, chunk_rows=8,
        store_terminal_obs=True,
        store_full_state=True, state_shape=(base._grid_cells, 16),
        store_counterfactual=True,
    )
    policy = MaskedRandomPolicy(base.action_nvec)
    coll = OfflineCollector(env, policy, writer, steps_per_segment=5,
                            counterfactual_frac=0.25)
    # 70 > max_steps=64 so every lane crosses one episode boundary (truncation
    # counts: the vec client raises done and swaps in the reset frame either way).
    coll.collect(70, map_id=0, opponent_id=np.array([0, 1], dtype=np.int32))
    writer.close()
    env.close()

    with H5SequenceDataset(path) as ds:
        assert ds.num_steps == 140                         # 70 steps x 2 lanes
        b = ds.sample(batch=3, seq_len=4)
        assert b["obs"].shape[1:] == (4, *base.obs_shape)
        # every recorded action component must be legal under its stored mask.
        assert b["mask"].any()

    with h5py.File(path, "r") as f:
        # The scripted bots really act: their recorded channel is non-degenerate.
        assert (f["opponent_action"][:] != 0).any(), \
            "opponent_action all zero — jar patch not surfacing bot actions"
        assert f.attrs["format_version"] == 4 and f.attrs["has_full_state"]
        assert f["state"].shape[1:] == (base._grid_cells, 16)
        assert f["counterfactual_valid"][:].any()
        assert (f["counterfactual_action"][:] != f["action"][:]).any()

    with h5py.File(path, "r") as f:
        done = f["done"][:].astype(bool)
        assert done.any(), "no episode boundary despite steps > max_steps"
        term = f["terminal_obs"][:]
        nz = term.reshape(len(term), -1).any(axis=1)
        assert (nz == done).all()
        # A real terminal frame, not the next episode's reset frame (the old
        # bug). The reset frame is the NEXT row (is_first) of the lane block —
        # obs at the done row itself is the departure frame, which a quiet
        # truncation tick may legitimately leave unchanged.
        is_first = f["is_first"][:].astype(bool)
        nxt = np.flatnonzero(done) + 1
        nxt = nxt[nxt < len(done)]
        nxt = nxt[is_first[nxt]]
        assert nxt.size > 0
        assert (term[nxt - 1] != f["obs"][nxt]).reshape(nxt.size, -1).any(axis=1).all()
        # Structured arrivals use the same pre-reset terminal state and advance
        # one engine tick rather than jumping to the next episode's tick zero.
        done_idx = np.flatnonzero(done)
        assert (f["next_globals"][done_idx, 0] == f["globals"][done_idx, 0] + 1).all()
