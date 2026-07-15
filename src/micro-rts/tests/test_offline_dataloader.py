"""Offline DataLoader tests — the torch input pipeline for DreamerV4 pretraining.

Exercises :class:`MRTSSequenceDataset` and :func:`build_mrts_loader` against a
synthetic HDF5 store fabricated directly with :class:`HDF5Writer` (no JVM). The
store marks each step's global index in ``obs`` plane-0 cell (0,0) so window
contiguity and no-trajectory-crossing can be asserted exactly. A final
model-readiness test runs both pretraining losses (``tokenizer_loss`` /
``dynamics_loss``) on a sampled batch.
"""

import numpy as np
import pytest
import torch

import models.dreamer as D
from collectors.offline_data import (
    MRTSSequenceDataset,
    build_mrts_loader,
    cycle,
    to_device,
)
from collectors.offline_data import HDF5Writer
from loss.dreamer import dynamics_loss, tokenizer_loss

OBS = (6, 8, 8)
GRID = 8 * 8
NVEC = [GRID, 6, 4, 4, 4, 4, 7, 49]
N_COMP = 7
MASK_W = 1 + sum(NVEC[1:])          # 79
CELL_NVEC = np.asarray(NVEC[1:])


def _make_h5(path, *, num_lanes=4, seg_len=16, segments=2, opponent_id=0):
    """Fabricate a store: ``segments`` blocks of ``seg_len`` steps over ``num_lanes``
    lanes -> ``segments*num_lanes`` trajectories of length ``seg_len``. obs plane-0
    cell (0,0) carries the global step index (contiguity marker)."""
    writer = HDF5Writer(
        path, obs_shape=OBS, action_shape=(GRID, N_COMP), mask_shape=(GRID, MASK_W),
        action_nvec=NVEC, grid_hw=OBS[1:], reward_weight=[1.0] * 6,
        maps=["mapA"], opponents=["botX", "botY"], gzip=1, chunk_rows=8)
    gstep = 0
    for _ in range(segments):
        for t in range(seg_len):
            n = num_lanes
            obs = np.zeros((n, *OBS), np.float32)
            obs[:, 0, 0, 0] = float(gstep)                 # step-index marker
            action = np.random.randint(0, CELL_NVEC, size=(n, GRID, N_COMP)).astype(np.int64)
            opp_action = np.random.randint(0, CELL_NVEC, size=(n, GRID, N_COMP)).astype(np.int64)
            is_first = np.zeros((n,), bool)
            if t == 0:
                is_first[:] = True                          # trajectory (segment) start
            writer.add_batch({
                "obs": obs,
                "action": action,
                "opponent_action": opp_action,
                "mask": np.ones((n, GRID, MASK_W), np.float32),
                "reward": np.full((n,), 0.5, np.float32),
                "raw_rewards": np.zeros((n, 6), np.float32),
                "done": np.zeros((n,), bool),
                "is_first": is_first,
            })
            gstep += 1
        writer.end_segment(map_id=0, opponent_id=opponent_id)
    writer.close()


# --- dataset: windows + shapes + task fields ------------------------------
def test_window_count_and_tokenizer_fields(tmp_path):
    path = tmp_path / "d.h5"
    _make_h5(path, num_lanes=4, seg_len=16, segments=2)   # 8 trajectories, len 16
    ds = MRTSSequenceDataset(path, seq_len=4, task="tokenizer")
    # each trajectory (len 16) -> 16-4+1 = 13 windows; 8 trajectories -> 104.
    assert len(ds) == 8 * (16 - 4 + 1)
    assert ds.obs_shape == OBS and ds.action_nvec == NVEC
    item = ds[0]
    assert set(item) == {"obs", "mask"}                    # tokenizer task fields only
    assert item["obs"].shape == (4, *OBS) and item["obs"].dtype == torch.float32
    assert item["mask"].shape == (4, GRID, MASK_W) and item["mask"].dtype == torch.bool
    ds.close()


def test_dynamics_task_fields_and_dtypes(tmp_path):
    path = tmp_path / "d.h5"
    _make_h5(path, num_lanes=2, seg_len=12, segments=1)
    ds = MRTSSequenceDataset(path, seq_len=5, task="dynamics")
    item = ds[0]
    assert set(item) == {"obs", "action", "opponent_action", "reward",
                         "cont", "is_first"}                # no mask
    assert item["action"].shape == (5, GRID, N_COMP) and item["action"].dtype == torch.int64
    assert item["opponent_action"].shape == (5, GRID, N_COMP)
    assert item["opponent_action"].dtype == torch.int64
    assert (item["opponent_action"] < torch.as_tensor(CELL_NVEC)).all()
    assert item["reward"].shape == (5,) and item["reward"].dtype == torch.float32
    assert item["cont"].dtype == torch.float32 and torch.all((item["cont"] == 1.0))
    assert item["is_first"].dtype == torch.bool
    assert (item["action"] < torch.as_tensor(CELL_NVEC)).all()
    ds.close()


def test_stride_reduces_windows(tmp_path):
    path = tmp_path / "d.h5"
    _make_h5(path, num_lanes=1, seg_len=16, segments=1)   # one trajectory, len 16
    dense = MRTSSequenceDataset(path, seq_len=4, task="tokenizer", stride=1)
    strided = MRTSSequenceDataset(path, seq_len=4, task="tokenizer", stride=4)
    assert len(dense) == 13
    assert len(strided) == len(range(0, 16 - 4 + 1, 4))   # offsets 0,4,8 -> 3
    dense.close(); strided.close()


# --- contiguity + no trajectory crossing ----------------------------------
def test_windows_are_contiguous_and_never_cross_trajectories(tmp_path):
    path = tmp_path / "d.h5"
    _make_h5(path, num_lanes=3, seg_len=8, segments=2)    # markers 0..7 then 8..15 per lane
    ds = MRTSSequenceDataset(path, seq_len=4, task="dynamics")
    for i in range(len(ds)):
        idx = ds[i]["obs"][..., 0, 0, 0]                   # step-index marker over T
        # strictly consecutive within a window (never jumps a trajectory boundary).
        assert torch.all(idx[1:] - idx[:-1] == 1)
        # is_first can only fire on the first slot of a window (a trajectory start).
        assert not bool(ds[i]["is_first"][1:].any())
    ds.close()


# --- terminal-frame substitution -------------------------------------------
def _make_h5_with_terminal(path, *, seg_len=12, done_at=5):
    """One lane, one segment; episode ends at ``done_at`` (autoreset: the row
    after it is the reset frame flagged is_first) and the true terminal arrival
    frame is stored sparsely at the done row. Markers: obs plane-0 (0,0) carries
    the step index; the terminal obs carries 200."""
    writer = HDF5Writer(
        path, obs_shape=OBS, action_shape=(GRID, N_COMP), mask_shape=(GRID, MASK_W),
        action_nvec=NVEC, grid_hw=OBS[1:], reward_weight=[1.0] * 6,
        maps=["mapA"], opponents=["botX"], gzip=1, chunk_rows=8,
        store_terminal_obs=True)
    for t in range(seg_len):
        obs = np.zeros((1, *OBS), np.float32)
        obs[:, 0, 0, 0] = float(t)
        batch = {
            "obs": obs,
            "action": np.zeros((1, GRID, N_COMP), np.int64),
            "opponent_action": np.zeros((1, GRID, N_COMP), np.int64),
            "mask": np.ones((1, GRID, MASK_W), np.float32),
            "reward": np.full((1,), 10.0 if t == done_at else 0.5, np.float32),
            "raw_rewards": np.zeros((1, 6), np.float32),
            "done": np.array([t == done_at], bool),
            "is_first": np.array([t == 0 or t == done_at + 1], bool),
        }
        if t == done_at:
            term = np.zeros((1, *OBS), np.float32)
            term[:, 0, 0, 0] = 200.0  # obs is uint8 on disk
            batch["terminal_obs"] = term
        writer.add_batch(batch)
    writer.end_segment(map_id=0, opponent_id=0)
    writer.close()


def test_terminal_frame_substitution(tmp_path):
    path = tmp_path / "term.h5"
    done_at = 5
    _make_h5_with_terminal(path, seg_len=12, done_at=done_at)
    ds = MRTSSequenceDataset(path, seq_len=12, task="all")
    assert ds.has_terminal_obs
    item = ds[0]
    reset_slot = done_at + 1                        # raw store: reset frame here
    # The terminal arrival replaces the reset frame...
    assert float(item["obs"][reset_slot, 0, 0, 0]) == 200.0
    # ...with an all-invalid action mask (game over)...
    assert not bool(item["mask"][reset_slot].any())
    # ...and it is a REAL supervised arrival: not an episode start, while the
    # next slot inherits the boundary.
    assert not bool(item["is_first"][reset_slot])
    assert bool(item["is_first"][reset_slot + 1])
    # Its transition targets are the terminal ones: cont=0 and the win reward,
    # both read arrive-aligned from the departure row.
    assert float(item["cont"][done_at]) == 0.0
    assert float(item["reward"][done_at]) == 10.0
    # Frames after the boundary are untouched.
    assert float(item["obs"][reset_slot + 1, 0, 0, 0]) == float(reset_slot + 1)
    ds.close()


def test_store_without_terminal_field_unchanged(tmp_path):
    path = tmp_path / "d.h5"
    _make_h5(path, num_lanes=1, seg_len=8, segments=1)
    ds = MRTSSequenceDataset(path, seq_len=8, task="all")
    assert not ds.has_terminal_obs
    item = ds[0]
    assert bool(item["mask"].all())                 # no zeroing applied
    assert float(item["obs"][3, 0, 0, 0]) == 3.0
    ds.close()


# --- held-out split ---------------------------------------------------------
def test_val_split_is_disjoint_deterministic_and_complete(tmp_path):
    path = tmp_path / "d.h5"
    _make_h5(path, num_lanes=4, seg_len=8, segments=3)    # 12 trajectories
    kw = dict(seq_len=4, task="tokenizer", val_frac=0.25, split_seed=7)
    train = MRTSSequenceDataset(path, split="train", **kw)
    val = MRTSSequenceDataset(path, split="val", **kw)
    full = MRTSSequenceDataset(path, seq_len=4, task="tokenizer")
    tr_traj = set(np.unique(train.traj_idx))
    va_traj = set(np.unique(val.traj_idx))
    assert tr_traj.isdisjoint(va_traj)                    # no leakage
    assert tr_traj | va_traj == set(np.unique(full.traj_idx))
    assert len(va_traj) == 3                              # ceil(0.25 * 12)
    # Deterministic: rebuilding with the same seed reproduces the split.
    val2 = MRTSSequenceDataset(path, split="val", **kw)
    assert set(np.unique(val2.traj_idx)) == va_traj
    for ds in (train, val, full, val2):
        ds.close()


def test_lr_scheduler_warms_up_and_decays():
    from entrypoints.pretrain_common import make_lr_scheduler

    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = make_lr_scheduler(opt, total_steps=100, warmup_steps=10, min_frac=0.1)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < 0.2                                   # warmup starts low
    assert abs(lrs[10] - 1.0) < 0.01                      # peaks at base LR
    assert lrs[50] < lrs[10]                              # decays
    assert abs(opt.param_groups[0]["lr"] - 0.1) < 0.01    # ends at min_frac


# --- filtering ------------------------------------------------------------
def test_opponent_filter_selects_subset(tmp_path):
    path = tmp_path / "d.h5"
    writer = HDF5Writer(
        path, obs_shape=OBS, action_shape=(GRID, N_COMP), mask_shape=(GRID, MASK_W),
        action_nvec=NVEC, grid_hw=OBS[1:], reward_weight=[1.0] * 6,
        maps=["mapA"], opponents=["botX", "botY"], gzip=1, chunk_rows=8)
    # 4 lanes, per-lane opponents [0,1,0,1] in one segment of 8 steps.
    for t in range(8):
        writer.add_batch({
            "obs": np.zeros((4, *OBS), np.float32),
            "action": np.zeros((4, GRID, N_COMP), np.int64),
            "opponent_action": np.zeros((4, GRID, N_COMP), np.int64),
            "mask": np.ones((4, GRID, MASK_W), np.float32),
            "reward": np.zeros((4,), np.float32),
            "raw_rewards": np.zeros((4, 6), np.float32),
            "done": np.zeros((4,), bool),
            "is_first": np.zeros((4,), bool),
        })
    writer.end_segment(map_id=0, opponent_id=np.array([0, 1, 0, 1], np.int32))
    writer.close()

    both = MRTSSequenceDataset(path, seq_len=4, task="tokenizer")
    only0 = MRTSSequenceDataset(path, seq_len=4, task="tokenizer", opponent_ids=[0])
    assert both.num_trajectories == 4 and only0.num_trajectories == 2
    assert len(only0) == 2 * (8 - 4 + 1)
    both.close(); only0.close()


def test_short_trajectories_are_skipped(tmp_path):
    path = tmp_path / "d.h5"
    _make_h5(path, num_lanes=2, seg_len=4, segments=1)     # trajectories of length 4
    assert len(MRTSSequenceDataset(path, seq_len=8, task="tokenizer")) == 0
    with pytest.raises(ValueError):
        build_mrts_loader(path, task="tokenizer", seq_len=8, batch_size=2, num_workers=0)


# --- loader: batching (single- and multi-worker) --------------------------
def test_loader_batches_shapes_single_worker(tmp_path):
    path = tmp_path / "d.h5"
    _make_h5(path, num_lanes=4, seg_len=16, segments=1)
    loader = build_mrts_loader(path, task="dynamics", seq_len=6, batch_size=5,
                               num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    assert batch["obs"].shape == (5, 6, *OBS)
    assert batch["action"].shape == (5, 6, GRID, N_COMP)
    assert batch["reward"].shape == (5, 6)
    assert batch["cont"].shape == (5, 6)
    assert batch["is_first"].shape == (5, 6)
    # contiguity survives collation.
    idx = batch["obs"][..., 0, 0, 0]
    assert torch.all(idx[:, 1:] - idx[:, :-1] == 1)


def test_loader_multiworker_yields_full_batches(tmp_path):
    path = tmp_path / "d.h5"
    _make_h5(path, num_lanes=4, seg_len=16, segments=2)
    loader = build_mrts_loader(path, task="tokenizer", seq_len=4, batch_size=8,
                               num_workers=2, pin_memory=False,
                               persistent_workers=False)
    seen = 0
    for batch in loader:
        assert batch["obs"].shape == (8, 4, *OBS)
        assert batch["mask"].shape == (8, 4, GRID, MASK_W)
        seen += 1
    assert seen > 0


def test_fixed_chunk_validation_is_repeatable_and_batched(tmp_path):
    path = tmp_path / "d.h5"
    _make_h5(path, num_lanes=4, seg_len=16, segments=3)

    def sample():
        loader = build_mrts_loader(
            path,
            task="tokenizer",
            seq_len=1,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            fixed_chunk_batches=3,
            fixed_chunk_seed=17,
        )
        return [batch["obs"][:, 0, 0, 0, 0].clone() for batch in loader]

    first, second = sample(), sample()
    assert len(first) == len(second) == 3
    for a, b in zip(first, second):
        assert a.shape == (4,)
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_cycle_and_to_device_are_endless(tmp_path):
    path = tmp_path / "d.h5"
    _make_h5(path, num_lanes=2, seg_len=8, segments=1)
    loader = build_mrts_loader(path, task="tokenizer", seq_len=4, batch_size=3,
                               num_workers=0, pin_memory=False, drop_last=True)
    stream = cycle(loader)
    for _ in range(10):                                     # more than one epoch's batches
        batch = to_device(next(stream), "cpu")
        assert batch["obs"].shape[0] == 3


# --- model readiness: both pretraining losses run on a batch --------------
def _tiny_model():
    cfg = D.DreamerV4Config.from_dict({
        "tokenizer": {"d_latent": 8, "enc_channels": 16},
        "dynamics": {"d_model": 32, "depth": 2, "n_heads": 4, "n_register": 2,
                     "k_max": 4, "action_emb": 8, "action_channels": 16},
    })
    return D.DreamerV4(OBS, NVEC, cfg)


def test_tokenizer_loss_on_loader_batch(tmp_path):
    path = tmp_path / "d.h5"
    _make_h5(path, num_lanes=4, seg_len=16, segments=1)
    loader = build_mrts_loader(path, task="tokenizer", seq_len=4, batch_size=3,
                               num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    model = _tiny_model()
    loss, metrics, z = tokenizer_loss(model, batch["obs"], batch["mask"])
    assert torch.isfinite(loss) and z.shape[:2] == (3, 4)
    loss.backward()
    assert model.tokenizer.to_latent.weight.grad is not None


def test_dynamics_loss_on_loader_batch(tmp_path):
    path = tmp_path / "d.h5"
    _make_h5(path, num_lanes=4, seg_len=16, segments=1)
    loader = build_mrts_loader(path, task="dynamics", seq_len=6, batch_size=3,
                               num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    model = _tiny_model()
    model.tokenizer.requires_grad_(False)
    with torch.no_grad():
        z = model.tokenizer.encode(batch["obs"])
    loss, metrics = dynamics_loss(model, z, batch["action"], batch["reward"],
                                  batch["cont"], batch["is_first"])
    assert torch.isfinite(loss)
    for key in ("wm/reward", "wm/continue", "flow/total", "wm/total"):
        assert key in metrics and np.isfinite(metrics[key])
    loss.backward()
    # dynamics phase trains the world model, not the frozen tokenizer.
    assert model.world_model.flow_x_head.weight.grad is not None
    assert model.tokenizer.to_latent.weight.grad is None
