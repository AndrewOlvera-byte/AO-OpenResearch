import importlib.util
from pathlib import Path

import h5py
import numpy as np

from collectors.offline_data.fog_of_war import (
    OBS_CHANNELS,
    OBS_PLANE_SIZES,
    canonical_visibility,
    empty_observation_cell,
    project_ego_observation,
)


_SCRIPT = Path(__file__).parents[1] / "entrypoints" / "util" / "augment_fog_observations.py"
_SPEC = importlib.util.spec_from_file_location("augment_fog_observations", _SCRIPT)
augment_fog = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(augment_fog)


def _empty_state(batch=1, h=16, w=16):
    state = np.zeros((batch, h * w, 16), np.int32)
    state[..., 2:] = -1
    state[..., 0] = 0
    state[..., 1] = 0
    state[..., 6] = 0
    state[..., 7] = 0
    state[..., 14:] = 0
    return state


def _put(state, x, y, *, owner, unit_type):
    i = y * 16 + x
    state[:, i, 1] = 1
    state[:, i, 2] = i
    state[:, i, 3] = owner
    state[:, i, 4] = unit_type


def _empty_obs(batch=1, h=16, w=16):
    cell = empty_observation_cell()
    return np.broadcast_to(cell[:, None, None], (batch, OBS_CHANNELS, h, w)).copy()


def test_visibility_uses_euclidean_radius_and_clips_boundaries():
    state = _empty_state()
    _put(state, 0, 0, owner=1, unit_type=1)  # Base, radius 5.
    vis = canonical_visibility(state, (16, 16))[0]
    assert vis[0, 0] and vis[0, 5] and vis[4, 3]
    assert not vis[4, 4]  # sqrt(32) > 5
    assert int(vis.sum()) == sum(
        x * x + y * y <= 25 for y in range(16) for x in range(16)
    )


def test_visibility_has_no_occlusion_and_uses_unit_specific_radius():
    state = _empty_state()
    _put(state, 8, 8, owner=1, unit_type=4)  # Light, radius 2.
    _put(state, 8, 7, owner=0, unit_type=0)  # Resource does not occlude.
    vis = canonical_visibility(state, (16, 16))[0]
    assert vis[6, 8] and vis[8, 10]
    assert not vis[5, 8] and not vis[6, 10]


def test_projection_preserves_visible_cells_and_empty_encodes_fog():
    state = _empty_state()
    _put(state, 1, 1, owner=1, unit_type=3)  # Worker, radius 3.
    _put(state, 2, 1, owner=2, unit_type=4)  # visible enemy
    _put(state, 15, 15, owner=2, unit_type=5)  # hidden enemy
    obs = _empty_obs()
    obs[0, :, 1, 2] = 0
    obs[0, 1, 1, 2] = 1
    obs[0, 5, 1, 2] = 1
    obs[0, 12, 1, 2] = 1
    obs[0, 18, 1, 2] = 1
    obs[0, 21, 1, 2] = 1
    obs[0, :, 15, 15] = obs[0, :, 1, 2]

    ego, visibility = project_ego_observation(obs, state, (16, 16))
    assert np.array_equal(ego[0, :, 1, 2], obs[0, :, 1, 2])
    assert np.array_equal(ego[0, :, 15, 15], empty_observation_cell())
    assert visibility.shape == (1, 1, 16, 16)
    assert visibility[0, 0, 1, 2] and not visibility[0, 0, 15, 15]
    offset = 0
    for size in OBS_PLANE_SIZES:
        assert np.all(ego[:, offset : offset + size].sum(1) == 1)
        offset += size


def _make_source(path):
    rows, h, w = 3, 16, 16
    state = _empty_state(rows, h, w)
    for row in range(rows):
        _put(state[row : row + 1], row + 1, 1, owner=1, unit_type=3)
        _put(state[row : row + 1], 15, 15, owner=2, unit_type=4)
    obs = _empty_obs(rows, h, w)
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = 4
        f.attrs["grid_hw"] = (h, w)
        f.create_dataset("obs", data=obs, chunks=(1, 27, h, w), compression="gzip")
        f.create_dataset("state", data=state, chunks=(1, h * w, 16))
        f.create_dataset("globals", data=np.zeros((rows, 8), np.int32))
        f.create_dataset("next_state", data=state)
        f.create_dataset("next_globals", data=np.zeros((rows, 8), np.int32))
        f.create_dataset("sentinel", data=np.arange(rows, dtype=np.int64))


def test_augmentation_copies_preserves_resumes_and_fully_audits(tmp_path):
    source = tmp_path / "source.h5"
    output = tmp_path / "output.h5"
    _make_source(source)
    augment_fog.augment(source, output, block_rows=2, gzip=1)
    stats = augment_fog.audit(output, block_rows=2)
    assert stats["rows"] == 3
    with h5py.File(source, "r") as src, h5py.File(output, "r") as out:
        assert np.array_equal(out["sentinel"][:], src["sentinel"][:])
        assert np.array_equal(out["obs"][:], src["obs"][:])
        assert bool(out.attrs["fog_augmentation_complete"])
        assert int(out.attrs["fog_rows_written"]) == 3
    # A completed run is idempotent and keeps passing the exhaustive audit.
    augment_fog.augment(source, output, block_rows=1, gzip=1)
    augment_fog.audit(output, block_rows=1)

