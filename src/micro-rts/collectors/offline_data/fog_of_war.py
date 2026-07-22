"""Canonical MicroRTS fog-of-war projection for structured v4 states.

The projection follows ``rts.PartiallyObservableGameState``: a cell is visible
when it lies inside the Euclidean sight-radius disk of any ego-owned unit. There
is no occlusion. The Gym-MicroRTS 0.3.2 raster has no visibility channel, so an
invisible cell is encoded exactly like an empty cell and visibility is returned
separately.
"""

from __future__ import annotations

import numpy as np


# UnitTypeTable insertion order in the canonical MicroRTS engine.
UNIT_TYPE_NAMES = (
    "Resource",
    "Base",
    "Barracks",
    "Worker",
    "Light",
    "Heavy",
    "Ranged",
)
SIGHT_RADII = np.asarray((0, 5, 3, 3, 2, 2, 3), dtype=np.int16)

# gym_microrts 0.3.2 one-hot groups: hp, carried resources, owner, unit type,
# active action type. Value zero is the empty/default category in each group.
OBS_PLANE_SIZES = (5, 5, 3, 8, 6)
OBS_CHANNELS = sum(OBS_PLANE_SIZES)


def empty_observation_cell(dtype=np.uint8) -> np.ndarray:
    """Return the exact 27-channel encoding of an empty Gym-MicroRTS cell."""
    out = np.zeros(OBS_CHANNELS, dtype=dtype)
    out[np.cumsum((0, *OBS_PLANE_SIZES[:-1]))] = 1
    return out


def canonical_visibility(
    state: np.ndarray,
    grid_hw: tuple[int, int],
    *,
    ego_owner: int = 1,
) -> np.ndarray:
    """Compute canonical visibility for ``state[..., H*W, 16]``.

    Returns ``[..., H, W]`` boolean masks. Unit type IDs and owner roles use the
    structured-v2 schema stored in format-v4 corpora.
    """
    state = np.asarray(state)
    h, w = (int(grid_hw[0]), int(grid_hw[1]))
    if state.shape[-2:] != (h * w, 16):
        raise ValueError(f"state tail {state.shape[-2:]} != {(h * w, 16)}")

    lead = state.shape[:-2]
    flat = state.reshape(-1, h, w, 16)
    present = flat[..., 1] == 1
    owner = flat[..., 3]
    unit_type = flat[..., 4]
    ego = present & (owner == int(ego_owner))
    bad = ego & ((unit_type < 0) | (unit_type >= len(SIGHT_RADII)))
    if bad.any():
        values = np.unique(unit_type[bad]).tolist()
        raise ValueError(f"ego units have unknown unit type IDs: {values}")

    visible = np.zeros((flat.shape[0], h, w), dtype=bool)
    for type_id, radius in enumerate(SIGHT_RADII.tolist()):
        batch, ys, xs = np.nonzero(ego & (unit_type == type_id))
        if batch.size == 0:
            continue
        r2 = radius * radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > r2:
                    continue
                yy, xx = ys + dy, xs + dx
                valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
                visible[batch[valid], yy[valid], xx[valid]] = True
    return visible.reshape(*lead, h, w)


def project_ego_observation(
    obs: np.ndarray,
    state: np.ndarray,
    grid_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project full 27-plane observations into ego fog observations.

    Invisible cells become the standard empty-cell categorical encoding. The
    returned visibility has shape ``[..., 1, H, W]`` so it can be concatenated
    with the observation without reshaping.
    """
    obs = np.asarray(obs)
    h, w = (int(grid_hw[0]), int(grid_hw[1]))
    if obs.shape[-3:] != (OBS_CHANNELS, h, w):
        raise ValueError(
            f"obs tail {obs.shape[-3:]} != {(OBS_CHANNELS, h, w)}; "
            "this projector targets gym_microrts 0.3.2's 27-plane encoding"
        )
    if obs.shape[:-3] != state.shape[:-2]:
        raise ValueError(
            f"obs leading shape {obs.shape[:-3]} != state {state.shape[:-2]}"
        )
    visible = canonical_visibility(state, (h, w))
    empty = empty_observation_cell(obs.dtype).reshape(
        *((1,) * len(obs.shape[:-3])), OBS_CHANNELS, 1, 1
    )
    ego_obs = np.where(visible[..., None, :, :], obs, empty)
    return ego_obs, visible[..., None, :, :]

