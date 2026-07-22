"""Copy and augment a format-v4 MicroRTS HDF5 corpus with canonical fog.

The source is never modified. The output preserves every source dataset and
attribute and adds only:

``ego_obs``
    ``[S, 27, H, W] uint8``; the existing observation with invisible cells
    replaced by the valid Gym-MicroRTS empty-cell encoding.
``ego_visibility``
    ``[S, 1, H, W] bool``; distinguishes fogged cells from observed-empty cells.

The augmentation is resumable. ``--audit-only`` performs a full streamed,
row-for-row recomputation and comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    _here = Path(__file__).resolve()
    sys.path.insert(0, str(_here.parents[1]))  # src/micro-rts
    sys.path.insert(0, str(_here.parents[2]))  # src

import h5py
import numpy as np

from collectors.offline_data.fog_of_war import (
    OBS_CHANNELS,
    OBS_PLANE_SIZES,
    SIGHT_RADII,
    UNIT_TYPE_NAMES,
    empty_observation_cell,
    project_ego_observation,
)


FOG_VERSION = 1
ENGINE_COMMIT = "72f2fd94039adf3323f4591947e98661a572f927"
NEW_DATASETS = ("ego_obs", "ego_visibility")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", nargs="?", help="immutable source format-v4 .h5")
    p.add_argument("output", help="augmented output .h5")
    p.add_argument("--block-rows", type=int, default=2048)
    p.add_argument("--gzip", type=int, default=1)
    p.add_argument("--audit-only", action="store_true")
    return p.parse_args(argv)


def _validate_base(f: h5py.File) -> tuple[int, int, int]:
    required = ("obs", "state", "globals", "next_state", "next_globals")
    missing = [name for name in required if name not in f]
    if missing:
        raise ValueError(f"missing required format-v4 datasets: {missing}")
    if int(f.attrs.get("format_version", 0)) < 4:
        raise ValueError("fog augmentation requires format_version >= 4")
    h, w = (int(x) for x in f.attrs["grid_hw"])
    rows = int(f["state"].shape[0])
    if f["state"].shape != (rows, h * w, 16):
        raise ValueError(f"unexpected state shape {f['state'].shape}")
    if f["obs"].shape != (rows, OBS_CHANNELS, h, w):
        raise ValueError(f"unexpected obs shape {f['obs'].shape}")
    return rows, h, w


def _copy_source(source: Path, output: Path) -> None:
    if output.exists():
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".copying")
    if temporary.exists():
        raise RuntimeError(
            f"incomplete prior copy exists: {temporary}; remove it and retry"
        )
    print(f"copying immutable source: {source} -> {output}", flush=True)
    shutil.copy2(source, temporary)
    os.replace(temporary, output)


def _init_augmentation(f: h5py.File, source: Path, gzip: int) -> int:
    rows, h, w = _validate_base(f)
    present = [name for name in NEW_DATASETS if name in f]
    if present and len(present) != len(NEW_DATASETS):
        raise ValueError(f"partial fog schema present: {present}")
    if not present:
        obs_chunks = f["obs"].chunks or (min(256, rows), OBS_CHANNELS, h, w)
        chunk_rows = int(obs_chunks[0])
        f.create_dataset(
            "ego_obs",
            shape=(rows, OBS_CHANNELS, h, w),
            dtype=np.uint8,
            chunks=(chunk_rows, OBS_CHANNELS, h, w),
            compression="gzip",
            compression_opts=int(gzip),
        )
        f.create_dataset(
            "ego_visibility",
            shape=(rows, 1, h, w),
            dtype=np.bool_,
            chunks=(chunk_rows, 1, h, w),
            compression="gzip",
            compression_opts=int(gzip),
        )
        f.attrs["has_ego_obs"] = True
        f.attrs["fog_of_war_version"] = FOG_VERSION
        f.attrs["fog_source_path"] = str(source.resolve())
        f.attrs["fog_engine_commit"] = ENGINE_COMMIT
        f.attrs["fog_observation_plane_sizes"] = OBS_PLANE_SIZES
        f.attrs["fog_unit_type_names_json"] = json.dumps(UNIT_TYPE_NAMES)
        f.attrs["fog_sight_radii"] = SIGHT_RADII
        f.attrs["fog_projection"] = (
            "canonical MicroRTS Euclidean sight disks; no occlusion; "
            "invisible cells use the Gym-MicroRTS empty-cell encoding"
        )
        f.attrs["fog_rows_written"] = 0
        f.attrs["fog_augmentation_complete"] = False
        f.flush()
    else:
        if f["ego_obs"].shape != (rows, OBS_CHANNELS, h, w):
            raise ValueError(f"unexpected ego_obs shape {f['ego_obs'].shape}")
        if f["ego_visibility"].shape != (rows, 1, h, w):
            raise ValueError(
                f"unexpected ego_visibility shape {f['ego_visibility'].shape}"
            )
        if int(f.attrs.get("fog_of_war_version", -1)) != FOG_VERSION:
            raise ValueError("existing fog augmentation uses a different version")
    return int(f.attrs.get("fog_rows_written", 0))


def augment(source: Path, output: Path, *, block_rows: int, gzip: int) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("source and output must differ; the source is immutable")
    _copy_source(source, output)
    started = time.monotonic()
    with h5py.File(output, "r+") as f:
        start = _init_augmentation(f, source, gzip)
        rows, h, w = _validate_base(f)
        if bool(f.attrs.get("fog_augmentation_complete", False)):
            print("augmentation already complete", flush=True)
            return
        if not 0 <= start <= rows:
            raise ValueError(f"invalid fog_rows_written={start} for {rows} rows")
        print(f"augmenting rows {start:,}..{rows:,} in blocks of {block_rows:,}")
        for begin in range(start, rows, block_rows):
            end = min(rows, begin + block_rows)
            obs = f["obs"][begin:end]
            state = f["state"][begin:end]
            ego_obs, visibility = project_ego_observation(obs, state, (h, w))
            f["ego_obs"][begin:end] = ego_obs
            f["ego_visibility"][begin:end] = visibility
            f.attrs.modify("fog_rows_written", end)
            if end == rows or begin == start or (begin // block_rows) % 32 == 0:
                f.flush()
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = (end - start) / elapsed
                eta = (rows - end) / rate if rate else float("inf")
                print(
                    f"  {end:,}/{rows:,} ({100 * end / rows:.2f}%) "
                    f"{rate:,.0f} rows/s ETA {eta / 60:.1f} min",
                    flush=True,
                )
        f.attrs.modify("fog_augmentation_complete", True)
        f.attrs["fog_completed_unix_time"] = int(time.time())
        f.flush()


def audit(path: Path, *, block_rows: int) -> dict[str, int | float]:
    """Fully recompute and compare every derived row; raise on any mismatch."""
    totals = {
        "rows": 0,
        "visible_cells": 0,
        "hidden_cells": 0,
        "visible_opponent_units": 0,
        "hidden_opponent_units": 0,
        "visible_neutral_units": 0,
        "hidden_neutral_units": 0,
        "frames_without_visible_opponent": 0,
    }
    empty = empty_observation_cell()
    with h5py.File(path, "r") as f:
        rows, h, w = _validate_base(f)
        for name in NEW_DATASETS:
            if name not in f:
                raise ValueError(f"missing augmented dataset {name!r}")
        if not bool(f.attrs.get("has_ego_obs", False)):
            raise ValueError("has_ego_obs is not true")
        if not bool(f.attrs.get("fog_augmentation_complete", False)):
            raise ValueError("fog_augmentation_complete is not true")
        if int(f.attrs.get("fog_rows_written", -1)) != rows:
            raise ValueError("fog_rows_written does not equal source row count")
        if f["ego_obs"].dtype != np.uint8:
            raise ValueError(f"ego_obs dtype is {f['ego_obs'].dtype}, not uint8")
        if f["ego_visibility"].dtype != np.bool_:
            raise ValueError(
                f"ego_visibility dtype is {f['ego_visibility'].dtype}, not bool"
            )

        for begin in range(0, rows, block_rows):
            end = min(rows, begin + block_rows)
            obs = f["obs"][begin:end]
            state = f["state"][begin:end]
            expected_obs, expected_vis = project_ego_observation(obs, state, (h, w))
            actual_obs = f["ego_obs"][begin:end]
            actual_vis = f["ego_visibility"][begin:end]
            if not np.array_equal(actual_vis, expected_vis):
                bad = np.argwhere(actual_vis != expected_vis)[0]
                raise AssertionError(
                    f"ego_visibility mismatch at row {begin + int(bad[0])}, "
                    f"channel/y/x={bad[1:].tolist()}"
                )
            if not np.array_equal(actual_obs, expected_obs):
                bad = np.argwhere(actual_obs != expected_obs)[0]
                raise AssertionError(
                    f"ego_obs mismatch at row {begin + int(bad[0])}, "
                    f"channel/y/x={bad[1:].tolist()}"
                )

            # Every categorical group remains exactly one-hot at every cell.
            offset = 0
            for size in OBS_PLANE_SIZES:
                if not np.all(actual_obs[:, offset : offset + size].sum(1) == 1):
                    raise AssertionError(
                        f"ego_obs group {offset}:{offset + size} is not one-hot"
                    )
                offset += size
            vis = actual_vis[:, 0]
            flat = state.reshape(-1, h, w, 16)
            present = flat[..., 1] == 1
            owner = flat[..., 3]
            if not np.all(vis[present & (owner == 1)]):
                raise AssertionError("an ego-owned unit cell is not visible")
            hidden = ~vis
            # A second independent invariant: every hidden raster cell is the
            # exact empty categorical template.
            hidden_obs = actual_obs.transpose(0, 2, 3, 1)[hidden]
            if hidden_obs.size and not np.all(hidden_obs == empty):
                raise AssertionError("a hidden cell is not empty-encoded")

            opponent = present & (owner == 2)
            neutral = present & (owner == 0)
            visible_opponent_per_frame = (opponent & vis).reshape(len(obs), -1).any(1)
            totals["rows"] += len(obs)
            totals["visible_cells"] += int(vis.sum())
            totals["hidden_cells"] += int(hidden.sum())
            totals["visible_opponent_units"] += int((opponent & vis).sum())
            totals["hidden_opponent_units"] += int((opponent & hidden).sum())
            totals["visible_neutral_units"] += int((neutral & vis).sum())
            totals["hidden_neutral_units"] += int((neutral & hidden).sum())
            totals["frames_without_visible_opponent"] += int(
                (~visible_opponent_per_frame).sum()
            )
            if begin == 0 or end == rows or (begin // block_rows) % 64 == 0:
                print(f"  audited {end:,}/{rows:,} rows", flush=True)

    cells = totals["visible_cells"] + totals["hidden_cells"]
    totals["visible_cell_fraction"] = totals["visible_cells"] / cells
    totals["frames_without_visible_opponent_fraction"] = (
        totals["frames_without_visible_opponent"] / totals["rows"]
    )
    print(json.dumps(totals, indent=2, sort_keys=True), flush=True)
    return totals


def main(argv=None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    if args.audit_only:
        audit(output, block_rows=args.block_rows)
    else:
        if not args.source:
            raise SystemExit("source is required unless --audit-only is used")
        augment(
            Path(args.source),
            output,
            block_rows=max(1, args.block_rows),
            gzip=args.gzip,
        )
        audit(output, block_rows=max(1, args.block_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
