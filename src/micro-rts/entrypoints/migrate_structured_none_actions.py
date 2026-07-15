"""Recover explicitly issued TYPE_NONE actions in legacy HDF5 v4 corpora.

The original bot-action inverse encoder serialized an issued TYPE_NONE exactly
like no issued action. A newly created TYPE_NONE assignment in the arrival is
the engine's lossless record of that command. This one-time migration places an
archival marker (255) in the inactive attack-offset component of the correct
role's dense action. State and arrival datasets are never changed.
"""

# ruff: noqa: E402 -- runnable as a bare script; package roots are inserted below.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for p in (HERE.parents[1], HERE.parents[2]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from entrypoints.structured_v2_common import resolve_latest
from models.dreamer_v2.schema import EXPLICIT_NONE_MARKER


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument(
        "--write",
        action="store_true",
        help="apply in place; without this flag only count recoverable actions",
    )
    args = p.parse_args(argv)
    import h5py

    path = resolve_latest(args.data)
    mode = "r+" if args.write else "r"
    self_count = opponent_count = already = 0
    with h5py.File(path, mode) as f:
        if int(f.attrs.get("format_version", 0)) < 4:
            raise SystemExit("migration requires HDF5 format v4")
        total = len(f["state"])
        for start in range(0, total, args.chunk):
            end = min(start + args.chunk, total)
            state = f["state"][start:end]
            nxt = f["next_state"][start:end]
            glob = f["globals"][start:end]
            action = f["action"][start:end]
            opponent = f["opponent_action"][start:end]
            new_none = (
                (state[..., 1] == 1)
                & (state[..., 7] == 0)
                & (nxt[..., 7] == 1)
                & (nxt[..., 8] == 0)
                & (nxt[..., 13] == glob[:, None, 0])
            )
            self_mask = new_none & (state[..., 3] == 1) & (action[..., 0] == 0)
            opp_mask = new_none & (state[..., 3] == 2) & (opponent[..., 0] == 0)
            already += int(
                (self_mask & (action[..., 6] == EXPLICIT_NONE_MARKER)).sum()
                + (opp_mask & (opponent[..., 6] == EXPLICIT_NONE_MARKER)).sum()
            )
            self_count += int(self_mask.sum())
            opponent_count += int(opp_mask.sum())
            if args.write and (self_mask.any() or opp_mask.any()):
                action[..., 6][self_mask] = EXPLICIT_NONE_MARKER
                opponent[..., 6][opp_mask] = EXPLICIT_NONE_MARKER
                f["action"][start:end] = action
                f["opponent_action"][start:end] = opponent
        if args.write:
            f.attrs["explicit_none_action_marker"] = EXPLICIT_NONE_MARKER
            f.flush()
    print(
        f"[none-action-migration] data={path} write={str(args.write).lower()} "
        f"self={self_count} opponent={opponent_count} already_marked={already}"
    )


if __name__ == "__main__":
    main()
