"""Audit HDF5 v4 schema validity and deterministic Markov contradictions."""

# ruff: noqa: E402 -- runnable as a bare script; package roots are inserted below.

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for p in (HERE.parents[2], HERE.parents[3]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

from entrypoints.structured_v2_common import resolve_latest
from models.dreamer_v2.schema import EXPLICIT_NONE_MARKER


def _effective(action, state, role):
    out = np.zeros_like(action)
    legal = (state[:, 1] == 1) & (state[:, 3] == role) & (state[:, 7] == 0)
    out[legal] = action[legal]
    typ = out[:, 0]
    explicit_none = (typ == 0) & (out[:, 6] == EXPLICIT_NONE_MARKER)
    for comp, required in enumerate((None, 1, 2, 3, 4, 4, 5)):
        if comp and required is not None:
            out[typ != required, comp] = 0
    out[explicit_none, 6] = EXPLICIT_NONE_MARKER
    return out


def _digest(*arrays):
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.digest()


def _canonicalize_unit_ids(state):
    """Remove the JVM-global allocation counter from Markov comparisons."""
    state = state.copy()
    state[..., 2] = -1
    return state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument(
        "--show-conflicts",
        type=int,
        default=0,
        help="print this many first contradictory arrivals, then stop",
    )
    args = p.parse_args(argv)
    import h5py

    path = resolve_latest(args.data)
    seen = {}
    repeated = contradictory = rows = 0
    max_units = max_assignments = max_remaining = 0
    cf_valid = cf_action_diff = cf_effect = 0
    with h5py.File(path, "r") as f:
        if int(f.attrs.get("format_version", 0)) < 4:
            raise SystemExit("structured audit requires HDF5 format v4")
        total = len(f["state"])
        total = min(total, args.max_rows) if args.max_rows else total
        for start in range(0, total, args.chunk):
            end = min(start + args.chunk, total)
            state, glob = f["state"][start:end], f["globals"][start:end]
            nxt, nglob = f["next_state"][start:end], f["next_globals"][start:end]
            act, opp = f["action"][start:end], f["opponent_action"][start:end]
            max_units = max(max_units, int(state[..., 1].sum(axis=1).max()))
            max_assignments = max(max_assignments, int(state[..., 7].sum(axis=1).max()))
            max_remaining = max(max_remaining, int(state[..., 15].max()))
            state = _canonicalize_unit_ids(state)
            nxt = _canonicalize_unit_ids(nxt)
            if "counterfactual_valid" in f:
                cv = f["counterfactual_valid"][start:end].astype(bool)
                if cv.any():
                    ca = f["counterfactual_action"][start:end]
                    cn = f["counterfactual_next_state"][start:end]
                    cn = _canonicalize_unit_ids(cn)
                    cf_valid += int(cv.sum())
                    cf_action_diff += int(
                        ((ca != act).reshape(len(cv), -1).any(axis=1) & cv).sum()
                    )
                    cf_effect += int(
                        ((cn != nxt).reshape(len(cv), -1).any(axis=1) & cv).sum()
                    )
            for i in range(end - start):
                a = _effective(act[i], state[i], 1)
                o = _effective(opp[i], state[i], 2)
                key = _digest(state[i], glob[i], a, o)
                val = _digest(nxt[i], nglob[i])
                old = seen.get(key)
                if old is not None:
                    repeated += 1
                    old_val = old[0] if args.show_conflicts else old
                    conflict = old_val != val
                    contradictory += conflict
                    if conflict and args.show_conflicts:
                        old_row = old[1]
                        old_nxt = _canonicalize_unit_ids(f["next_state"][old_row])
                        old_nglob = f["next_globals"][old_row]
                        cell_diff = np.argwhere(old_nxt != nxt[i])
                        global_diff = np.argwhere(old_nglob != nglob[i]).flatten()
                        print(
                            f"[structured-conflict] rows=({old_row},{start + i}) "
                            f"cell_diff={cell_diff.tolist()} "
                            f"old_cell_values={old_nxt[tuple(cell_diff.T)].tolist()} "
                            f"new_cell_values={nxt[i][tuple(cell_diff.T)].tolist()} "
                            f"global_diff={global_diff.tolist()} "
                            f"old_global_values={old_nglob[global_diff].tolist()} "
                            f"new_global_values={nglob[i][global_diff].tolist()}"
                        )
                        if contradictory >= args.show_conflicts:
                            raise SystemExit(3)
                else:
                    seen[key] = (val, start + i) if args.show_conflicts else val
                rows += 1
    rate = contradictory / max(repeated, 1)
    print(
        f"[structured-audit] data={path} canonical_unit_ids=true rows={rows} "
        f"unique_inputs={len(seen)} "
        f"repeated_inputs={repeated} contradictions={contradictory} rate={rate:.8%} "
        f"max_units={max_units} max_assignments={max_assignments} "
        f"max_remaining={max_remaining} cf_valid={cf_valid} "
        f"cf_action_diff={cf_action_diff} cf_effect={cf_effect}"
    )
    if contradictory:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
