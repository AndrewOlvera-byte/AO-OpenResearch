"""Audit complete-episode world-action HDF5 splits and their balance."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


REQUIRED_ROWS = (
    "obs", "action", "opponent_action", "mask", "reward", "raw_rewards",
    "done", "is_first", "terminal_obs", "state", "globals", "next_state",
    "next_globals", "counterfactual_action", "counterfactual_opponent_action",
    "counterfactual_obs", "counterfactual_next_state",
    "counterfactual_next_globals", "counterfactual_valid", "ego_obs",
    "ego_visibility", "counterfactual_ego_obs",
    "counterfactual_ego_visibility",
)
REQUIRED_EPISODES = (
    "start", "length", "map_id", "opponent_id", "policy_id", "action_noise",
    "episode_id", "collection_seed", "seat", "terminal_outcome",
)


def _length_bucket(length: int) -> str:
    for edge in (64, 128, 256, 512, 1024):
        if length <= edge:
            return f"<= {edge}"
    return "> 1024"


def audit(path: Path) -> dict:
    counters = {
        name: Counter()
        for name in ("map", "opponent", "policy", "seat", "outcome", "length_bucket",
                     "action_noise", "contact_phase")
    }
    with h5py.File(path, "r") as f:
        missing = [name for name in REQUIRED_ROWS if name not in f]
        if missing:
            raise AssertionError(f"{path}: missing row datasets: {missing}")
        if "traj" not in f:
            raise AssertionError(f"{path}: missing traj/ episode index")
        missing = [name for name in REQUIRED_EPISODES if name not in f["traj"]]
        if missing:
            raise AssertionError(f"{path}: missing episode datasets: {missing}")
        if not bool(f.attrs.get("episode_aware", False)):
            raise AssertionError(f"{path}: episode_aware is not true")
        if not bool(f.attrs.get("fog_augmentation_complete", False)):
            raise AssertionError(f"{path}: fog augmentation is incomplete")

        rows = len(f["obs"])
        for name in REQUIRED_ROWS:
            if len(f[name]) != rows:
                raise AssertionError(f"{path}: {name} row count differs from obs")
        g = f["traj"]
        episodes = len(g["start"])
        for name in REQUIRED_EPISODES:
            if len(g[name]) != episodes:
                raise AssertionError(f"{path}: traj/{name} length mismatch")

        starts = g["start"][:]
        lengths = g["length"][:]
        if episodes == 0:
            raise AssertionError(f"{path}: no complete episodes")
        if starts[0] != 0 or not np.array_equal(starts[1:], starts[:-1] + lengths[:-1]):
            raise AssertionError(f"{path}: episode rows are not contiguous and gap-free")
        if int(starts[-1] + lengths[-1]) != rows:
            raise AssertionError(f"{path}: episode index does not cover all rows")
        if len(np.unique(g["episode_id"][:])) != episodes:
            raise AssertionError(f"{path}: duplicate episode_id")
        if not np.all(g["collection_seed"][:] == int(f.attrs["collection_seed"])):
            raise AssertionError(f"{path}: inconsistent collection_seed")

        maps = list(f.attrs["maps"])
        opponents = list(f.attrs["opponents"])
        policies = list(f.attrs["policies"])
        cf_valid_total = cf_changed_total = contact_rows = 0
        for i, (start, length) in enumerate(zip(starts, lengths)):
            start, length = int(start), int(length)
            end = start + length
            first = f["is_first"][start:end].astype(bool)
            done = f["done"][start:end].astype(bool)
            if not first[0] or first[1:].any():
                raise AssertionError(f"{path}: bad is_first mask in episode {i}")
            if not done[-1] or done[:-1].any():
                raise AssertionError(f"{path}: bad done mask in episode {i}")
            if not f["terminal_obs"][end - 1].any():
                raise AssertionError(f"{path}: episode {i} lost terminal_obs")
            if f["next_globals"][end - 1, 0] != f["globals"][end - 1, 0] + 1:
                raise AssertionError(f"{path}: episode {i} terminal tick is misaligned")

            map_id = int(g["map_id"][i])
            opp_id = int(g["opponent_id"][i])
            policy_id = int(g["policy_id"][i])
            outcome = int(g["terminal_outcome"][i])
            actual_outcome = int(np.sign(float(f["raw_rewards"][end - 1, 0])))
            if outcome != actual_outcome:
                raise AssertionError(f"{path}: episode {i} terminal outcome mismatch")
            counters["map"][str(maps[map_id])] += 1
            counters["opponent"][str(opponents[opp_id])] += 1
            counters["policy"][str(policies[policy_id])] += 1
            counters["seat"][str(int(g["seat"][i]))] += 1
            counters["outcome"][str(outcome)] += 1
            counters["length_bucket"][_length_bucket(length)] += 1
            counters["action_noise"][f"{float(g['action_noise'][i]):g}"] += 1

            cv = f["counterfactual_valid"][start:end].astype(bool)
            cf_next = f["counterfactual_next_state"][start:end]
            factual_next = f["next_state"][start:end]
            # JVM unit allocation IDs are process-global and not semantic state.
            cf_next[..., 2] = -1
            factual_next[..., 2] = -1
            changed = (cf_next != factual_next).reshape(length, -1).any(1)
            cf_valid_total += int(cv.sum())
            cf_changed_total += int((cv & changed).sum())
            if cv.any():
                if not np.all(
                    f["counterfactual_next_globals"][start:end][cv, 0]
                    == f["globals"][start:end][cv, 0] + 1
                ):
                    raise AssertionError(f"{path}: counterfactual tick alignment failure")
                if not np.array_equal(
                    f["counterfactual_opponent_action"][start:end][cv],
                    f["opponent_action"][start:end][cv],
                ):
                    raise AssertionError(f"{path}: counterfactual opponent action not fixed")

            state = f["state"][start:end]
            visible = f["ego_visibility"][start:end, 0].reshape(length, -1)
            opponent = (state[..., 1] == 1) & (state[..., 3] == 2)
            contact = (opponent & visible).any(1)
            contact_rows += int(contact.sum())
            thirds = np.minimum(np.arange(length) * 3 // max(length, 1), 2)
            for phase, label in enumerate(("early", "middle", "late")):
                idx = thirds == phase
                counters["contact_phase"][f"{label}/contact"] += int((contact & idx).sum())
                counters["contact_phase"][f"{label}/hidden"] += int((~contact & idx).sum())

        result = {
            "path": str(path),
            "split": str(f.attrs.get("split", "")),
            "collection_seed": int(f.attrs["collection_seed"]),
            "manifest_hash": str(f.attrs["manifest_hash"]),
            "rows": rows,
            "episodes": episodes,
            "dropped_partial_rows": int(f.attrs.get("dropped_partial_rows", 0)),
            "counterfactual_valid_fraction": cf_valid_total / rows,
            "counterfactual_state_change_fraction": (
                cf_changed_total / max(cf_valid_total, 1)
            ),
            "opponent_contact_fraction": contact_rows / rows,
            "balance": {name: dict(sorted(value.items())) for name, value in counters.items()},
        }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    reports = [audit(path) for path in args.paths]
    splits = [report["split"] for report in reports]
    seeds = [report["collection_seed"] for report in reports]
    hashes = [report["manifest_hash"] for report in reports]
    if len(set(splits)) != len(splits):
        raise AssertionError(f"duplicate split labels: {splits}")
    if len(set(seeds)) != len(seeds):
        raise AssertionError(f"collection seeds are not disjoint: {seeds}")
    if len(set(hashes)) != len(hashes):
        raise AssertionError("manifest hashes are not disjoint")
    text = json.dumps(reports, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
