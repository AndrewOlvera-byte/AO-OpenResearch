"""Evaluate one-step mechanics and flow sampling for MicroRTS world model v2."""

# ruff: noqa: E402 -- runnable as a bare script; package roots are inserted below.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for p in (HERE.parents[1], HERE.parents[2]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

from collectors.offline_data import build_mrts_loader, to_device
from entrypoints.pretrain_common import amp_ctx, setup_backend
from models.dreamer_v2 import (
    StructuredDynamicsConfig,
    StructuredTokenizerConfig,
    StructuredWorldModelV2,
)
from entrypoints.structured_v2_common import device_from, resolve_latest


def _f1(pred, true):
    tp = (pred & true).sum().float()
    fp = (pred & ~true).sum().float()
    fn = (~pred & true).sum().float()
    return float((2 * tp / (2 * tp + fp + fn).clamp_min(1)))


def _canonicalize_unit_ids(state):
    state = state.clone()
    state[..., 2] = -1
    return state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--flow-steps", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    device = device_from(a.device)
    setup_backend(device)
    ckpt = torch.load(a.checkpoint, map_location="cpu")
    tc = StructuredTokenizerConfig.from_dict(ckpt["tokenizer_cfg"])
    dc = StructuredDynamicsConfig.from_dict(ckpt["dynamics_cfg"])
    model = StructuredWorldModelV2(tuple(ckpt["grid_hw"]), tc, dc).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    loader = build_mrts_loader(
        resolve_latest(a.data),
        task="structured_dynamics_eval",
        seq_len=1,
        batch_size=a.batch,
        num_workers=0,
        locking=False,
        val_frac=a.val_frac,
        split="val",
        shuffle=False,
        drop_last=False,
        fixed_chunk_batches=a.batches,
        fixed_chunk_seed=a.seed,
    )
    sums = {
        "latent_mse": 0.0,
        "copy_mse": 0.0,
        "present_acc": 0.0,
        "type_acc": 0.0,
        "exact_cell": 0.0,
        "exact_occupied_cell": 0.0,
        "exact_assigned_cell": 0.0,
        "exact_frame": 0.0,
        "exact_globals": 0.0,
        "exact_roundtrip": 0.0,
        "changed_f1": 0.0,
        "self_cf_gap": 0.0,
        "paired_cf_latent_mse": 0.0,
        "paired_cf_effect_f1": 0.0,
    }
    n = cf_n = 0
    with torch.no_grad():
        for batch in loader:
            if n >= a.batches:
                break
            b = to_device(batch, device)
            state, glob = b["state"][:, 0], b["globals"][:, 0]
            nxt, nglob = b["next_state"][:, 0], b["next_globals"][:, 0]
            state = _canonicalize_unit_ids(state)
            nxt = _canonicalize_unit_ids(nxt)
            act, opp = b["action"][:, 0], b["opponent_action"][:, 0]
            z0 = model.tokenizer.encode(state, glob)
            z1 = model.tokenizer.encode(nxt, nglob)
            state_valid = model.state_token_valid(state)
            ev, valid, _ = model.action_events(state, act, opp)
            with amp_ctx(device, device.type == "cuda"):
                predz = model.dynamics.sample_next(
                    z0,
                    ev,
                    valid,
                    a.flow_steps,
                    state_token_valid=state_valid,
                )
                pred, pred_globals = model.tokenizer.discretize(
                    model.tokenizer.decode(predz)
                )
            sums["latent_mse"] += float(
                (model.dynamics.normalize(predz) - model.dynamics.normalize(z1))
                .pow(2)
                .mean()
            )
            sums["copy_mse"] += float(
                (model.dynamics.normalize(z0) - model.dynamics.normalize(z1))
                .pow(2)
                .mean()
            )
            sums["present_acc"] += float((pred[..., 1] == nxt[..., 1]).float().mean())
            occupied = nxt[..., 1].bool()
            sums["type_acc"] += (
                float((pred[..., 4][occupied] == nxt[..., 4][occupied]).float().mean())
                if occupied.any()
                else 1.0
            )
            exact_cell = (pred == nxt).all(-1)
            exact_globals = (pred_globals == nglob).all(-1)
            sums["exact_cell"] += float(exact_cell.float().mean())
            sums["exact_occupied_cell"] += float(
                exact_cell[occupied].float().mean() if occupied.any() else 1.0
            )
            assigned = nxt[..., 7].bool()
            sums["exact_assigned_cell"] += float(
                exact_cell[assigned].float().mean() if assigned.any() else 1.0
            )
            sums["exact_frame"] += float(exact_cell.all(-1).float().mean())
            sums["exact_globals"] += float(exact_globals.float().mean())
            sums["exact_roundtrip"] += float(
                (exact_cell.all(-1) & exact_globals).float().mean()
            )
            changed = (state != nxt).any(-1)
            pred_changed = (state != pred).any(-1)
            sums["changed_f1"] += _f1(pred_changed, changed)
            perm = torch.arange(state.shape[0] - 1, -1, -1, device=device)
            sev, sv, _ = model.action_events(state, act[perm], opp)
            shuf = model.dynamics.sample_next(
                z0,
                sev,
                sv,
                a.flow_steps,
                state_token_valid=state_valid,
            )
            true_mse = (
                (model.dynamics.normalize(predz) - model.dynamics.normalize(z1))
                .pow(2)
                .mean()
            )
            shuf_mse = (
                (model.dynamics.normalize(shuf) - model.dynamics.normalize(z1))
                .pow(2)
                .mean()
            )
            sums["self_cf_gap"] += float(shuf_mse - true_mse)
            n += 1
            if "counterfactual_valid" in b:
                cv = b["counterfactual_valid"][:, 0].bool()
                if cv.any():
                    ca = b["counterfactual_action"][:, 0]
                    co = b["counterfactual_opponent_action"][:, 0]
                    cn = b["counterfactual_next_state"][:, 0]
                    cn = _canonicalize_unit_ids(cn)
                    cg = b["counterfactual_next_globals"][:, 0]
                    cev, cvalid, _ = model.action_events(state, ca, co)
                    cpz = model.dynamics.sample_next(
                        z0,
                        cev,
                        cvalid,
                        a.flow_steps,
                        state_token_valid=state_valid,
                    )
                    ctz = model.tokenizer.encode(cn, cg)
                    sums["paired_cf_latent_mse"] += float(
                        (
                            model.dynamics.normalize(cpz[cv])
                            - model.dynamics.normalize(ctz[cv])
                        )
                        .pow(2)
                        .mean()
                    )
                    cps, _ = model.tokenizer.discretize(model.tokenizer.decode(cpz))
                    true_effect = (cn != nxt).any(-1)[cv]
                    pred_effect = (cps != pred).any(-1)[cv]
                    sums["paired_cf_effect_f1"] += _f1(pred_effect, true_effect)
                    cf_n += 1
    parts = []
    for k, v in sums.items():
        denom = cf_n if k.startswith("paired_cf") else n
        parts.append(f"{k}={v / max(denom, 1):.6f}")
    print("[structured-eval] " + " ".join(parts))


if __name__ == "__main__":
    main()
