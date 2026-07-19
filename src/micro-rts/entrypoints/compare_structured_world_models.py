"""Compare continuous-flow and discrete MicroRTS world models on held-out rollouts."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
for path in (HERE.parents[1], HERE.parents[2]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from collectors.offline_data import build_mrts_loader, to_device  # noqa: E402
from entrypoints.structured_v2_common import device_from  # noqa: E402
from models.dreamer_v2 import (  # noqa: E402
    DiscreteDynamicsConfig,
    DiscreteStructuredWorldModel,
    DiscreteTokenizerConfig,
    StructuredDynamicsConfig,
    StructuredTokenizerConfig,
    StructuredWorldModelV2,
)


def _canonical(state):
    state = state.clone()
    state[..., 2] = -1
    return state


def _f1(pred, target):
    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    return float(2 * tp / (2 * tp + fp + fn).clamp_min(1))


def _measure(pred, pred_globals, target, target_globals, origin):
    occupied = target[..., 1].bool()
    exact = (pred == target).all(-1)
    changed_pred = (pred != origin).any(-1)
    changed_target = (target != origin).any(-1)
    return {
        "present": float((pred[..., 1] == target[..., 1]).float().mean()),
        "unit_type": (
            float((pred[..., 4][occupied] == target[..., 4][occupied]).float().mean())
            if occupied.any()
            else 1.0
        ),
        "exact_cell": float(exact.float().mean()),
        "exact_frame": float(exact.all(-1).float().mean()),
        "exact_globals": float((pred_globals == target_globals).all(-1).float().mean()),
        "changed_f1": _f1(changed_pred, changed_target),
    }


def _add(store, model_name, horizon, values):
    for key, value in values.items():
        store[(model_name, horizon, key)].append(float(value))


def _load_discrete(path, device):
    payload = torch.load(path, map_location="cpu")
    tc = DiscreteTokenizerConfig.from_dict(payload["discrete_tokenizer_cfg"])
    dc = DiscreteDynamicsConfig.from_dict(payload["discrete_dynamics_cfg"])
    model = DiscreteStructuredWorldModel(tuple(payload["grid_hw"]), tc, dc).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


def _load_continuous(path, device):
    payload = torch.load(path, map_location="cpu")
    tc = StructuredTokenizerConfig.from_dict(payload["tokenizer_cfg"])
    dc = StructuredDynamicsConfig.from_dict(payload["dynamics_cfg"])
    model = StructuredWorldModelV2(tuple(payload["grid_hw"]), tc, dc).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discrete", required=True)
    parser.add_argument("--continuous", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = device_from(args.device)
    discrete = _load_discrete(args.discrete, device)
    continuous = _load_continuous(args.continuous, device)
    loader = build_mrts_loader(
        args.data,
        task="structured_dynamics_eval",
        seq_len=args.horizon,
        batch_size=args.batch,
        num_workers=0,
        locking=False,
        val_frac=args.val_frac,
        split="val",
        shuffle=False,
        drop_last=False,
        fixed_chunk_batches=args.batches,
        fixed_chunk_seed=args.seed,
    )
    horizons = sorted({1, 2, 4, args.horizon})
    totals = defaultdict(list)

    with torch.no_grad():
        for index, raw in enumerate(loader):
            if index >= args.batches:
                break
            batch = to_device(raw, device)
            origin = _canonical(batch["state"][:, 0])
            d_state = origin.clone()
            d_globals = batch["globals"][:, 0].clone()
            c_state = origin.clone()
            c_globals = batch["globals"][:, 0].clone()
            c_latent = continuous.tokenizer.encode(c_state, c_globals)

            for step in range(args.horizon):
                action = batch["action"][:, step]
                opponent = batch["opponent_action"][:, step]
                target = _canonical(batch["next_state"][:, step])
                target_globals = batch["next_globals"][:, step]

                (d_state, d_globals), _ = discrete.generate_next(
                    d_state, d_globals, action, opponent
                )
                d_state = _canonical(d_state)

                events, valid, _ = continuous.action_events(c_state, action, opponent)
                c_latent = continuous.dynamics.sample_next(
                    c_latent,
                    events,
                    valid,
                    args.flow_steps,
                    state_token_valid=continuous.state_token_valid(c_state),
                )
                c_state, c_globals = continuous.tokenizer.discretize(
                    continuous.tokenizer.decode(c_latent)
                )
                c_state = _canonical(c_state)

                horizon = step + 1
                if horizon in horizons:
                    _add(
                        totals,
                        "discrete",
                        horizon,
                        _measure(d_state, d_globals, target, target_globals, origin),
                    )
                    _add(
                        totals,
                        "continuous",
                        horizon,
                        _measure(c_state, c_globals, target, target_globals, origin),
                    )
                    _add(
                        totals,
                        "copy",
                        horizon,
                        _measure(
                            origin,
                            batch["globals"][:, 0],
                            target,
                            target_globals,
                            origin,
                        ),
                    )

    print(
        f"[world-model-compare] heldout batches={args.batches} batch={args.batch} "
        f"horizon={args.horizon} flow_steps={args.flow_steps}"
    )
    header = (
        "model horizon present unit_type exact_cell exact_frame "
        "exact_globals changed_f1"
    )
    print(header)
    for horizon in horizons:
        for model_name in ("copy", "continuous", "discrete"):
            values = []
            for key in (
                "present",
                "unit_type",
                "exact_cell",
                "exact_frame",
                "exact_globals",
                "changed_f1",
            ):
                items = totals[(model_name, horizon, key)]
                values.append(sum(items) / max(len(items), 1))
            print(
                f"{model_name:10s} {horizon:7d} "
                + " ".join(f"{value:.6f}" for value in values)
            )


if __name__ == "__main__":
    main()
