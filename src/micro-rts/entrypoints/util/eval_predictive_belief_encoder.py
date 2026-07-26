"""Deterministic conditioning evaluation for predictive-belief checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
for root in (HERE.parents[2], HERE.parents[3]):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import registry_imports  # noqa: F401,E402
from collectors.offline_data import to_device  # noqa: E402
from core.registry import build  # noqa: E402
from entrypoints.incomplete_info_common import common_parser, load_config  # noqa: E402


def parser():
    p = common_parser(__doc__, "")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batches", type=int, default=64)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--json-out")
    return p


def _norm(x):
    return F.layer_norm(x.float(), x.shape[-1:])


def _cos(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def _mean_valid(value, valid):
    weight = valid.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1)


@torch.no_grad()
def evaluate(trainer, checkpoint, batches, seed):
    model = trainer.model
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed = ("ego_tokenizer.", "self_action_tokenizer.", "opponent_tokenizer.")
    bad_missing = [name for name in missing if not name.startswith(allowed)]
    if bad_missing or unexpected:
        raise ValueError(f"incompatible checkpoint: missing={bad_missing}, unexpected={unexpected}")
    model.eval()
    torch.manual_seed(seed)
    totals = defaultdict(float)
    count = 0

    for index, raw in enumerate(trainer.val_loader):
        if index >= batches:
            break
        batch = to_device(raw, trainer.device)
        encoded = model(batch)
        _, native = trainer.loss_fn(batch)
        for key, value in native.items():
            totals[f"native/{key.rsplit('/', 1)[-1]}"] += float(value)

        online, target = encoded["online"], encoded["target"]
        action, valid = encoded["action_tokens"], encoded["action_valid"]
        perm = torch.roll(torch.arange(action.shape[0], device=action.device), 1)
        shuffled = model.predictor(
            online["tokens"],
            encoded["action_pool"][perm],
            action_tokens=action[perm],
            action_valid=valid[perm],
        )
        zeroed = model.predictor(
            online["tokens"],
            torch.zeros_like(encoded["action_pool"]),
            action_tokens=torch.zeros_like(action),
            action_valid=torch.zeros_like(valid),
        )

        spatial, visibility, hist_action, hist_valid, _, _ = model.tokenize(batch)
        hist_online, _, _ = model.encode_tokenized(
            spatial, visibility, hist_action[perm], hist_valid[perm]
        )
        hist_pred = model.predictor(
            hist_online["tokens"],
            encoded["action_pool"],
            action_tokens=action,
            action_valid=valid,
        )

        for horizon, pred in encoded["predictions"].items():
            tgt = target["tokens"][:, horizon:]
            true_mse = (_norm(pred) - _norm(tgt)).square().mean()
            shuffle_mse = (_norm(shuffled[horizon]) - _norm(tgt)).square().mean()
            zero_mse = (_norm(zeroed[horizon]) - _norm(tgt)).square().mean()
            hist_mse = (_norm(hist_pred[horizon]) - _norm(tgt)).square().mean()
            prefix = f"h{horizon}"
            totals[f"{prefix}/matched_mse"] += float(true_mse)
            totals[f"{prefix}/action_shuffle_mse"] += float(shuffle_mse)
            totals[f"{prefix}/action_shuffle_gap"] += float(shuffle_mse - true_mse)
            totals[f"{prefix}/action_shuffle_degradation_pct"] += float(
                100 * (shuffle_mse - true_mse) / true_mse.clamp_min(1e-8)
            )
            totals[f"{prefix}/action_zero_gap"] += float(zero_mse - true_mse)
            totals[f"{prefix}/history_action_shuffle_gap"] += float(hist_mse - true_mse)
            totals[f"{prefix}/action_output_sensitivity"] += float(
                (_norm(shuffled[horizon]) - _norm(pred)).square().mean()
            )
            offset = 0
            for branch, size in zip(
                ("self", "opponent", "static", "interaction"),
                model.cfg.branch_sizes,
            ):
                sl = slice(offset, offset + size)
                branch_true = (
                    _norm(pred[..., sl, :]) - _norm(tgt[..., sl, :])
                ).square().mean()
                branch_shuffle = (
                    _norm(shuffled[horizon][..., sl, :]) - _norm(tgt[..., sl, :])
                ).square().mean()
                totals[f"{prefix}/{branch}_matched_mse"] += float(branch_true)
                totals[f"{prefix}/{branch}_shuffle_gap"] += float(
                    branch_shuffle - branch_true
                )
                totals[f"{prefix}/{branch}_shuffle_degradation_pct"] += float(
                    100
                    * (branch_shuffle - branch_true)
                    / branch_true.clamp_min(1e-8)
                )
                totals[f"{prefix}/{branch}_output_sensitivity"] += float(
                    (
                        _norm(shuffled[horizon][..., sl, :])
                        - _norm(pred[..., sl, :])
                    ).square().mean()
                )
                offset += size

        anchors = batch["state"].shape[1] - model.opponent_tokenizer.max_horizon
        opponent_prediction = model.opponent_plan_head(
            online["opponent"][:, :anchors]
        )
        opponent_target = model.opponent_tokenizer.encode(
            batch["state"], batch["opponent_action"]
        )[0]
        opponent_cos = _cos(opponent_prediction, opponent_target)
        opponent_shuffle_cos = _cos(
            opponent_prediction, opponent_target[perm]
        )
        totals["probe/opponent_plan_cosine"] += float(opponent_cos.mean())
        totals["probe/opponent_plan_shuffle_gap"] += float(
            (opponent_cos - opponent_shuffle_cos).mean()
        )

        pooled = {
            name: online[name].float().mean(-2)
            for name in ("self", "opponent", "static", "interaction")
        }
        for other in ("self", "static", "interaction"):
            totals[f"probe/opponent_{other}_pooled_cosine"] += float(
                _cos(pooled["opponent"], pooled[other]).mean()
            )

        future = model.inverse_action_set(
            online["self"][:, :-1], encoded["predictions"][1].split(
                tuple(model.cfg.branch_sizes), dim=-2
            )[0]
        )
        inverse_cos = _cos(future, action[:, :-1])
        inverse_shuffle_cos = _cos(future, action[perm, :-1])
        inv_valid = valid[:, :-1]
        totals["probe/action_set_inverse_cosine"] += float(
            _mean_valid(inverse_cos, inv_valid)
        )
        totals["probe/action_set_inverse_shuffle_gap"] += float(
            _mean_valid(inverse_cos - inverse_shuffle_cos, inv_valid)
        )
        count += 1

    if not count:
        raise RuntimeError("validation loader produced no batches")
    result = {key: value / count for key, value in sorted(totals.items())}
    result.update(
        checkpoint_step=int(checkpoint["step"]),
        evaluated_batches=count,
        seed=seed,
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.no_wandb = True
    cfg = load_config(args)
    trainer = build(
        "trainer",
        type=(cfg.trainer or {})["type"],
        cfg=cfg,
        args=args,
    )
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    result = evaluate(trainer, checkpoint, args.batches, args.seed)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


if __name__ == "__main__":
    main()
