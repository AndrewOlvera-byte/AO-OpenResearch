"""Deterministic promotion evaluation for incomplete-information belief dynamics.

This evaluation complements the stochastic training-time validation loss.  It
uses fixed held-out batches and fixed Gaussian noise, runs the same multi-step
Euler sampler used by :meth:`BeliefDynamicsModel.sample_next`, and measures
whether matched history, opponent intent, and self action improve the sampled
next-state belief over paired cross-batch shuffles.

Example (inside the research container)::

    python src/micro-rts/entrypoints/eval_belief_dynamics.py \
      --ckpt checkpoints/pretrain_belief_dynamics_medium_v1/best.pt \
      --out checkpoints/pretrain_belief_dynamics_medium_v1/promotion_best.json

The final verdict is deliberately conservative.  Reconstruction gates only
ask for a useful structured belief, not exact rendering.  Conditioning gates
are mandatory because a plausible unconditional state prior cannot support
counterfactual planning.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
for root in (HERE.parents[1], HERE.parents[2]):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from collectors.offline_data import to_device  # noqa: E402
from entrypoints.incomplete_info_common import (  # noqa: E402
    load_config,
    load_full_state_tokenizer,
    load_stage_weights,
    make_loaders,
    resolve_device,
    resolve_path,
)
from models.incomplete_info import (  # noqa: E402
    BeliefDynamicsConfig,
    BeliefDynamicsModel,
    EgoTokenizerConfig,
    HistoryConfig,
    IntentPriorConfig,
    OpponentIntentPriorModel,
    OpponentPlanTokenizerConfig,
    SelfActionTokenizerConfig,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--exp",
        default="micro-rts/paper/incomplete_info/pretrain_belief_dynamics_medium_v1",
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--flow-steps", type=int, nargs="+", default=[4, 8])
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--out")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="K=V")
    return p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(cfg, args, device):
    """Recreate the frozen conditioner and load a trainable-only checkpoint."""
    model_cfg, training = cfg.model or {}, cfg.training or {}
    flow_cfg = BeliefDynamicsConfig.from_dict(model_cfg.get("belief_dynamics"))
    intent_path = resolve_path(training["intent_prior_ckpt"])
    intent_checkpoint = torch.load(intent_path, map_location="cpu", weights_only=False)
    history_cfg = HistoryConfig.from_dict(intent_checkpoint["history_cfg"])
    opponent_cfg = OpponentPlanTokenizerConfig.from_dict(
        intent_checkpoint["opponent_tokenizer_cfg"]
    )
    seq_len = int(training.get("seq_len", history_cfg.context_length))
    _, loader, data = make_loaders(
        cfg, args, task="incomplete_dynamics", seq_len=seq_len
    )
    if loader is None:
        raise ValueError("promotion evaluation requires a nonzero validation split")

    full_tokenizer_path = training.get(
        "full_state_tokenizer_ckpt",
        intent_checkpoint["full_state_tokenizer_ckpt"],
    )
    teacher, tokenizer_cfg, _ = load_full_state_tokenizer(
        full_tokenizer_path, loader.dataset, device
    )
    ego_cfg = EgoTokenizerConfig.from_dict(intent_checkpoint["ego_tokenizer_cfg"])
    action_cfg = SelfActionTokenizerConfig.from_dict(
        intent_checkpoint["self_action_tokenizer_cfg"]
    )
    intent_cfg = IntentPriorConfig.from_dict(intent_checkpoint["intent_prior_cfg"])
    intent_model = OpponentIntentPriorModel(
        teacher,
        loader.dataset.grid_hw,
        ego_cfg=ego_cfg,
        self_action_cfg=action_cfg,
        opponent_cfg=opponent_cfg,
        history_cfg=history_cfg,
        intent_cfg=intent_cfg,
    ).to(device)
    load_stage_weights(
        intent_model.ego_tokenizer,
        training.get("ego_tokenizer_ckpt", intent_checkpoint["ego_tokenizer_ckpt"]),
        ("tokenizer.", ""),
    )
    load_stage_weights(
        intent_model.self_action_tokenizer,
        training.get(
            "self_action_tokenizer_ckpt",
            intent_checkpoint["self_action_tokenizer_ckpt"],
        ),
    )
    load_stage_weights(
        intent_model.opponent_tokenizer,
        training.get(
            "opponent_tokenizer_ckpt",
            intent_checkpoint["opponent_tokenizer_ckpt"],
        ),
    )
    load_stage_weights(intent_model, intent_path)

    model = BeliefDynamicsModel(intent_model, flow_cfg).to(device)
    checkpoint_path = resolve_path(args.ckpt)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = tuple(f"intent_model.{name}" for name in intent_model.state_dict())
    bad_missing = [name for name in missing if name not in allowed_missing]
    if bad_missing or unexpected:
        raise ValueError(
            f"incompatible checkpoint: missing={bad_missing}, unexpected={unexpected}"
        )
    model.freeze_conditioner().eval()
    return model, loader, checkpoint, checkpoint_path, data, tokenizer_cfg


def _sum(acc, key, value, weight=1.0):
    acc[key] += float(value) * float(weight)
    acc[f"{key}__weight"] += float(weight)


def _finish(acc):
    return {
        key: (
            value / max(acc[f"{key}__weight"], 1.0)
            if f"{key}__weight" in acc
            else value
        )
        for key, value in acc.items()
        if not key.endswith("__weight")
    }


def _latent_error(pred, target):
    return (pred.float() - target.float()).square().mean(dim=(-1, -2))


def _condition_probe(model, condition, target, noise, steps, component):
    matched = model.flow.sample(
        condition["history"],
        condition["intent"],
        condition["action"],
        condition["action_valid"],
        steps=steps,
        noise=noise,
    )
    shuffled = dict(condition)
    shuffled[component] = condition[component].roll(1, 0)
    if component == "action":
        shuffled["action_valid"] = condition["action_valid"].roll(1, 0)
        union = condition["action_valid"] | shuffled["action_valid"]
        token_change = (
            (condition["action"].float() - shuffled["action"].float())
            .square()
            .mean(-1)
        )
        eligible = (
            (condition["action_valid"] != shuffled["action_valid"]).any(-1)
            | ((token_change > 1e-8) & union).any(-1)
        )
    else:
        difference = (
            condition[component].float() - shuffled[component].float()
        ).square().flatten(1).mean(-1)
        eligible = difference > 1e-8
    counterfactual = model.flow.sample(
        shuffled["history"],
        shuffled["intent"],
        shuffled["action"],
        shuffled["action_valid"],
        steps=steps,
        noise=noise,
    )
    matched_error = _latent_error(matched, target)
    shuffled_error = _latent_error(counterfactual, target)
    delta = shuffled_error - matched_error
    output_delta = _latent_error(counterfactual, matched)
    return matched, matched_error, delta, output_delta, eligible


def _decoded_counts(model, sampled, batch, anchors):
    raw = model.denormalize_state(sampled)
    decoded = model.full_state_tokenizer.decode(raw)
    predicted_state, predicted_globals = model.full_state_tokenizer.discretize(decoded)
    true_state = batch["state"][:, anchors].clone()
    true_state[..., 2] = -1
    visible = batch["local_visibility"][:, anchors].squeeze(-3).flatten(-2).bool()
    exact = (predicted_state == true_state).all(-1)
    occupied = true_state[..., 1].bool()
    return {
        "visible_correct": (exact & visible).sum().item(),
        "visible_count": visible.sum().item(),
        "hidden_correct": (exact & ~visible).sum().item(),
        "hidden_count": (~visible).sum().item(),
        "hidden_occupied_correct": (exact & ~visible & occupied).sum().item(),
        "hidden_occupied_count": ((~visible) & occupied).sum().item(),
        "occupied_correct": (exact & occupied).sum().item(),
        "occupied_count": occupied.sum().item(),
        "global_correct": (
            predicted_globals == batch["globals"][:, anchors]
        ).all(-1).sum().item(),
        "global_count": predicted_globals.shape[0],
    }


@torch.no_grad()
def evaluate(model, loader, device, args):
    totals = {steps: defaultdict(float) for steps in args.flow_steps}
    counts = {steps: defaultdict(float) for steps in args.flow_steps}
    evaluated = 0
    amp = args.amp and device.type == "cuda"
    amp_ctx = (
        lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        if amp
        else contextlib.nullcontext()
    )

    for batch_index, batch in enumerate(loader):
        if batch_index >= args.batches:
            break
        if batch["state"].shape[0] < 2:
            continue
        batch = to_device(batch, device)
        with amp_ctx():
            encoded = model.encode_condition(batch, sample_intent=False)
            anchors = encoded["history_registers"].shape[1]
            target_raw = model.full_state_tokenizer.encode(
                batch["state"][:, anchors], batch["globals"][:, anchors]
            )
            target = model.normalize_state(target_raw)
            condition = {
                "history": encoded["history_registers"][:, -1],
                "intent": encoded["plan_tokens"][:, -1],
                "action": encoded["action_tokens"][:, -1],
                "action_valid": encoded["action_valid"][:, -1],
            }

        batch_size = target.shape[0]
        for steps in args.flow_steps:
            sample_errors = []
            sample_noises = []
            first_sample = None
            for sample_index in range(args.samples):
                generator = torch.Generator(device=device)
                generator.manual_seed(
                    args.seed + batch_index * 100_003 + steps * 1_009 + sample_index
                )
                noise = torch.randn(
                    target.shape,
                    generator=generator,
                    device=device,
                    dtype=target.dtype,
                )
                with amp_ctx():
                    sampled = model.flow.sample(
                        condition["history"],
                        condition["intent"],
                        condition["action"],
                        condition["action_valid"],
                        steps=steps,
                        noise=noise,
                    )
                error = _latent_error(sampled, target)
                sample_errors.append(error)
                sample_noises.append(noise)
                if first_sample is None:
                    first_sample = sampled
            errors = torch.stack(sample_errors)
            _sum(totals[steps], "latent/mse", errors[0].mean(), batch_size)
            _sum(totals[steps], "latent/best_of_n_mse", errors.min(0).values.mean(), batch_size)
            _sum(
                totals[steps],
                "latent/sample_mse_std",
                errors.std(0, unbiased=False).mean(),
                batch_size,
            )

            decoded = _decoded_counts(model, first_sample, batch, anchors)
            for key, value in decoded.items():
                counts[steps][key] += value

            for component in ("history", "intent", "action"):
                matched_errors = []
                shuffled_errors = []
                output_deltas = []
                for probe_noise in sample_noises:
                    with amp_ctx():
                        (
                            _,
                            matched_error,
                            delta,
                            output_delta,
                            eligible,
                        ) = _condition_probe(
                            model, condition, target, probe_noise, steps, component
                        )
                    matched_errors.append(matched_error)
                    shuffled_errors.append(matched_error + delta)
                    output_deltas.append(output_delta)
                matched_errors = torch.stack(matched_errors)
                shuffled_errors = torch.stack(shuffled_errors)
                output_deltas = torch.stack(output_deltas)
                ensemble_delta = shuffled_errors.mean(0) - matched_errors.mean(0)
                best_delta = (
                    shuffled_errors.min(0).values - matched_errors.min(0).values
                )
                prefix = f"condition/{component}"
                eligible_count = int(eligible.sum())
                totals[steps][f"{prefix}_eligible"] += eligible_count
                if not eligible_count:
                    continue
                _sum(
                    totals[steps],
                    f"{prefix}_advantage",
                    ensemble_delta[eligible].mean(),
                    eligible_count,
                )
                _sum(
                    totals[steps],
                    f"{prefix}_best_of_n_advantage",
                    best_delta[eligible].mean(),
                    eligible_count,
                )
                _sum(
                    totals[steps],
                    f"{prefix}_preference",
                    (ensemble_delta[eligible] > 0).float().mean(),
                    eligible_count,
                )
                _sum(
                    totals[steps],
                    f"{prefix}_margin_0.01",
                    (ensemble_delta[eligible] > 0.01).float().mean(),
                    eligible_count,
                )
                _sum(
                    totals[steps],
                    f"{prefix}_output_delta",
                    output_deltas[:, eligible].mean(),
                    eligible_count,
                )
                if component == "history":
                    _sum(
                        totals[steps],
                        "latent/matched_probe_mse",
                        matched_errors.mean(),
                        batch_size,
                    )
        evaluated += batch_size

    if not evaluated:
        raise RuntimeError("no full paired validation batch was available")

    result = {}
    for steps in args.flow_steps:
        metrics = _finish(totals[steps])
        c = counts[steps]
        for name in ("visible", "hidden", "hidden_occupied", "occupied", "global"):
            metrics[f"decoded/{name}_exact"] = c[f"{name}_correct"] / max(
                c[f"{name}_count"], 1.0
            )
            metrics[f"decoded/{name}_count"] = int(c[f"{name}_count"])
        result[str(steps)] = metrics
    return result, evaluated


def gate(metrics):
    """Return auditable, provisional planner-readiness gates."""
    criteria = {
        "latent_mse": (metrics["latent/mse"], 0.60, 1.00, "lower"),
        "visible_exact": (metrics["decoded/visible_exact"], 0.90, 0.80, "higher"),
        "hidden_occupied_exact": (
            metrics["decoded/hidden_occupied_exact"],
            0.25,
            0.10,
            "higher",
        ),
    }
    for component in ("history", "intent", "action"):
        criteria[f"{component}_preference"] = (
            metrics[f"condition/{component}_preference"],
            0.60,
            0.55,
            "higher",
        )
        criteria[f"{component}_advantage"] = (
            metrics[f"condition/{component}_advantage"],
            0.01,
            0.0,
            "higher",
        )
    verdicts = {}
    for name, (value, pass_at, warn_at, direction) in criteria.items():
        if not math.isfinite(value):
            status = "FAIL"
        elif direction == "lower":
            status = "PASS" if value <= pass_at else "WARN" if value <= warn_at else "FAIL"
        else:
            status = "PASS" if value >= pass_at else "WARN" if value >= warn_at else "FAIL"
        verdicts[name] = {
            "status": status,
            "value": value,
            "pass_at": pass_at,
            "warn_at": warn_at,
            "direction": direction,
        }
    overall = "PASS"
    if any(item["status"] == "FAIL" for item in verdicts.values()):
        overall = "FAIL"
    elif any(item["status"] == "WARN" for item in verdicts.values()):
        overall = "WARN"
    return overall, verdicts


def main(argv=None):
    args = parser().parse_args(argv)
    # ``make_loaders`` shares the pretraining CLI contract.
    args.smoke = False
    if args.batch < 2:
        raise SystemExit("--batch must be at least 2 for paired condition shuffles")
    if args.batches <= 0 or args.samples <= 0 or any(x <= 0 for x in args.flow_steps):
        raise SystemExit("batches, samples, and flow steps must be positive")
    seed_everything(args.seed)
    cfg = load_config(args)
    device = resolve_device(args.device)
    model, loader, checkpoint, checkpoint_path, data, tokenizer_cfg = build_model(
        cfg, args, device
    )
    metrics, examples = evaluate(model, loader, device, args)
    promotion_steps = str(model.flow.cfg.sample_steps)
    if promotion_steps not in metrics:
        promotion_steps = str(args.flow_steps[0])
    overall, criteria = gate(metrics[promotion_steps])
    report = {
        "verdict": overall,
        "criteria": criteria,
        "promotion_flow_steps": int(promotion_steps),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "data": str(data),
        "seed": args.seed,
        "batches": args.batches,
        "examples": examples,
        "samples": args.samples,
        "flow_steps": args.flow_steps,
        "tokenizer_cfg": tokenizer_cfg.__dict__,
        "metrics": metrics,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(out.suffix + ".tmp")
        temporary.write_text(encoded + "\n")
        temporary.replace(out)
    return 0 if overall == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
