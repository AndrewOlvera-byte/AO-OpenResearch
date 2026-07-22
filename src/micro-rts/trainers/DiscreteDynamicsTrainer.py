"""Registered trainer for causal dynamics over frozen discrete state codes."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
for path in (HERE.parents[1], HERE.parents[2]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from collectors.offline_data import build_mrts_loader, cycle, to_device  # noqa: E402
from core.config import Config  # noqa: E402
from core.registry import register  # noqa: E402
from entrypoints.pretrain_common import (  # noqa: E402
    amp_ctx,
    make_adam,
    make_lr_scheduler,
    pick,
    resolve_data,
    setup_backend,
)
from models.dreamer_v2 import (  # noqa: E402
    DiscreteDynamicsConfig,
    DiscreteStructuredWorldModel,
    DiscreteTokenizerConfig,
    discrete_causal_paired_loss,
    discrete_prior_geometry,
    load_discrete_action_jepa,
)
from trainers.BaseTrainer import BaseTrainer, resolve_device  # noqa: E402
from trainers.PretrainCheckpointManager import PretrainCheckpointManager  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp",
        default="micro-rts/dynamics/discrete_v3/pretrain_discrete_causal_transformer_v3",
    )
    parser.add_argument("--data", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-key", default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="K=V")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


@torch.no_grad()
def _audit_step_zero_geometry(model, loader, device, training_cfg):
    """Certify the transferred action prior on real paired rows."""
    count = int(training_cfg.get("preflight_batches", 4))
    if count <= 0:
        return {}
    iterator = iter(loader)
    batches = []
    for _ in range(count):
        try:
            batches.append(next(iterator))
        except StopIteration:
            break
    if not batches:
        raise RuntimeError("step-zero geometry preflight received no data")
    joined = {
        key: torch.cat([batch[key] for batch in batches], dim=0).to(device)
        for key in batches[0]
    }
    model.eval()
    metrics = discrete_prior_geometry(model, joined)
    values = {key: float(value) for key, value in metrics.items()}
    print(
        "[discrete-dynamics] STEP-ZERO geometry "
        f"rows={int(values['geometry/paired_rows'])} "
        f"effect_codes={int(values['geometry/paired_effect_codes'])} "
        f"code={values['geometry/code_acc']:.4f} "
        f"copy={values['geometry/copy_code_acc']:.4f} "
        f"changed={values['geometry/changed_code_acc']:.4f} "
        f"factual_pref={values['geometry/factual_preference']:.4f} "
        f"cf_pref={values['geometry/counterfactual_preference']:.4f} "
        f"both_pref={values['geometry/bidirectional_preference']:.4f} "
        f"margin={values['geometry/preference_margin']:.4f} "
        f"branch_disagree={values['geometry/branch_disagreement']:.4f} "
        f"overflow={values['geometry/action_overflow']:.6f}"
    )

    min_effects = int(training_cfg.get("preflight_min_effect_codes", 1))
    min_preference = float(
        training_cfg.get("preflight_min_bidirectional_preference", 0.5)
    )
    max_overflow = float(training_cfg.get("preflight_max_action_overflow", 0.0))
    failures = []
    if values["geometry/paired_effect_codes"] < min_effects:
        failures.append(
            f"paired effect codes {int(values['geometry/paired_effect_codes'])} "
            f"< {min_effects}"
        )
    if values["geometry/bidirectional_preference"] < min_preference:
        failures.append(
            "bidirectional paired preference "
            f"{values['geometry/bidirectional_preference']:.4f} < {min_preference:.4f}"
        )
    if values["geometry/action_overflow"] > max_overflow:
        failures.append(
            f"action overflow {values['geometry/action_overflow']:.6f} "
            f"> {max_overflow:.6f}"
        )
    if failures:
        raise RuntimeError(
            "step-zero categorical geometry failed: " + "; ".join(failures)
        )
    return metrics


def _run(cfg, args):
    torch.manual_seed(int(cfg.run.get("seed", 0)))
    tr, mc = cfg.training or {}, cfg.model or {}
    steps = pick(args.steps, tr.get("steps"), 160000)
    batch = pick(args.batch, tr.get("batch"), 32)
    workers = pick(args.num_workers, tr.get("num_workers"), 8)
    data = resolve_data(pick(args.data, (cfg.data or {}).get("path"), None))
    device = resolve_device(args.device or cfg.run.get("device", "auto"))
    if args.smoke:
        steps, batch, workers, device = 2, min(batch, 2), 0, torch.device("cpu")
    setup_backend(device)
    amp = bool(tr.get("amp", True)) and not args.smoke
    val_frac = 0.0 if args.smoke else float(tr.get("val_frac", 0.05))
    common = dict(
        path=data, seq_len=1, batch_size=batch, locking=False, val_frac=val_frac
    )
    loader = build_mrts_loader(
        **common,
        task="structured_dynamics_paired",
        num_workers=workers,
        split="train",
        paired_batch_fraction=tr.get("paired_batch_fraction", 0.5),
    )
    val_loader = (
        build_mrts_loader(
            **common,
            task="structured_dynamics_eval",
            num_workers=0,
            split="val",
            drop_last=False,
            shuffle=False,
            fixed_chunk_batches=int(tr.get("eval_batches", 16)),
            fixed_chunk_seed=int(tr.get("fixed_val_seed", 0)),
        )
        if val_frac
        else None
    )
    tokenizer_path = tr.get("tokenizer_ckpt")
    if not tokenizer_path:
        raise SystemExit("training.tokenizer_ckpt is required")
    tokenizer_payload = torch.load(tokenizer_path, map_location="cpu")
    tc = DiscreteTokenizerConfig.from_dict(tokenizer_payload["discrete_tokenizer_cfg"])
    dc = DiscreteDynamicsConfig.from_dict(mc.get("dynamics"))
    init_from = None if args.smoke else tr.get("init_from")
    if args.smoke:
        dc.d_model, dc.depth, dc.n_heads = 64, 1, 4
        dc.action_field_dim, dc.max_action_events = 8, 8
        dc.pretrained_action_router = False
    model = DiscreteStructuredWorldModel(loader.dataset.grid_hw, tc, dc).to(device)
    model.tokenizer.load_state_dict(tokenizer_payload["model"])
    model.tokenizer.requires_grad_(False)
    model.tokenizer.eval()
    action_jepa_path = None if args.smoke else tr.get("action_jepa_ckpt")
    if init_from:
        init_payload = torch.load(init_from, map_location="cpu")
        model.load_state_dict(init_payload["model"])
        print(
            f"[discrete-dynamics] initialized model weights from {init_from} "
            f"step={init_payload.get('step', 'unknown')}"
        )
    elif action_jepa_path:
        action_payload = torch.load(action_jepa_path, map_location="cpu")
        load_discrete_action_jepa(model, action_payload)
        print(
            f"[discrete-dynamics] loaded action-JEPA {action_jepa_path} "
            f"step={action_payload.get('step', 'unknown')}"
        )
    freeze_action = bool(
        tr.get("freeze_action_interface", action_jepa_path is not None)
    )
    if freeze_action:
        model.dynamics.action_encoder.requires_grad_(False)
        model.dynamics.action_position.requires_grad_(False)
        if model.dynamics.action_router is not None:
            model.dynamics.action_router.requires_grad_(False)
    has_action_prior = bool(init_from or action_jepa_path)
    step_zero_geometry = (
        _audit_step_zero_geometry(model, loader, device, tr) if has_action_prior else {}
    )
    parameters = [item for item in model.dynamics.parameters() if item.requires_grad]
    lr = float(args.lr if args.lr is not None else tr.get("lr", 1e-4))
    action_lr_scale = float(tr.get("action_interface_lr_scale", 1.0))
    if action_lr_scale <= 0.0:
        raise ValueError("training.action_interface_lr_scale must be positive")
    action_parameters = []
    if not freeze_action:
        action_parameters.extend(model.dynamics.action_encoder.parameters())
        action_parameters.append(model.dynamics.action_position)
        if model.dynamics.action_router is not None:
            action_parameters.extend(model.dynamics.action_router.parameters())
    action_ids = {id(item) for item in action_parameters}
    dynamics_parameters = [item for item in parameters if id(item) not in action_ids]
    optimizer_parameters = (
        [
            {"params": dynamics_parameters, "lr": lr},
            {"params": action_parameters, "lr": lr * action_lr_scale},
        ]
        if action_parameters
        else parameters
    )
    opt = make_adam(
        optimizer_parameters,
        lr,
        device,
        weight_decay=float(tr.get("weight_decay", 1e-4)),
    )
    sched = make_lr_scheduler(
        opt,
        steps,
        tr.get("warmup_steps", 1000),
        tr.get("lr_min_frac", 0.025),
        stages=tr.get("lr_stages"),
    )
    trainer = BaseTrainer(cfg, device=str(device))
    trainer.use_wandb = not (args.no_wandb or args.smoke)
    trainer._wandb_key = args.wandb_key
    trainer.init_wandb()
    if step_zero_geometry:
        trainer.log(step_zero_geometry, step=0)
    out = args.out or tr.get("out", "checkpoints/discrete_causal_transformer_v3.pt")
    checkpoints = PretrainCheckpointManager(
        trainer,
        model,
        opt,
        sched,
        metadata={
            "phase": "discrete_causal_transformer_v3",
            "discrete_tokenizer_cfg": tc.__dict__,
            "discrete_dynamics_cfg": dc.__dict__,
            "grid_hw": loader.dataset.grid_hw,
            "tokenizer_ckpt": tokenizer_path,
            "action_jepa_ckpt": action_jepa_path,
            "init_from": init_from,
            "freeze_action_interface": freeze_action,
            "action_interface_lr_scale": action_lr_scale,
            "step_zero_geometry": {
                key: float(value) for key, value in step_zero_geometry.items()
            },
        },
        default_monitor="val/dynamics/total",
    )
    start = checkpoints.load_if_requested(args.resume, args.resume_from)

    def loss_fn(value):
        return discrete_causal_paired_loss(
            model,
            value,
            factual_coef=float(tr.get("factual_coef", 1.0)),
            counterfactual_coef=float(tr.get("counterfactual_coef", 1.0)),
            effect_margin_coef=float(tr.get("effect_margin_coef", 2.0)),
            effect_margin=float(tr.get("effect_margin", 1.0)),
        )

    @torch.no_grad()
    def run_val(step):
        model.eval()
        sums, count = {}, 0
        for value in val_loader:
            value = to_device(value, device)
            with amp_ctx(device, amp):
                _, metrics = loss_fn(value)
            for key, item in metrics.items():
                name = f"val/{key}"
                sums[name] = sums.get(name, 0.0) + item
            count += 1
        values = {key: item / count for key, item in sums.items()}
        trainer.log(values, step=step)
        checkpoints.record_eval(step, values)
        model.train()
        model.tokenizer.eval()
        if freeze_action:
            model.dynamics.action_encoder.eval()
            if model.dynamics.action_router is not None:
                model.dynamics.action_router.eval()
        print(
            f"[discrete-dynamics] step={step} VAL "
            f"loss={values['val/dynamics/total']:.4f} "
            f"code={values['val/dynamics/code_acc']:.4f} "
            f"changed={values['val/dynamics/changed_code_acc']:.4f} "
            f"cell={values['val/dynamics/exact_cell_teacher_forced']:.4f}"
        )

    sequence_length = 2 * model.tokenizer.n_code_tokens + dc.max_action_events + 1
    print(
        f"[discrete-dynamics] data={data} device={device} "
        f"state_codes={model.tokenizer.n_code_tokens} "
        f"actions={dc.max_action_events} sequence={sequence_length} flow=off"
    )
    model.train()
    model.tokenizer.eval()
    if freeze_action:
        model.dynamics.action_encoder.eval()
        if model.dynamics.action_router is not None:
            model.dynamics.action_router.eval()
    t0, last = time.time(), {}
    for step, value in zip(range(start + 1, steps + 1), cycle(loader)):
        value = to_device(value, device)
        with amp_ctx(device, amp):
            loss, metrics = loss_fn(value)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            parameters, float(tr.get("grad_clip", 5.0))
        )
        opt.step()
        sched.step()
        if step == 1 or step % int(tr.get("log_every", 100)) == 0 or step == steps:
            diag = {
                "dynamics/grad_norm": float(grad),
                "dynamics/lr": opt.param_groups[0]["lr"],
                "dynamics/seq_per_s": step * batch / max(time.time() - t0, 1e-6),
            }
            trainer.log({**metrics, **diag}, step=step)
            print(
                f"[discrete-dynamics] step={step}/{steps} "
                f"loss={float(metrics['dynamics/total']):.4f} "
                f"code={float(metrics['dynamics/code_acc']):.4f} "
                f"changed={float(metrics['dynamics/changed_code_acc']):.4f} "
                f"effect={float(metrics['dynamics/effect_margin']):.4f}"
            )
        if val_loader is not None and step % int(tr.get("eval_every", 1000)) == 0:
            run_val(step)
        last = metrics
        checkpoints.save_periodic(step, metrics)
    checkpoints.finish(steps, last, out)
    trainer.finish()
    print(f"[discrete-dynamics] saved -> {out}")


@register("trainer", "discrete_v3_dynamics")
class DiscreteDynamicsTrainer:
    def __init__(self, cfg, args, **_):
        self.cfg, self.args = cfg, args

    def train(self):
        return _run(self.cfg, self.args)

    smoke_test = train


def main(argv=None):
    args = parse_args(argv)
    cfg = Config.from_experiment(args.exp)
    cfg.apply_overrides(args.set)
    trainer = DiscreteDynamicsTrainer(cfg, args)
    return trainer.smoke_test() if args.smoke else trainer.train()


if __name__ == "__main__":
    main()
