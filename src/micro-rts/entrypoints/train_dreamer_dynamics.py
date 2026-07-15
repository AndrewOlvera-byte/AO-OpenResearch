"""Entrypoint: DreamerV4 **dynamics** pretraining with a frozen tokenizer.

Phase 2 of the Dreamer 4 recipe: with the tokenizer frozen (loaded from the phase-1
checkpoint), train the block-causal transformer world model on the offline rollouts
— the Dreamer 4 shortcut-forcing flow objective on the latents plus the
arrive-aligned reward / continue heads. Consumes the store via
:func:`build_mrts_loader` (``task="dynamics"`` -> obs + action + reward + cont +
is_first; the heavy mask planes are skipped because the frozen tokenizer's mask
head is not trained here).

Usage (inside the container)::

    python src/micro-rts/entrypoints/train_dreamer_dynamics.py \
        --data '/data/micro-rts/tokdyn_pretrain_v1__*.h5' \
        --tokenizer-ckpt checkpoints/dreamer_tokenizer.pt \
        --steps 30000 --batch 16 --seq-len 16 --num-workers 6 \
        --out checkpoints/dreamer_dynamics.pt

    # tiny CPU smoke (fresh frozen tokenizer, a couple of steps):
    python src/micro-rts/entrypoints/train_dreamer_dynamics.py \
        --data '/data/micro-rts/debug__*.h5' --exp micro-rts/smoke_dreamerv4 --smoke

Without ``--tokenizer-ckpt`` the tokenizer is randomly initialized and frozen
(only meaningful for the smoke path). Model arch comes from ``--exp``; obs/action
shapes come from the dataset attrs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_PKG = _HERE.parents[1]  # src/micro-rts
_SRC = _HERE.parents[2]  # src
for p in (str(_PKG), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from core.config import Config  # noqa: E402
import models.dreamer  # noqa: E402,F401  (registry side effect)
from collectors.offline_data import build_mrts_loader, cycle, to_device  # noqa: E402
from loss.dreamer import cell_weights, dynamics_loss, mask_actions_to_sources  # noqa: E402
from trainers.BaseTrainer import BaseTrainer, resolve_device  # noqa: E402
from trainers.PretrainCheckpointManager import PretrainCheckpointManager  # noqa: E402
from entrypoints.pretrain_common import (  # noqa: E402
    amp_ctx,
    build_model,
    make_adam,
    make_lr_scheduler,
    measure_latent_scale,
    pick as _pick,
    resolve_data,
    setup_backend,
)


def load_pretrained_action_tokenizer(model, ckpt_path, map_location="cpu"):
    """Load the transferable, multi-token action representation into dynamics.

    The SSL-only inverse and forward heads are intentionally discarded.  The
    event encoder and action-slot positional embeddings are the representation
    consumed by the transition model.
    """
    payload = torch.load(ckpt_path, map_location=map_location)
    state = payload.get("model", payload)
    prefix = "action_encoder."
    encoder_state = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not encoder_state:
        raise ValueError(
            f"{ckpt_path} does not contain an action-tokenizer action_encoder"
        )
    model.dynamics.action_encoder.load_state_dict(encoder_state, strict=True)
    if "action_position" not in state:
        raise ValueError(f"{ckpt_path} does not contain action_position")
    if state["action_position"].shape != model.dynamics.action_position.shape:
        raise ValueError(
            "pretrained action-position shape mismatch: "
            f"checkpoint={tuple(state['action_position'].shape)} "
            f"model={tuple(model.dynamics.action_position.shape)}"
        )
    model.dynamics.action_position.data.copy_(state["action_position"])
    if model.dynamics.cfg.pretrained_action_router:
        transfers = {
            "forward_state": "action_router_state",
            "forward_attn": "action_router_attn",
            "forward_norm": "action_router_norm",
            "forward_head": "action_router_head",
        }
        for source, target in transfers.items():
            source_state = {
                key[len(source) + 1 :]: value
                for key, value in state.items()
                if key.startswith(f"{source}.")
            }
            if not source_state:
                raise ValueError(
                    f"{ckpt_path} does not contain pretrained router module {source}"
                )
            getattr(model.dynamics, target).load_state_dict(source_state, strict=True)
    return payload


def _main_structured(args, cfg):
    """Complete-state dynamics on the shared bf16/fused/W&B training harness."""
    from models.dreamer_v2 import (
        StructuredDynamicsConfig,
        StructuredTokenizerConfig,
        StructuredWorldModelV2,
    )
    from models.dreamer_v2.dynamics import (
        structured_action_gap_probe,
        structured_causal_paired_loss,
        structured_dreamer4_loss,
        structured_flow_loss,
    )
    from entrypoints.structured_v2_common import measure_token_stats

    tr, mc = cfg.training or {}, cfg.model or {}
    objective = str(tr.get("objective", "structured_custom"))
    allowed_objectives = ("structured_custom", "dreamer4", "causal_paired")
    if objective not in allowed_objectives:
        raise ValueError(
            f"training.objective must be one of {allowed_objectives}, got {objective!r}"
        )
    steps = _pick(args.steps, tr.get("steps"), 60000)
    batch = _pick(args.batch, tr.get("batch"), 32)
    seq_len = _pick(args.seq_len, tr.get("seq_len"), 1)
    workers = _pick(args.num_workers, tr.get("num_workers"), 8)
    path = resolve_data(_pick(args.data, (cfg.data or {}).get("path"), None))
    tok_path = _pick(args.tokenizer_ckpt, tr.get("tokenizer_ckpt"), None)
    if not tok_path and not args.smoke:
        raise SystemExit("training.tokenizer_ckpt or --tokenizer-ckpt is required")
    device = resolve_device(args.device or cfg.run.get("device", "auto"))
    if args.smoke:
        steps, batch, seq_len, workers, device = (
            2,
            min(batch, 2),
            min(seq_len, 2),
            0,
            torch.device("cpu"),
        )
    setup_backend(device)
    amp = bool(tr.get("amp", True))
    val_frac = 0.0 if args.smoke else float(tr.get("val_frac", 0.0))
    eval_every, eval_batches = (
        int(tr.get("eval_every", 1000)),
        int(tr.get("eval_batches", 8)),
    )
    fixed_val = bool(tr.get("fixed_val", False))
    loader = build_mrts_loader(
        path,
        task=(
            "structured_dynamics_paired"
            if objective == "causal_paired"
            else "structured_dynamics"
        ),
        seq_len=seq_len,
        batch_size=batch,
        num_workers=workers,
        locking=False,
        val_frac=val_frac,
        split="train",
        paired_batch_fraction=(
            tr.get("paired_batch_fraction")
            if objective == "causal_paired"
            else None
        ),
    )
    val_loader = (
        build_mrts_loader(
            path,
            task="structured_dynamics_eval",
            seq_len=seq_len,
            batch_size=batch,
            num_workers=0,
            locking=False,
            val_frac=val_frac,
            split="val",
            drop_last=False,
            shuffle=not fixed_val,
            fixed_chunk_batches=eval_batches if fixed_val else None,
            fixed_chunk_seed=int(tr.get("fixed_val_seed", 0)),
        )
        if val_frac
        else None
    )
    ds = loader.dataset
    ckpt = torch.load(tok_path, map_location="cpu") if tok_path else None
    tc = StructuredTokenizerConfig.from_dict(
        ckpt["tokenizer_cfg"] if ckpt else mc.get("tokenizer")
    )
    tc.mask_width, tc.legacy_obs_channels = 1 + sum(ds.action_nvec[1:]), ds.obs_channels
    dc = StructuredDynamicsConfig.from_dict(mc.get("dynamics"))
    if args.smoke:
        dc.d_model, dc.depth, dc.n_heads, dc.max_action_events, dc.k_max = (
            32,
            2,
            4,
            16,
            4,
        )
        if ckpt is None:
            tc.d_cell, tc.d_latent, tc.depth, tc.n_heads = 16, 16, 1, 4
    model = StructuredWorldModelV2(ds.grid_hw, tc, dc).to(device)
    if ckpt:
        model.tokenizer.load_state_dict(ckpt["model"])
    model.tokenizer.requires_grad_(False)
    model.tokenizer.eval()
    if ckpt and "latent_mean" in ckpt:
        mean, std = ckpt["latent_mean"].to(device), ckpt["latent_std"].to(device)
    else:
        mean, std = measure_token_stats(
            model.tokenizer, loader, device, batches=2 if args.smoke else 32
        )
    model.dynamics.set_latent_stats(mean, std)
    init_from = None if args.smoke else tr.get("init_from")
    action_tokenizer_path = (
        None if args.smoke else tr.get("action_tokenizer_ckpt")
    )
    freeze_action_encoder = bool(
        tr.get("freeze_action_encoder", action_tokenizer_path is not None)
    )
    freeze_action_router = bool(tr.get("freeze_action_router", False))
    if init_from and action_tokenizer_path:
        raise ValueError(
            "training.init_from and training.action_tokenizer_ckpt cannot be "
            "combined; use a clean initialization for the pretraining ablation"
        )
    if action_tokenizer_path:
        if dc.action_encoder_type != "factorized":
            raise ValueError(
                "a pretrained action tokenizer requires "
                "model.dynamics.action_encoder_type=factorized"
            )
        payload = load_pretrained_action_tokenizer(
            model, action_tokenizer_path, map_location="cpu"
        )
        print(
            f"[action-tokenizer] loaded {action_tokenizer_path} "
            f"(source step={payload.get('step', 'unknown')})",
            flush=True,
        )
    if freeze_action_encoder:
        if dc.action_encoder_type != "factorized":
            raise ValueError(
                "training.freeze_action_encoder requires the factorized encoder"
            )
        model.dynamics.action_encoder.requires_grad_(False)
        model.dynamics.action_position.requires_grad_(False)
        model.dynamics.action_encoder.eval()
    router_modules = tuple(
        module
        for module in (
            model.dynamics.action_router_state,
            model.dynamics.action_router_attn,
            model.dynamics.action_router_norm,
            model.dynamics.action_router_head,
        )
        if module is not None
    )
    if freeze_action_router:
        if not router_modules:
            raise ValueError(
                "training.freeze_action_router requires "
                "model.dynamics.pretrained_action_router=true"
            )
        for module in router_modules:
            module.requires_grad_(False)
    lr = float(args.lr if args.lr is not None else tr.get("lr", 1e-4))
    trainable_dynamics = [
        parameter for parameter in model.dynamics.parameters() if parameter.requires_grad
    ]
    router_lr_scale = float(tr.get("action_router_lr_scale", 1.0))
    if router_lr_scale <= 0.0:
        raise ValueError("training.action_router_lr_scale must be positive")
    router_parameters = [
        parameter
        for module in router_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if router_parameters and router_lr_scale != 1.0:
        router_ids = {id(parameter) for parameter in router_parameters}
        optimizer_parameters = [
            {
                "params": [
                    parameter
                    for parameter in trainable_dynamics
                    if id(parameter) not in router_ids
                ]
            },
            {"params": router_parameters, "lr": lr * router_lr_scale},
        ]
    else:
        optimizer_parameters = trainable_dynamics
    opt = make_adam(
        optimizer_parameters,
        lr,
        device,
        weight_decay=float(tr.get("weight_decay", 0.0)),
    )
    sched = make_lr_scheduler(
        opt,
        steps,
        tr.get("warmup_steps", 0),
        tr.get("lr_min_frac", 0.1),
        stages=None if args.smoke else tr.get("lr_stages"),
    )
    trainer = BaseTrainer(cfg, device=str(device))
    trainer.use_wandb = not (args.no_wandb or args.smoke)
    trainer._wandb_key = args.wandb_key
    trainer.init_wandb()
    resume_cfg = bool((tr.get("checkpoint") or {}).get("resume", False))
    if init_from:
        if args.resume or args.resume_from or resume_cfg:
            raise ValueError("training.init_from cannot be combined with resume")
        payload = trainer.load_weights_from(model, init_from)
        print(
            f"[checkpoint] initialized model weights from {init_from} "
            f"(source step={payload.get('step', 'unknown')}); optimizer is fresh",
            flush=True,
        )
    log_every = int(tr.get("log_every", 100))
    probe_flow_steps = int(tr.get("probe_flow_steps", min(4, dc.k_max)))
    probe_samples = int(tr.get("probe_samples", 8))
    self_frac = float(tr.get("flow_self_frac", dc.skip_fraction))

    def loss_fn(batch):
        if objective == "dreamer4":
            return structured_dreamer4_loss(model, batch, self_frac=self_frac)
        if objective == "causal_paired":
            return structured_causal_paired_loss(
                model,
                batch,
                factual_coef=float(tr.get("causal_factual_coef", 1.0)),
                counterfactual_coef=float(tr.get("causal_counterfactual_coef", 1.0)),
                effect_coef=float(tr.get("causal_effect_coef", 1.0)),
                active_token_boost=float(tr.get("active_token_boost", 1.0)),
                changed_token_boost=float(tr.get("changed_token_boost", 4.0)),
                change_threshold=float(tr.get("change_threshold", 1e-6)),
                padding_token_weight=float(tr.get("padding_token_weight", 0.0)),
                effect_cosine_coef=float(tr.get("causal_effect_cosine_coef", 0.0)),
                effect_norm_coef=float(tr.get("causal_effect_norm_coef", 0.0)),
                canonical_grounding_coef=float(
                    tr.get("canonical_grounding_coef", 0.0)
                ),
                canonical_changed_boost=float(
                    tr.get("canonical_changed_boost", 0.0)
                ),
                residual_correction_coef=float(
                    tr.get("residual_correction_coef", 0.0)
                ),
            )
        return structured_flow_loss(
            model,
            batch,
            structured_coef=float(tr.get("structured_coef", 0.2)),
            skip_coef=float(tr.get("skip_coef", 0.25)),
        )

    out = args.out or tr.get("out", "checkpoints/structured_dynamics_v2.pt")
    checkpoints = PretrainCheckpointManager(
        trainer,
        model,
        opt,
        sched,
        metadata={
            "phase": "structured_dynamics_v2",
            "tokenizer_cfg": tc.__dict__,
            "dynamics_cfg": dc.__dict__,
            "grid_hw": ds.grid_hw,
            "action_nvec": ds.action_nvec,
            "objective": objective,
            "init_from": init_from,
            "action_tokenizer_ckpt": action_tokenizer_path,
            "freeze_action_encoder": freeze_action_encoder,
            "freeze_action_router": freeze_action_router,
            "action_router_lr_scale": router_lr_scale,
            "paired_batch_fraction": tr.get("paired_batch_fraction"),
        },
        default_monitor="val/wm/total",
    )
    start_step = checkpoints.load_if_requested(args.resume, args.resume_from)
    amp_name = "bf16" if amp and device.type == "cuda" else "off"
    print(
        f"[dynamics:structured_v2] data={path} device={device} amp={amp_name} "
        f"(SDPA flash attention enabled when supported)"
    )
    init_kind = (
        "restored"
        if start_step
        else "warm-started"
        if init_from
        else "zero-initialized"
    )
    if objective == "dreamer4":
        print(
            f"[dynamics:structured_v2] objective=dreamer4 k_max={dc.k_max} "
            f"empirical_fraction={1.0 - self_frac:.2f} "
            f"bootstrap_fraction={self_frac:.2f} ramp=0.9*tau+0.1 "
            f"bootstrap_x_rescale=true flow_head={init_kind}; "
            f"tokens={model.tokenizer.n_tokens}x2+{dc.max_action_events}+2"
        )
    elif objective == "causal_paired":
        print(
            f"[dynamics:structured_v2] objective=causal_paired query=tau0,d1 "
            f"factual_coef={float(tr.get('causal_factual_coef', 1.0))} "
            f"counterfactual_coef={float(tr.get('causal_counterfactual_coef', 1.0))} "
            f"effect_coef={float(tr.get('causal_effect_coef', 1.0))} "
            f"active_boost={float(tr.get('active_token_boost', 1.0))} "
            f"changed_boost={float(tr.get('changed_token_boost', 4.0))} "
            f"padding_weight={float(tr.get('padding_token_weight', 0.0))} "
            f"correction_coef={float(tr.get('residual_correction_coef', 0.0))} "
            f"initial_noise={dc.initial_noise} "
            f"flow_head={init_kind}; one-step inference; "
            f"residual={dc.residual_prediction} "
            f"pretrained_router={dc.pretrained_action_router} "
            f"spatial_routing={dc.explicit_spatial_action_routing} "
            f"state_padding_mask={dc.mask_empty_entity_tokens} "
            f"paired_batch_fraction={tr.get('paired_batch_fraction')} "
            f"tokens={model.tokenizer.n_tokens}x2+{dc.max_action_events}+2"
        )
    else:
        print(
            f"[dynamics:structured_v2] objective=structured_custom k_max={dc.k_max} "
            f"prior_fraction={dc.prior_fraction} flow_head={init_kind}; "
            f"shortcut forcing: fraction={dc.skip_fraction} "
            f"coef={float(tr.get('skip_coef', 0.25))} shortcut_head={init_kind}; "
            f"tokens={model.tokenizer.n_tokens}x2+{dc.max_action_events}+2"
        )

    @torch.no_grad()
    def run_val(step):
        model.eval()
        sums, counts, n = {}, {}, 0
        rng_devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=rng_devices):
            val_seed = int(tr.get("fixed_val_seed", 0)) if fixed_val else step
            torch.manual_seed(val_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(val_seed)
            for vb in val_loader:
                if n >= eval_batches:
                    break
                vb = to_device(vb, device)
                with amp_ctx(device, amp):
                    _, vm = loss_fn(vb)
                    if n == 0:
                        vm.update(
                            structured_action_gap_probe(
                                model,
                                vb,
                                flow_steps=probe_flow_steps,
                                max_samples=probe_samples,
                            )
                        )
                for key, value in vm.items():
                    name = f"val/{key}"
                    sums[name] = sums.get(name, 0.0) + value
                    counts[name] = counts.get(name, 0) + 1
                n += 1
        vals = {key: value / counts[key] for key, value in sums.items()} if n else {}
        # Aggregate sparse geometry over the complete validation set. Averaging
        # per-batch ratios lets tiny-effect batches create misleading norm spikes.
        rows_key = "val/causal/effect_geometry_rows"
        if rows_key in sums and float(sums[rows_key]) > 0.0:
            vals["val/causal/effect_cosine"] = (
                sums["val/causal/effect_cosine_sum"] / sums[rows_key]
            )
            target_norm_sum = sums["val/causal/effect_target_norm_sum"]
            vals["val/causal/effect_norm_ratio_aggregate"] = (
                sums["val/causal/effect_predicted_norm_sum"]
                / target_norm_sum.clamp_min(1e-8)
            )
        trainer.log(vals, step=step)
        checkpoints.record_eval(step, vals)
        model.train()
        model.tokenizer.eval()
        if freeze_action_encoder:
            model.dynamics.action_encoder.eval()
        if freeze_action_router:
            for module in router_modules:
                module.eval()
        if vals:
            if objective == "dreamer4":
                objective_metrics = (
                    f"mse={vals['val/flow/mse']:.4f} "
                    f"consistency={vals['val/flow/consistency']:.4f}"
                )
            elif objective == "causal_paired":
                objective_metrics = (
                    f"valid={vals['val/causal/valid_mse']:.4f} "
                    f"pad={vals['val/causal/padding_mse']:.4f} "
                    f"cf={vals['val/causal/counterfactual']:.4f} "
                    f"effect={vals['val/causal/effect']:.4f} "
                    f"effect_cos={vals['val/causal/effect_cosine']:.4f} "
                    f"effect_norm={vals['val/causal/effect_norm_ratio']:.3f} "
                    f"effect_norm_agg="
                    f"{vals['val/causal/effect_norm_ratio_aggregate']:.3f}"
                )
            else:
                objective_metrics = (
                    f"prior={vals['val/flow/prior']:.4f} "
                    f"shortcut={vals['val/flow/skip']:.4f}"
                )
            print(
                f"[dynamics:structured_v2] step={step} VAL "
                f"total={vals['val/wm/total']:.4f} "
                f"flow={vals['val/flow/matching']:.4f} {objective_metrics} "
                f"self_gap={vals.get('val/probe/self_gap', float('nan')):.5f} "
                f"opp_gap={vals.get('val/probe/opp_gap', float('nan')):.5f} "
                f"paired_cf={vals.get('val/probe/paired_cf_mse', float('nan')):.5f}"
            )

    model.dynamics.train()
    if freeze_action_encoder:
        model.dynamics.action_encoder.eval()
    if freeze_action_router:
        for module in router_modules:
            module.eval()
    t0 = time.time()
    last_metrics = {}
    average_train_metrics = bool(tr.get("average_train_metrics", False))
    metric_sums, metric_count = {}, 0
    for step, b in zip(range(start_step + 1, steps + 1), cycle(loader)):
        b = to_device(b, device)
        with amp_ctx(device, amp):
            loss, metrics = loss_fn(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            model.dynamics.parameters(), float(tr.get("grad_clip", 10.0))
        )
        opt.step()
        sched.step()
        if average_train_metrics:
            for key, value in metrics.items():
                value = value.detach() if torch.is_tensor(value) else value
                metric_sums[key] = metric_sums.get(key, 0.0) + value
            metric_count += 1
        if step == 1 or step % log_every == 0 or step == steps:
            report = (
                {key: value / metric_count for key, value in metric_sums.items()}
                if average_train_metrics and metric_count
                else metrics
            )
            diag = {
                "wm/grad_norm": float(grad),
                "wm/lr": opt.param_groups[0]["lr"],
                "wm/seq_per_s": step * batch / max(time.time() - t0, 1e-6),
            }
            trainer.log({**report, **diag}, step=step)
            if objective == "dreamer4":
                objective_metrics = (
                    f"mse={float(report['flow/mse']):.4f} "
                    f"consistency={float(report['flow/consistency']):.4f}"
                )
            elif objective == "causal_paired":
                objective_metrics = (
                    f"valid={float(report['causal/valid_mse']):.4f} "
                    f"pad={float(report['causal/padding_mse']):.4f} "
                    f"cf={float(report['causal/counterfactual']):.4f} "
                    f"effect={float(report['causal/effect']):.4f} "
                    f"effect_cos={float(report['causal/effect_cosine']):.4f} "
                    f"effect_norm={float(report['causal/effect_norm_ratio']):.3f} "
                    f"effect_norm_agg="
                    f"{float(report['causal/effect_norm_ratio_aggregate']):.3f}"
                )
            else:
                objective_metrics = (
                    f"prior={float(report['flow/prior']):.4f} "
                    f"skip={float(report['flow/skip']):.4f}"
                )
            print(
                f"[dynamics:structured_v2] step={step}/{steps} "
                f"total={float(report['wm/total']):.4f} "
                f"flow={float(report['flow/matching']):.4f} {objective_metrics} "
                f"({diag['wm/seq_per_s']:.0f} seq/s)",
                flush=True,
            )
            metric_sums, metric_count = {}, 0
        if val_loader is not None and step % eval_every == 0:
            run_val(step)
        last_metrics = metrics
        checkpoints.save_periodic(step, metrics)
    checkpoints.finish(steps, last_metrics, out)
    trainer.finish()
    print(f"[dynamics:structured_v2] saved -> {out}")


def load_tokenizer(model, ckpt_path, device):
    """Load only the ``tokenizer.*`` submodule from a phase-1 checkpoint.

    Returns ``(phase1_step, latent_scale)``; ``latent_scale`` is the latent RMS
    the tokenizer run measured (None for pre-normalization checkpoints).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"]
    prefix = "tokenizer."
    tok_sd = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = model.tokenizer.load_state_dict(tok_sd, strict=False)
    if missing or unexpected:
        print(
            f"[dynamics] tokenizer load: missing={list(missing)} unexpected={list(unexpected)}"
        )
    return ckpt.get("step"), ckpt.get("latent_scale")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="MicroRTS DreamerV4 dynamics pretraining. Everything lives in the "
        "--exp config (training:/data: blocks, incl. training.tokenizer_ckpt); "
        "the flags below are optional overrides. `--set training.batch=32` works too."
    )
    p.add_argument(
        "--exp",
        default="micro-rts/pretrain_dreamerv4_dynamics",
        help="experiment config (holds model arch + training:/data: blocks)",
    )
    # All optional overrides — None means 'take it from the config'.
    p.add_argument(
        "--data", default=None, help="override data.path (.h5 glob; newest used)"
    )
    p.add_argument(
        "--tokenizer-ckpt", default=None, help="override training.tokenizer_ckpt"
    )
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="override model.world_lr")
    p.add_argument("--reward-coef", type=float, default=None)
    p.add_argument("--cont-coef", type=float, default=None)
    p.add_argument("--flow-coef", type=float, default=None)
    p.add_argument("--flow-self-frac", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="held-out trajectory fraction (0 disables validation)",
    )
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None, help="0 = only at the end")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-wandb", action="store_true", help="disable remote W&B logging")
    p.add_argument("--wandb-key", default=None, metavar="KEY", help="W&B API key")
    p.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="resume model/optimizer/scheduler from the configured checkpoint",
    )
    p.add_argument(
        "--resume-from",
        default=None,
        metavar="TAG_OR_PATH",
        help="resume from latest, best, step_N, or a checkpoint path",
    )
    p.add_argument("--set", action="append", default=[], metavar="K=V")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny CPU run (2 steps, no workers, fresh tokenizer)",
    )
    return p.parse_args(argv)


def save_ckpt(path, model, model_cfg, obs_shape, action_nvec, step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "phase": "dynamics",
            "model": model.state_dict(),
            "model_cfg": model_cfg,
            "obs_shape": tuple(obs_shape),
            "action_nvec": list(action_nvec),
            "step": step,
        },
        path,
    )


def main(argv=None):
    args = parse_args(argv)

    # Config is authoritative; CLI flags (and --set / --lr) are overrides.
    cfg = Config.from_experiment(args.exp)
    overrides = list(args.set)
    if args.lr is not None:
        overrides.append(f"model.world_lr={args.lr}")
    cfg.apply_overrides(overrides)
    tr = cfg.training or {}
    if (cfg.model or {}).get("type") == "structured_v2":
        _main_structured(args, cfg)
        return

    steps = _pick(args.steps, tr.get("steps"), 30000)
    batch = _pick(args.batch, tr.get("batch"), 16)
    seq_len = _pick(args.seq_len, tr.get("seq_len"), 16)
    num_workers = _pick(args.num_workers, tr.get("num_workers"), 6)
    reward_coef = _pick(args.reward_coef, tr.get("reward_coef"), 1.0)
    cont_coef = _pick(args.cont_coef, tr.get("cont_coef"), 1.0)
    flow_coef = _pick(args.flow_coef, tr.get("flow_coef"), 1.0)
    flow_self_frac = _pick(args.flow_self_frac, tr.get("flow_self_frac"), 0.25)
    # Optional linear ramp of the self-consistency fraction over training: the
    # distillation signal matters more once the empirical flow is good (probe
    # symptom when it lags: many-step rollouts worse than one-step).
    flow_self_frac_final = tr.get("flow_self_frac_final")
    warmup_steps = _pick(args.warmup_steps, tr.get("warmup_steps"), 0)
    lr_min_frac = tr.get("lr_min_frac", 0.1)
    val_frac = _pick(args.val_frac, tr.get("val_frac"), 0.0)
    eval_every = _pick(args.eval_every, tr.get("eval_every"), 1000)
    eval_batches = tr.get("eval_batches", 8)
    rollout_context = tr.get("rollout_context", 8)
    rollout_samples = tr.get("rollout_samples", 4)
    log_every = _pick(args.log_every, tr.get("log_every"), 50)
    if args.save_every is not None:
        tr.setdefault("checkpoint", {})["every_steps"] = args.save_every
    out = _pick(args.out, tr.get("out"), "checkpoints/dreamer_dynamics.pt")
    tokenizer_ckpt = _pick(args.tokenizer_ckpt, tr.get("tokenizer_ckpt"), None)
    data_glob = _pick(args.data, (cfg.data or {}).get("path"), None)
    if not data_glob:
        raise SystemExit("no dataset: set data.path in the config or pass --data")

    device_spec = args.device or cfg.run.get("device", "auto")
    if args.smoke:
        steps, num_workers = min(steps, 2), 0
        batch, seq_len = min(batch, 4), min(seq_len, 8)
        val_frac = 0.0
        device_spec = args.device or "cpu"
        tokenizer_ckpt = None  # smoke uses a fresh random frozen tokenizer
    device = resolve_device(device_spec)
    setup_backend(device)
    # Fast by default on CUDA; set training.amp=false only for debugging.
    amp = bool(tr.get("amp", True))
    path = resolve_data(data_glob)

    loader = build_mrts_loader(
        path,
        task="dynamics",
        seq_len=seq_len,
        batch_size=batch,
        num_workers=num_workers,
        locking=False,
        val_frac=val_frac,
        split="train",
    )
    val_loader = None
    if val_frac > 0.0:
        val_loader = build_mrts_loader(
            path,
            task="dynamics",
            seq_len=seq_len,
            batch_size=batch,
            num_workers=0,
            locking=False,
            val_frac=val_frac,
            split="val",
            drop_last=False,
        )
    ds = loader.dataset
    print(f"[dynamics] data={path}")
    print(
        f"[dynamics] windows={len(ds)} "
        f"val_windows={len(val_loader.dataset) if val_loader else 0} "
        f"obs_shape={ds.obs_shape} nvec={ds.action_nvec} device={device}"
    )
    if not getattr(ds, "has_terminal_obs", False):
        print(
            "[dynamics] WARNING: store has no terminal frames — every cont=0 "
            "target sits on a masked reset slot, so the continue head and the "
            "terminal (win) reward CANNOT be learned from this dataset. "
            "wm/continue==0 means 'no signal', not 'converged'."
        )

    model, model_cfg = build_model(cfg, ds.obs_shape, ds.action_nvec, device)

    latent_scale = None
    if tokenizer_ckpt:
        step, latent_scale = load_tokenizer(model, tokenizer_ckpt, device)
        print(
            f"[dynamics] loaded frozen tokenizer from {tokenizer_ckpt} (phase-1 step={step})"
        )
    else:
        print("[dynamics] WARNING: no tokenizer_ckpt; using a random frozen tokenizer")
    model.tokenizer.requires_grad_(
        False
    )  # frozen: dynamics phase only trains the world model

    # The flow objective runs in unit-RMS latent space; the scale comes from the
    # phase-1 checkpoint, or is measured over the data for older checkpoints.
    if latent_scale is None:
        latent_scale = measure_latent_scale(
            model.tokenizer, loader, device, batches=4 if args.smoke else 32
        )
        print(
            f"[dynamics] no latent_scale in checkpoint; measured RMS = {latent_scale:.4f}"
        )
    model.world_model.set_latent_scale(latent_scale)
    print(f"[dynamics] latent_scale = {float(model.world_model.latent_scale):.4f}")

    dyn_cfg = model.cfg.dynamics
    cell_occ_boost = dyn_cfg.cell_occ_boost
    cell_changed_boost = dyn_cfg.cell_changed_boost
    cell_weight_floor = dyn_cfg.cell_weight_floor
    if (cell_occ_boost, cell_changed_boost, cell_weight_floor) != (1.0, 1.0, 1.0):
        print(
            f"[dynamics] cell-weighted flow loss: occ={cell_occ_boost} "
            f"changed={cell_changed_boost} floor={cell_weight_floor}"
        )
    print(
        f"[dynamics] amp={'bf16' if amp else 'off'} (bf16 = SDPA flash-attention path)"
    )
    mask_junk = bool(getattr(dyn_cfg, "mask_junk_actions", False))
    if mask_junk:
        print(
            "[dynamics] mask_junk_actions=on: self-actions NOOPed at cells "
            "with no idle own unit before the action encoder"
        )
    has_opp_head = getattr(model.world_model, "opp_head", None) is not None
    opp_bc_coef = float(tr.get("opp_bc_coef", 1.0 if has_opp_head else 0.0))
    if has_opp_head:
        print(
            f"[dynamics] opponent-policy head: BC on executed opponent actions "
            f"at source cells, coef={opp_bc_coef} (imagination samples this "
            f"head instead of the unknown_opp marginal)"
        )
    elif opp_bc_coef > 0.0:
        print(
            "[dynamics] WARNING: opp_bc_coef set but model.dynamics.opp_head "
            "is off — no opponent BC will be trained"
        )

    opt = make_adam(model.world_model.parameters(), model.cfg.world_lr, device)
    sched = make_lr_scheduler(opt, steps, warmup_steps, lr_min_frac)
    clip = model.cfg.grad_clip

    # Reuse BaseTrainer's W&B machinery (key resolution / init / log / finish).
    trainer = BaseTrainer(cfg, device=str(device))
    if args.no_wandb or args.smoke:
        trainer.use_wandb = False
    if args.wandb_key:
        trainer._wandb_key = args.wandb_key
    trainer.init_wandb()
    checkpoints = PretrainCheckpointManager(
        trainer,
        model,
        opt,
        sched,
        metadata={
            "phase": "dynamics",
            "model_cfg": model_cfg,
            "obs_shape": tuple(ds.obs_shape),
            "action_nvec": list(ds.action_nvec),
            "latent_scale": latent_scale,
        },
        default_monitor="val/mse",
    )
    start_step = checkpoints.load_if_requested(args.resume, args.resume_from)

    wm = model.world_model

    @torch.no_grad()
    def run_val(step):
        """Held-out dynamics loss + open-loop rollout probe.

        The rollout probe is the training-path verification signal: encode a
        held-out context, generate the remaining frames autoregressively with
        the real actions at few and many flow steps, and compare (normalized)
        latents against the encoder's. ``copylast`` is the no-dynamics baseline
        the model must beat to be predicting motion at all.
        """
        model.eval()  # val measures the conditional model: no opponent dropout
        vals, n = {}, 0
        for b in val_loader:
            if n >= eval_batches:
                break
            b = to_device(b, device)
            if mask_junk:
                b["action"] = mask_actions_to_sources(b["action"], b["obs"])
            z = model.tokenizer.encode(b["obs"])
            cw = cell_weights(
                b["obs"],
                occ_boost=cell_occ_boost,
                changed_boost=cell_changed_boost,
                floor=cell_weight_floor,
                downsample=model.tokenizer.cfg.downsample,
            )
            _, m = dynamics_loss(
                model,
                z,
                b["action"],
                b["reward"],
                b["cont"],
                b["is_first"],
                opponent_action=b["opponent_action"],
                cell_weight=cw,
                obs=b["obs"],
                reward_coef=reward_coef,
                cont_coef=cont_coef,
                flow_coef=flow_coef,
                opp_bc_coef=opp_bc_coef,
                self_frac=flow_self_frac,
            )
            for k in ("flow/mse", "wm/reward", "wm/continue", "wm/total"):
                vals[f"val/{k.split('/')[1]}"] = (
                    vals.get(f"val/{k.split('/')[1]}", 0.0) + m[k]
                )
            for k in ("opp_bc/loss", "opp_bc/type_acc"):
                if k in m:
                    kk = f"val/opp_bc_{k.split('/')[1]}"
                    vals[kk] = vals.get(kk, 0.0) + m[k]
            n += 1
        if not n:
            return
        vals = {k: v / n for k, v in vals.items()}

        b = to_device(next(iter(val_loader)), device)
        b = {k: v[:8] for k, v in b.items()}  # bound the probe's cost
        if mask_junk:
            b["action"] = mask_actions_to_sources(b["action"], b["obs"])
        z = model.tokenizer.encode(b["obs"])
        ctx = min(rollout_context, z.shape[1] - 1)
        z_tgt = wm.normalize(z[:, ctx:])
        obs_tgt = b["obs"][:, ctx:]
        copylast = wm.normalize(z[:, ctx - 1 : -1])
        vals["rollout/copylast_mse"] = float((copylast - z_tgt).pow(2).mean())

        # Best-of-N sampling: the opponent's actions are unobserved, so the next
        # frame is genuinely stochastic — a single sample's MSE vs the realized
        # frame has an irreducible floor. min over samples separates "can the
        # model reach the realized future at all" from that sampling entropy.
        probe_steps = sorted({1, min(4, wm.k_max), wm.k_max})
        pred_many = None
        for s in probe_steps:
            per_seq = None
            for _ in range(max(1, rollout_samples)):
                pred = model.open_loop(
                    z,
                    b["action"],
                    b["is_first"],
                    context=ctx,
                    flow_steps=s,
                    opponent_action=b["opponent_action"],
                )
                m = (wm.normalize(pred) - z_tgt).pow(2).mean(dim=(1, 2, 3))
                per_seq = m if per_seq is None else torch.minimum(per_seq, m)
                if s == probe_steps[-1]:
                    pred_many = pred  # one many-step sample kept below
            vals[f"rollout/mse_k{s}"] = float(per_seq.mean())
        vals["rollout/few_vs_many"] = (
            vals[f"rollout/mse_k{probe_steps[0]}"]
            - vals[f"rollout/mse_k{probe_steps[-1]}"]
        )

        # Counterfactual action probe (NEXT_PLAN.md gate 1 — THE go/no-go number):
        # open-loop MSE with true actions vs batch-shuffled self / opponent / both
        # action streams. If shuffling does not hurt, the model ignores that
        # channel and imagination training is pointless. `probes.py` has the
        # shared implementation; gaps must be > 0 and grow with horizon.
        from entrypoints.probes import counterfactual_action_probe

        vals.update(
            counterfactual_action_probe(
                model,
                z,
                b["action"],
                b["opponent_action"],
                b["is_first"],
                context=ctx,
                flow_steps=min(4, wm.k_max),
            )
        )

        # Decoded-obs binary-plane accuracy — interpretable "does the board look
        # right" signal — against the copy-last baseline in the same space.
        recon = model.tokenizer.decode(pred_many)
        vals["rollout/obs_acc"] = float(
            ((recon > 0.5) == (obs_tgt > 0.5)).float().mean()
        )
        vals["rollout/copylast_obs_acc"] = float(
            ((b["obs"][:, ctx - 1 : -1] > 0.5) == (obs_tgt > 0.5)).float().mean()
        )

        # Reward-head correlation along the generated latents (real actions).
        z_full = torch.cat([z[:, :ctx], pred_many], dim=1)
        c = wm.contextualize(
            z_full, b["action"], b["is_first"], opponent_action=b["opponent_action"]
        )
        pr = c["reward"][:, ctx:].flatten().float()
        tr_true = b["reward"][:, ctx - 1 : -1].flatten().float()
        if pr.std() > 1e-6 and tr_true.std() > 1e-6:
            vals["rollout/reward_corr"] = float(
                torch.corrcoef(torch.stack([pr, tr_true]))[0, 1]
            )

        trainer.log(vals, step=step)
        checkpoints.record_eval(step, vals)
        print(
            f"[dyn] step {step:>7d}  VAL flow_mse={vals['val/mse']:.5f}  "
            f"reward={vals['val/reward']:.5f}  "
            f"rollout mse k{probe_steps[0]}={vals[f'rollout/mse_k{probe_steps[0]}']:.4f} "
            f"k{probe_steps[-1]}={vals[f'rollout/mse_k{probe_steps[-1]}']:.4f} "
            f"copylast={vals['rollout/copylast_mse']:.4f}  "
            f"obs_acc={vals['rollout/obs_acc']:.4f} "
            f"(copy {vals['rollout/copylast_obs_acc']:.4f})  "
            f"rcorr={vals.get('rollout/reward_corr', float('nan')):.3f}"
            + (
                f"  oppBC={vals['val/opp_bc_loss']:.4f} "
                f"(type_acc={vals['val/opp_bc_type_acc']:.3f})"
                if "val/opp_bc_loss" in vals
                else ""
            )
        )
        print(
            f"[dyn] step {step:>7d}  CF-PROBE "
            f"ISSUED self={vals.get('probe/self_gap_issued', float('nan')):.4f} "
            f"opp={vals.get('probe/opp_gap_issued', float('nan')):.4f} "
            f"(primary gate; aggregate: self {vals.get('probe/self_gap', float('nan')):.4f} "
            f"opp {vals.get('probe/opp_gap', float('nan')):.4f}, "
            f"growth {vals.get('probe/self_gap_growth', float('nan')):.4f}/"
            f"{vals.get('probe/opp_gap_growth', float('nan')):.4f}) "
            f"— issued gaps must be > 0 and grow before any RL"
        )
        model.train()
        model.tokenizer.eval()

    model.train()
    model.tokenizer.eval()
    t0 = time.time()
    reward_ema = None
    last_metrics = {}
    for step, batch_data in zip(range(start_step + 1, steps + 1), cycle(loader)):
        batch_data = to_device(batch_data, device)
        if mask_junk:
            batch_data["action"] = mask_actions_to_sources(
                batch_data["action"], batch_data["obs"]
            )
        with torch.no_grad(), amp_ctx(device, amp):
            z = model.tokenizer.encode(
                batch_data["obs"]
            ).float()  # frozen tokenizer latents
        if flow_self_frac_final is not None:
            self_frac = flow_self_frac + (flow_self_frac_final - flow_self_frac) * (
                step / steps
            )
        else:
            self_frac = flow_self_frac
        cw = cell_weights(
            batch_data["obs"],
            occ_boost=cell_occ_boost,
            changed_boost=cell_changed_boost,
            floor=cell_weight_floor,
            downsample=model.tokenizer.cfg.downsample,
        )
        with amp_ctx(device, amp):
            loss, metrics = dynamics_loss(
                model,
                z,
                batch_data["action"],
                batch_data["reward"],
                batch_data["cont"],
                batch_data["is_first"],
                opponent_action=batch_data["opponent_action"],
                cell_weight=cw,
                obs=batch_data["obs"],
                reward_coef=reward_coef,
                cont_coef=cont_coef,
                flow_coef=flow_coef,
                opp_bc_coef=opp_bc_coef,
                self_frac=self_frac,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.world_model.parameters(), clip)
        opt.step()
        sched.step()
        r = metrics["wm/reward"]
        reward_ema = r if reward_ema is None else 0.99 * reward_ema + 0.01 * r
        if step % log_every == 0 or step == 1 or step == steps:
            sps = step * batch / max(time.time() - t0, 1e-6)
            diag = {
                "wm/grad_norm": float(grad_norm),
                "wm/lr": opt.param_groups[0]["lr"],
                "wm/seq_per_s": sps,
                "wm/reward_ema": reward_ema,
                "wm/self_frac": self_frac,
            }
            trainer.log({**metrics, **diag}, step=step)
            opp_bc = (
                f"  oppBC={metrics['opp_bc/loss']:.4f} "
                f"(type_acc={metrics['opp_bc/type_acc']:.3f})"
                if "opp_bc/loss" in metrics
                else ""
            )
            print(
                f"[dyn] step {step:>7d}  flow={metrics['flow/total']:.5f}  "
                f"(mse={metrics['flow/mse']:.5f} sc={metrics['flow/consistency']:.5f})  "
                f"reward={reward_ema:.5f}  continue={metrics['wm/continue']:.5f}  "
                f"total={metrics['wm/total']:.5f}{opp_bc}  "
                f"gnorm={diag['wm/grad_norm']:.2f}  ({sps:.0f} seq/s)"
            )
        if val_loader is not None and step % eval_every == 0:
            run_val(step)
        last_metrics = metrics
        checkpoints.save_periodic(step, metrics)

    checkpoints.finish(steps, last_metrics, out)
    trainer.finish()
    print(f"[dynamics] saved -> {out}")


if __name__ == "__main__":
    main()
