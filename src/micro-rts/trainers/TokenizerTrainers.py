"""Registered DreamerV4 and structured-v2 tokenizer trainers.

Phase 1 of the Dreamer 4 recipe: train just the grid autoencoder (encoder/decoder
+ the predicted action-mask head) to reconstruct obs, with everything else left
untouched. Consumes the offline HDF5 store collected by ``collect_mrts_data`` via
the multi-worker :func:`build_mrts_loader` (``task="tokenizer"`` -> obs + mask
only). The checkpoint it writes is the frozen tokenizer that
``train_dreamer_dynamics`` loads.

Usage (inside the container)::

    python src/micro-rts/entrypoints/train_dreamer_tokenizer.py \
        --data '/data/micro-rts/tokdyn_pretrain_v1__*.h5' \
        --steps 20000 --batch 32 --seq-len 16 --num-workers 6 \
        --out checkpoints/dreamer_tokenizer.pt

    # tiny CPU smoke (a couple of steps, no workers):
    python src/micro-rts/entrypoints/train_dreamer_tokenizer.py \
        --data '/data/micro-rts/debug__*.h5' --exp micro-rts/rl/smoke/smoke_dreamerv4 --smoke

The model architecture comes from ``--exp`` (its ``model:`` block); obs/action
shapes come from the dataset's stored attrs. ``--set K=V`` applies dotted-path
overrides to the config just like the other entrypoints.
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
from core.registry import build, register  # noqa: E402
import models.dreamer  # noqa: E402,F401  (registry side effect)
from collectors.offline_data import build_mrts_loader, cycle, to_device  # noqa: E402
from loss.dreamer import cell_weights, tokenizer_loss  # noqa: E402
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


def _main_structured(args, cfg):
    """Complete-state tokenizer using the same accelerated training harness."""
    from models.dreamer_v2 import (
        StructuredTokenizer,
        StructuredTokenizerConfig,
        TemporalJEPAConfig,
        TemporalJEPATokenizerPretrainer,
        structured_temporal_jepa_loss,
    )
    from models.dreamer_v2.tokenizer import structured_reconstruction_loss
    from entrypoints.structured_v2_common import measure_token_stats

    tr, mc = cfg.training or {}, cfg.model or {}
    objective = str(tr.get("objective", "reconstruction"))
    if objective not in ("reconstruction", "temporal_jepa"):
        raise ValueError(
            "structured tokenizer training.objective must be reconstruction or "
            f"temporal_jepa, got {objective!r}"
        )
    steps = _pick(args.steps, tr.get("steps"), 50000)
    batch = _pick(args.batch, tr.get("batch"), 64)
    seq_len = _pick(args.seq_len, tr.get("seq_len"), 1)
    workers = _pick(args.num_workers, tr.get("num_workers"), 8)
    path = resolve_data(_pick(args.data, (cfg.data or {}).get("path"), None))
    device = resolve_device(args.device or cfg.run.get("device", "auto"))
    if args.smoke:
        steps, batch, seq_len, workers, device = (
            2,
            min(batch, 4),
            1,
            0,
            torch.device("cpu"),
        )
    setup_backend(device)
    amp = bool(tr.get("amp", True))
    val_frac = 0.0 if args.smoke else float(tr.get("val_frac", 0.0))
    transition_data = objective == "temporal_jepa" or tr.get(
        "paired_batch_fraction"
    ) is not None
    loader = build_mrts_loader(
        path,
        task=(
            "structured_tokenizer_jepa"
            if transition_data
            else "structured_tokenizer"
        ),
        seq_len=seq_len,
        batch_size=batch,
        num_workers=workers,
        locking=False,
        val_frac=val_frac,
        split="train",
        paired_batch_fraction=(
            tr.get("paired_batch_fraction")
            if objective == "temporal_jepa"
            else None
        ),
    )
    val_loader = (
        build_mrts_loader(
            path,
            task=(
                "structured_tokenizer_jepa"
                if transition_data
                else "structured_tokenizer"
            ),
            seq_len=seq_len,
            batch_size=batch,
            num_workers=0,
            locking=False,
            val_frac=val_frac,
            split="val",
            drop_last=False,
            shuffle=not bool(tr.get("fixed_val", True)),
            fixed_chunk_batches=(
                int(tr.get("eval_batches", 8))
                if bool(tr.get("fixed_val", True))
                else None
            ),
            fixed_chunk_seed=int(tr.get("fixed_val_seed", 0)),
        )
        if val_frac
        else None
    )
    ds = loader.dataset
    tc = StructuredTokenizerConfig.from_dict(mc.get("tokenizer"))
    tc.mask_width, tc.legacy_obs_channels = 1 + sum(ds.action_nvec[1:]), ds.obs_channels
    if args.smoke:
        tc.d_cell, tc.d_latent, tc.depth, tc.n_heads = 16, 16, 1, 4
    jepa_cfg = TemporalJEPAConfig.from_dict(mc.get("temporal_jepa"))
    if args.smoke:
        jepa_cfg.d_model = 32
        jepa_cfg.field_dim = 8
        jepa_cfg.n_heads = 4
        jepa_cfg.max_action_events = min(jepa_cfg.max_action_events, 8)
    model = (
        TemporalJEPATokenizerPretrainer(ds.grid_hw, tc, jepa_cfg)
        if objective == "temporal_jepa"
        else StructuredTokenizer(ds.grid_hw, tc)
    ).to(device)
    tokenizer = model.tokenizer if objective == "temporal_jepa" else model
    lr = float(args.lr if args.lr is not None else tr.get("lr", 3e-4))
    opt = make_adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr,
        device,
        weight_decay=float(tr.get("weight_decay", 0.0)),
    )
    sched = make_lr_scheduler(
        opt, steps, tr.get("warmup_steps", 0), tr.get("lr_min_frac", 0.1)
    )
    trainer = BaseTrainer(cfg, device=str(device))
    trainer.use_wandb = not (args.no_wandb or args.smoke)
    trainer._wandb_key = args.wandb_key
    trainer.init_wandb()
    log_every = int(tr.get("log_every", 100))
    eval_every, eval_batches = (
        int(tr.get("eval_every", 1000)),
        int(tr.get("eval_batches", 8)),
    )
    out = args.out or tr.get("out", "checkpoints/structured_tokenizer_v2.pt")
    checkpoints = PretrainCheckpointManager(
        trainer,
        model,
        opt,
        sched,
        metadata={
            "phase": "structured_tokenizer_v2",
            "objective": objective,
            "transition_data": transition_data,
            "tokenizer_cfg": tc.__dict__,
            "temporal_jepa_cfg": (
                jepa_cfg.__dict__ if objective == "temporal_jepa" else None
            ),
            "grid_hw": ds.grid_hw,
        },
        default_monitor="val/tok/total",
    )
    start_step = checkpoints.load_if_requested(args.resume, args.resume_from)
    amp_name = "bf16" if amp and device.type == "cuda" else "off"
    print(f"[tokenizer:structured_v2] data={path} device={device} amp={amp_name}")

    @torch.no_grad()
    def run_val(step):
        model.eval()
        sums, n = {}, 0
        for vb in val_loader:
            if n >= eval_batches:
                break
            vb = to_device(vb, device)
            with amp_ctx(device, amp):
                if objective == "temporal_jepa":
                    _, vm = structured_temporal_jepa_loss(
                        model,
                        vb,
                        reconstruction_coef=float(tr.get("reconstruction_coef", 1.0)),
                        factual_coef=float(tr.get("jepa_factual_coef", 1.0)),
                        counterfactual_coef=float(
                            tr.get("jepa_counterfactual_coef", 1.0)
                        ),
                        effect_coef=float(tr.get("jepa_effect_coef", 2.0)),
                        factual_grounding_coef=float(
                            tr.get("jepa_factual_grounding_coef", 0.0)
                        ),
                        counterfactual_grounding_coef=float(
                            tr.get("jepa_counterfactual_grounding_coef", 0.0)
                        ),
                        changed_token_boost=float(
                            tr.get("changed_token_boost", 8.0)
                        ),
                        change_threshold=float(tr.get("change_threshold", 1e-3)),
                        padding_token_weight=float(
                            tr.get("padding_token_weight", 0.05)
                        ),
                    )
                else:
                    vd, _ = tokenizer(vb["state"], vb["globals"])
                    _, vm = structured_reconstruction_loss(
                        tokenizer,
                        vd,
                        vb["state"],
                        vb["globals"],
                        vb.get("obs"),
                        vb.get("mask"),
                    )
            for key, value in vm.items():
                sums[f"val/{key}"] = sums.get(f"val/{key}", 0.0) + value
            n += 1
        vals = {key: value / n for key, value in sums.items()} if n else {}
        trainer.log(vals, step=step)
        checkpoints.record_eval(step, vals)
        model.train()
        if vals:
            print(
                f"[tokenizer:structured_v2] step={step} VAL loss={vals['val/tok/total']:.4f}"
            )

    model.train()
    t0 = time.time()
    last_metrics = {}
    for step, b in zip(range(start_step + 1, steps + 1), cycle(loader)):
        b = to_device(b, device)
        with amp_ctx(device, amp):
            if objective == "temporal_jepa":
                loss, metrics = structured_temporal_jepa_loss(
                    model,
                    b,
                    reconstruction_coef=float(tr.get("reconstruction_coef", 1.0)),
                    factual_coef=float(tr.get("jepa_factual_coef", 1.0)),
                    counterfactual_coef=float(
                        tr.get("jepa_counterfactual_coef", 1.0)
                    ),
                    effect_coef=float(tr.get("jepa_effect_coef", 2.0)),
                    factual_grounding_coef=float(
                        tr.get("jepa_factual_grounding_coef", 0.0)
                    ),
                    counterfactual_grounding_coef=float(
                        tr.get("jepa_counterfactual_grounding_coef", 0.0)
                    ),
                    changed_token_boost=float(tr.get("changed_token_boost", 8.0)),
                    change_threshold=float(tr.get("change_threshold", 1e-3)),
                    padding_token_weight=float(
                        tr.get("padding_token_weight", 0.05)
                    ),
                )
                z = model.tokenizer.encode(b["state"], b["globals"])
            else:
                decoded, z = tokenizer(b["state"], b["globals"])
                loss, metrics = structured_reconstruction_loss(
                    tokenizer,
                    decoded,
                    b["state"],
                    b["globals"],
                    b.get("obs"),
                    b.get("mask"),
                )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            float(tr.get("grad_clip", 10.0)),
        )
        opt.step()
        sched.step()
        if objective == "temporal_jepa":
            progress = step / max(steps, 1)
            ema_decay = jepa_cfg.ema_decay + progress * (
                jepa_cfg.ema_end_decay - jepa_cfg.ema_decay
            )
            model.update_target(ema_decay)
        if step == 1 or step % log_every == 0 or step == steps:
            diag = {
                "tok/grad_norm": float(grad),
                "tok/lr": opt.param_groups[0]["lr"],
                "tok/seq_per_s": step * batch / max(time.time() - t0, 1e-6),
                "tok/z_rms": float(z.detach().float().pow(2).mean().sqrt()),
            }
            trainer.log({**metrics, **diag}, step=step)
            print(
                f"[tokenizer:structured_v2] step={step}/{steps} loss={metrics['tok/total']:.4f} "
                f"present={metrics['tok/present_acc']:.3f} type={metrics['tok/unit_type_acc']:.3f} "
                f"({diag['tok/seq_per_s']:.0f} seq/s)",
                flush=True,
            )
        if val_loader is not None and step % eval_every == 0:
            run_val(step)
        last_metrics = metrics
        checkpoints.save_periodic(step, metrics)
    tokenizer.eval()
    mean, std = measure_token_stats(
        tokenizer,
        loader,
        device,
        batches=2 if args.smoke else int(tr.get("stats_batches", 32)),
    )
    checkpoints.metadata.update(latent_mean=mean.cpu(), latent_std=std.cpu())
    checkpoints.finish(steps, last_metrics, out)
    trainer.finish()
    print(f"[tokenizer:structured_v2] saved -> {out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="MicroRTS DreamerV4 tokenizer pretraining. Everything lives in "
        "the --exp config (training:/data: blocks); the flags below are "
        "optional overrides. `--set training.batch=128` works too."
    )
    p.add_argument(
        "--exp",
        default="micro-rts/tokenizer/dreamerv4/pretrain_dreamerv4_tokenizer",
        help="experiment config (holds model arch + training:/data: blocks)",
    )
    # All optional overrides — None means 'take it from the config'.
    p.add_argument(
        "--data", default=None, help="override data.path (.h5 glob; newest used)"
    )
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="override model.world_lr")
    p.add_argument("--recon-coef", type=float, default=None)
    p.add_argument("--mask-coef", type=float, default=None)
    p.add_argument(
        "--latent-noise",
        type=float,
        default=None,
        help="std of Gaussian noise added to latents before decoding",
    )
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
        help="tiny CPU run (2 steps, no workers) for a fast sanity check",
    )
    return p.parse_args(argv)


def _run_dreamerv4(args, cfg):
    tr = cfg.training or {}

    steps = _pick(args.steps, tr.get("steps"), 20000)
    batch = _pick(args.batch, tr.get("batch"), 32)
    seq_len = _pick(args.seq_len, tr.get("seq_len"), 16)
    num_workers = _pick(args.num_workers, tr.get("num_workers"), 6)
    recon_coef = _pick(args.recon_coef, tr.get("recon_coef"), 1.0)
    mask_coef = _pick(args.mask_coef, tr.get("mask_coef"), 1.0)
    latent_noise = _pick(args.latent_noise, tr.get("latent_noise"), 0.0)
    # v4 objective sharpeners (defaults reproduce v3 exactly): per-group CE over
    # the one-hot channel groups, and occupied/changed-cell recon weighting at
    # the raw grid resolution (downsample=1 — this is the RECON loss, which
    # lives on the full H,W grid, unlike the dynamics flow loss).
    group_ce_coef = tr.get("group_ce_coef", 0.0)
    cell_occ_boost = tr.get("cell_occ_boost", 1.0)
    cell_changed_boost = tr.get("cell_changed_boost", 1.0)
    cell_weight_floor = tr.get("cell_weight_floor", 1.0)
    use_cell_weight = (cell_occ_boost, cell_changed_boost, cell_weight_floor) != (
        1.0,
        1.0,
        1.0,
    )
    warmup_steps = _pick(args.warmup_steps, tr.get("warmup_steps"), 0)
    lr_min_frac = tr.get("lr_min_frac", 0.1)
    val_frac = _pick(args.val_frac, tr.get("val_frac"), 0.0)
    eval_every = _pick(args.eval_every, tr.get("eval_every"), 2000)
    eval_batches = tr.get("eval_batches", 8)
    log_every = _pick(args.log_every, tr.get("log_every"), 50)
    if args.save_every is not None:
        tr.setdefault("checkpoint", {})["every_steps"] = args.save_every
    out = _pick(args.out, tr.get("out"), "checkpoints/dreamer_tokenizer.pt")
    data_glob = _pick(args.data, (cfg.data or {}).get("path"), None)
    if not data_glob:
        raise SystemExit("no dataset: set data.path in the config or pass --data")

    device_spec = args.device or cfg.run.get("device", "auto")
    if args.smoke:
        steps, num_workers = min(steps, 2), 0
        batch, seq_len = min(batch, 4), min(seq_len, 8)
        val_frac = 0.0
        device_spec = args.device or "cpu"
    device = resolve_device(device_spec)
    setup_backend(device)
    # Fast by default on CUDA; set training.amp=false only for debugging.
    amp = bool(tr.get("amp", True))
    path = resolve_data(data_glob)

    loader = build_mrts_loader(
        path,
        task="tokenizer",
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
            task="tokenizer",
            seq_len=seq_len,
            batch_size=batch,
            num_workers=0,
            locking=False,
            val_frac=val_frac,
            split="val",
            drop_last=False,
        )
    ds = loader.dataset
    print(f"[tokenizer] data={path}")
    print(
        f"[tokenizer] amp={'bf16' if amp else 'off'} (bf16 = SDPA flash-attention path)"
    )
    if group_ce_coef > 0.0 or use_cell_weight:
        print(
            f"[tokenizer] v4 objective: group_ce_coef={group_ce_coef} "
            f"cell weights occ={cell_occ_boost} changed={cell_changed_boost} "
            f"floor={cell_weight_floor}"
        )
    print(
        f"[tokenizer] windows={len(ds)} "
        f"val_windows={len(val_loader.dataset) if val_loader else 0} "
        f"obs_shape={ds.obs_shape} nvec={ds.action_nvec} device={device}"
    )

    model, model_cfg = build_model(cfg, ds.obs_shape, ds.action_nvec, device)
    opt = make_adam(model.tokenizer.parameters(), model.cfg.world_lr, device)
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
            "phase": "tokenizer",
            "model_cfg": model_cfg,
            "obs_shape": tuple(ds.obs_shape),
            "action_nvec": list(ds.action_nvec),
        },
        default_monitor="val/recon",
    )
    start_step = checkpoints.load_if_requested(args.resume, args.resume_from)

    @torch.no_grad()
    def run_val(step):
        """Held-out recon/mask BCE — evaluated without latent noise."""
        vals = {"val/recon": 0.0, "val/mask_bce": 0.0}
        n = 0
        for b in val_loader:
            if n >= eval_batches:
                break
            b = to_device(b, device)
            cw = (
                cell_weights(
                    b["obs"],
                    occ_boost=cell_occ_boost,
                    changed_boost=cell_changed_boost,
                    floor=cell_weight_floor,
                    downsample=1,
                )
                if use_cell_weight
                else None
            )
            _, m, _ = tokenizer_loss(
                model,
                b["obs"],
                b["mask"],
                recon_coef=recon_coef,
                mask_coef=mask_coef,
                group_ce_coef=group_ce_coef,
                cell_weight=cw,
            )
            vals["val/recon"] += m["tok/recon"]
            vals["val/mask_bce"] += m["tok/mask_bce"]
            n += 1
        if n:
            vals = {k: v / n for k, v in vals.items()}
            trainer.log(vals, step=step)
            checkpoints.record_eval(step, vals)
            print(
                f"[tok] step {step:>7d}  VAL recon={vals['val/recon']:.5f}  "
                f"mask_bce={vals['val/mask_bce']:.5f}"
            )

    model.train()
    t0 = time.time()
    last_metrics = {}
    for step, batch_data in zip(range(start_step + 1, steps + 1), cycle(loader)):
        batch_data = to_device(batch_data, device)
        cw = (
            cell_weights(
                batch_data["obs"],
                occ_boost=cell_occ_boost,
                changed_boost=cell_changed_boost,
                floor=cell_weight_floor,
                downsample=1,
            )
            if use_cell_weight
            else None
        )
        with amp_ctx(device, amp):
            loss, metrics, z = tokenizer_loss(
                model,
                batch_data["obs"],
                batch_data["mask"],
                recon_coef=recon_coef,
                mask_coef=mask_coef,
                latent_noise=latent_noise,
                group_ce_coef=group_ce_coef,
                cell_weight=cw,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.tokenizer.parameters(), clip)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == 1 or step == steps:
            sps = step * batch / max(time.time() - t0, 1e-6)
            # Latent-health diagnostics: a tanh bottleneck that saturates (|z|->1) or
            # collapses (|z|->0) is a classic tokenizer failure the loss alone hides.
            with torch.no_grad():
                z_abs = z.abs()
                diag = {
                    "tok/grad_norm": float(grad_norm),
                    "tok/lr": opt.param_groups[0]["lr"],
                    "tok/seq_per_s": sps,
                    "tok/z_abs_mean": float(z_abs.mean()),
                    "tok/z_rms": float(z.pow(2).mean().sqrt()),
                    "tok/z_sat": float((z_abs > 0.99).float().mean()),
                }
            trainer.log({**metrics, **diag}, step=step)
            print(
                f"[tok] step {step:>7d}  recon={metrics['tok/recon']:.5f}  "
                f"mask_bce={metrics['tok/mask_bce']:.5f}  total={metrics['tok/total']:.5f}  "
                f"|z|={diag['tok/z_abs_mean']:.3f} sat={diag['tok/z_sat']:.3f}  "
                f"gnorm={diag['tok/grad_norm']:.2f}  ({sps:.0f} seq/s)"
            )
        if val_loader is not None and step % eval_every == 0:
            run_val(step)
        last_metrics = metrics
        checkpoints.save_periodic(step, metrics)

    # Measure the trained latent RMS — the dynamics phase's normalization scale.
    # Stored both in the checkpoint dict and in the world_model buffer.
    latent_scale = measure_latent_scale(model.tokenizer, loader, device)
    model.world_model.set_latent_scale(latent_scale)
    print(f"[tokenizer] measured latent_scale (RMS) = {latent_scale:.4f}")
    trainer.log({"tok/latent_scale": latent_scale}, step=steps)

    checkpoints.metadata["latent_scale"] = latent_scale
    checkpoints.finish(steps, last_metrics, out)
    trainer.finish()
    print(f"[tokenizer] saved -> {out}")


class _TokenizerTrainer:
    def __init__(self, cfg, args, **_):
        self.cfg = cfg
        self.args = args

    def smoke_test(self):
        return self.train()


@register("trainer", "dreamerv4_tokenizer")
class DreamerV4TokenizerTrainer(_TokenizerTrainer):
    def train(self):
        return _run_dreamerv4(self.args, self.cfg)


@register("trainer", "structured_v2_tokenizer")
class StructuredV2TokenizerTrainer(_TokenizerTrainer):
    def train(self):
        return _main_structured(self.args, self.cfg)


def main(argv=None):
    args = parse_args(argv)
    cfg = Config.from_experiment(args.exp)
    overrides = list(args.set)
    if args.lr is not None:
        overrides.append(f"model.world_lr={args.lr}")
    cfg.apply_overrides(overrides)
    trainer_type = (cfg.trainer or {}).get("type")
    trainer = build("trainer", type=trainer_type, cfg=cfg, args=args)
    return trainer.smoke_test() if args.smoke else trainer.train()


if __name__ == "__main__":
    main()
