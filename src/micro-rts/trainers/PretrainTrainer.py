"""``PretrainTrainer`` — shared base for the staged pretraining trainers.

This absorbs the loop that used to live as the free function ``run_training`` in
``entrypoints/incomplete_info_common.py`` (optimizer / warmup-cosine schedule /
AMP / resume / grad-clip / W&B + console logging / periodic eval / best.pt /
atomic save) *and* the per-entrypoint ``main()`` scaffolding (seed, device,
loader construction, frozen-teacher loading, loss composition, metadata builder,
param-count banner).

A concrete stage becomes a small registered subclass that declares *what* to
build — never the loop.  Override the hooks:

- ``build_loaders(self)``      -> ``(train_loader, val_loader, data_path)``
- ``load_frozen_teachers(self)`` -> dict of frozen modules (default ``{}``)
- ``build_model(self)``        -> ``nn.Module``
- ``build_loss(self)``         -> ``callable(batch) -> (total, metrics)``
                                  (default: registry-composed from ``cfg.training.loss``)
- ``build_metadata(self)``     -> dict saved into the checkpoint (default ``{}``)
- ``after_step(self)``         -> post-optimizer hook, e.g. EMA (default no-op)
- ``checkpoint_policy(self)``  -> ``(trainable_only, include_prefixes)`` (default ``(False, ())``)

"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import torch

from collectors.offline_data import cycle, to_device
from core.registry import _REGISTRY, build
from entrypoints.pretrain_common import (
    amp_ctx,
    make_adam,
    make_lr_scheduler,
    setup_backend,
)

from .BaseTrainer import BaseTrainer


class PretrainTrainer(BaseTrainer):
    # Subclasses set these (or override the corresponding hook).
    phase: str = "pretrain"
    task: str = "incomplete_dynamics_paired"
    default_seq_len: int = 64
    loss_type: str | None = None

    def __init__(self, cfg, args, device=None):
        super().__init__(cfg, device=device or (args.device if args else None)
                         or (cfg.run or {}).get("device", "auto"))
        self.args = args
        self.use_wandb = not (args.no_wandb or args.smoke) if args else True
        self._wandb_key = getattr(args, "wandb_key", None)
        self._seed_all(int((cfg.run or {}).get("seed", 0)))

        self.frozen: dict = {}
        self.metadata: dict = {}
        self._trainable_checkpoint_only = False
        self._checkpoint_include_prefixes: tuple = ()
        self.data_path = None

        # Orchestrated build: loaders -> frozen teachers -> model -> loss.
        self.train_loader, self.val_loader, self.data_path = self.build_loaders()
        self.dataset = getattr(self.train_loader, "dataset", None)
        self.grid_hw = getattr(self.dataset, "grid_hw", None)
        self.frozen = self.load_frozen_teachers()
        self.model = self.build_model()
        self.loss_fn = self.build_loss()
        self.metadata = self.build_metadata()
        trainable, resident = self.checkpoint_policy()
        self._trainable_checkpoint_only, self._checkpoint_include_prefixes = (
            trainable,
            resident,
        )
        self._print_banner()

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _seed_all(seed):
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _print_banner(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        resident = sum(p.numel() for p in self.model.parameters())
        amp = (self.cfg.training or {}).get("amp", True)
        print(
            f"[{self.phase}] trainable={trainable:,} resident={resident:,} "
            f"amp={'bf16' if amp else 'off'}",
            flush=True,
        )

    # ------------------------------------------------------------------ hooks
    def build_loaders(self):
        from entrypoints.incomplete_info_common import make_loaders

        seq_len = int((self.cfg.training or {}).get("seq_len", self.default_seq_len))
        return make_loaders(self.cfg, self.args, task=self.task, seq_len=seq_len)

    def load_frozen_teachers(self):
        return {}

    def build_model(self):
        raise NotImplementedError

    def build_loss(self):
        """Default: compose the registered loss named by ``loss_type`` with the
        ``cfg.training.loss`` coefficient map (each key ``k`` -> kwarg ``k_coef``)."""
        loss_cfg = dict((self.cfg.training or {}).get("loss") or {})
        loss_type = loss_cfg.pop("type", None) or self.loss_type
        weights = loss_cfg.pop("weights", loss_cfg)
        if not loss_type:
            raise ValueError(
                f"{type(self).__name__} requires training.loss.type or loss_type"
            )
        if loss_cfg and weights is not loss_cfg:
            raise ValueError(
                f"unsupported training.loss keys for {loss_type!r}: {sorted(loss_cfg)}"
            )
        coefs = {f"{k}_coef": v for k, v in weights.items()}
        self._assert_coef_kwargs(loss_type, coefs)
        model = self.model

        def loss_fn(batch):
            return build("loss", type=loss_type, model=model, batch=batch, **coefs)

        return loss_fn

    @staticmethod
    def _assert_coef_kwargs(loss_type, coefs):
        fn = _REGISTRY["loss"][loss_type]
        params = inspect.signature(fn).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return
        unknown = [k for k in coefs if k not in params]
        if unknown:
            raise ValueError(
                f"loss '{loss_type}' has no coefficient kwargs {unknown}; "
                f"known: {sorted(k for k in params if k.endswith('_coef'))}"
            )

    def build_metadata(self):
        return {}

    def after_step(self):
        pass

    def checkpoint_policy(self):
        return (self._trainable_checkpoint_only, self._checkpoint_include_prefixes)

    # -------------------------------------------------------------- interface
    def train(self):
        return self._loop()

    def smoke_test(self):
        # ``--smoke`` (parsed into ``args.smoke``) already shrank the loaders and
        # step budget at construction; the loop honors it (2 steps, no W&B/save).
        return self._loop()

    # ------------------------------------------------------------------- loop
    def _loop(self):
        cfg, args, device = self.cfg, self.args, self.device
        model, loss_fn = self.model, self.loss_fn
        train_loader, val_loader = self.train_loader, self.val_loader
        phase, metadata = self.phase, self.metadata
        trainable_checkpoint_only, checkpoint_include_prefixes = self.checkpoint_policy()

        training = cfg.training or {}
        setup_backend(device)
        steps = int(args.steps or training.get("steps", 10000))
        if args.smoke:
            steps = min(steps, 2)
        lr = float(training.get("lr", 2e-4))
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = make_adam(
            parameters,
            lr,
            device,
            eps=float(training.get("adam_eps", 1e-5)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
        )
        scheduler = make_lr_scheduler(
            optimizer,
            steps,
            int(training.get("warmup_steps", 0)),
            float(training.get("lr_min_frac", 0.05)),
            training.get("lr_stages"),
        )
        out = Path(args.out or training.get("out", f"checkpoints/{phase}/model.pt"))
        out.parent.mkdir(parents=True, exist_ok=True)
        start = 0
        if args.resume and out.exists():
            resume = torch.load(out, map_location="cpu", weights_only=False)
            model.load_state_dict(resume["model"], strict=not trainable_checkpoint_only)
            optimizer.load_state_dict(resume["optimizer"])
            if "scheduler" in resume:
                scheduler.load_state_dict(resume["scheduler"])
            start = int(resume["step"])
        amp = bool(training.get("amp", True)) and device.type == "cuda" and not args.smoke
        log_every = max(1, int(training.get("log_every", 100)))
        eval_every = max(1, int(training.get("eval_every", 1000)))
        eval_batches = max(1, int(training.get("eval_batches", 8)))
        eval_seed = training.get("eval_seed")
        checkpoint_every = max(1, int(training.get("checkpoint_every", 5000)))
        grad_clip = float(training.get("grad_clip", 5.0))
        log_metrics = training.get("log_metrics")
        best = float("inf")

        def metrics_for_logging(metrics):
            if not log_metrics:
                return dict(metrics)
            return {key: metrics[key] for key in log_metrics if key in metrics}

        wandb_cfg = cfg.wandb or {}
        if not wandb_cfg.get("run_name"):
            wandb_cfg["run_name"] = (cfg.run or {}).get("name", phase)
        self.init_wandb()

        def save(path, step, metrics):
            state = model.state_dict()
            if trainable_checkpoint_only:
                trainable_names = {
                    name for name, p in model.named_parameters() if p.requires_grad
                }
                root_buffers = {"state_mean", "state_std", "plan_mean", "plan_std"}
                state = {
                    name: value
                    for name, value in state.items()
                    if name in trainable_names
                    or name in root_buffers
                    or any(name.startswith(prefix) for prefix in checkpoint_include_prefixes)
                }
            payload = {
                "phase": phase,
                "step": step,
                "model": state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": metrics,
                **metadata,
            }
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)

        @torch.no_grad()
        def evaluate():
            if val_loader is None:
                return {}
            cpu_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            if eval_seed is not None:
                torch.manual_seed(int(eval_seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(eval_seed))
            model.eval()
            try:
                sums, n = {}, 0
                for batch in val_loader:
                    if n >= eval_batches:
                        break
                    batch = to_device(batch, device)
                    with amp_ctx(device, amp):
                        _, metrics = loss_fn(batch)
                    for key, value in metrics.items():
                        sums[key] = sums.get(key, 0.0) + float(value)
                    n += 1
                return {key: value / max(n, 1) for key, value in sums.items()}
            finally:
                model.train()
                torch.random.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)

        model.train()
        iterator = cycle(train_loader)
        last_metrics = {}
        started = time.time()
        try:
            for step in range(start + 1, steps + 1):
                batch = to_device(next(iterator), device)
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx(device, amp):
                    loss, metrics = loss_fn(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{phase} step {step}: non-finite loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
                optimizer.step()
                scheduler.step()
                self.after_step()
                last_metrics = {key: float(value) for key, value in metrics.items()}
                if step == start + 1 or step % log_every == 0 or step == steps:
                    elapsed = max(time.time() - started, 1e-6)
                    diagnostics = {
                        f"{phase}/grad_norm": float(grad_norm),
                        f"{phase}/lr": optimizer.param_groups[0]["lr"],
                        f"{phase}/steps_per_s": (step - start) / elapsed,
                        f"{phase}/progress": step / max(steps, 1),
                    }
                    primary_metrics = metrics_for_logging(last_metrics)
                    self.log({**primary_metrics, **diagnostics}, step=step)
                    print(
                        json.dumps(
                            {"phase": phase, "step": step, **primary_metrics, **diagnostics},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if val_loader is not None and step % eval_every == 0:
                    validation = evaluate()
                    validation_log = metrics_for_logging(validation)
                    self.log(
                        {f"val/{key}": value for key, value in validation_log.items()},
                        step=step,
                    )
                    monitor_key = training.get("monitor", f"{phase}/total")
                    monitor = (
                        validation[monitor_key]
                        if monitor_key in validation
                        else float(loss.detach())
                    )
                    if monitor < best:
                        best = monitor
                        save(out.parent / "best.pt", step, validation)
                if step % checkpoint_every == 0:
                    save(out, step, last_metrics)
            save(out, steps, last_metrics)
            if val_loader is None:
                save(out.parent / "best.pt", steps, last_metrics)
            return out
        finally:
            self.finish()
