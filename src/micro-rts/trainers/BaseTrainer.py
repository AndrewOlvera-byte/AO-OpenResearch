"""``BaseTrainer`` — shared scaffolding for config-driven trainers.

Owns the cross-cutting concerns every trainer needs:

- **device** resolution (cuda-if-available),
- **Weights & Biases** setup / logging,
- **checkpointing** — full state (model = encoder/neck/actor/critic, optimizer,
  step, metrics, config), atomic writes, and rotation (``keep_last``),
- **best tracking** — monitor a configurable eval metric (min/max) and keep a
  ``best.pt`` + ``best.json`` snapshot of the best checkpoint so far,
- **eval history** — append every periodic eval to ``eval_history.json`` so you
  can see what each checkpoint did in eval over the course of training,
- a generic **non-finite collapse guard**.

Concrete trainers (``PPOTrainer``, ``DreamerRLTrainer``, ...) subclass this and
implement ``train`` (the real loop) and ``smoke_test`` (one fwd/bwd, no W&B / no
save). RL/PPO-specific collapse guards (entropy / KL / explained-variance) live in
``PPOTrainer``; see ``trainers/guards.py`` for the stateless diagnostics.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

from .guards import is_finite


def fmt_duration(seconds: float) -> str:
    """Seconds -> ``H:MM:SS`` (hours uncapped)."""
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def resolve_device(spec: str = "auto") -> torch.device:
    """``"auto"`` -> cuda if available else cpu; otherwise honor the request."""
    if spec in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class CollapseError(RuntimeError):
    """Raised when training health checks detect an unrecoverable collapse."""


class BaseTrainer:
    def __init__(self, cfg, device: str | None = None) -> None:
        self.cfg = cfg
        self.device = resolve_device(device or cfg.run.get("device", "auto"))
        self._wandb = None
        self.use_wandb = True  # entrypoints flip this off via --no-wandb
        self._wandb_key = None  # entrypoints set this via --wandb-key
        self.verbose = cfg.run.get("verbose", True)  # console prints
        self._t_start = None

        # Run / checkpoint directory.
        self.run_name = cfg.run.get("name", "run")
        self.run_dir = Path(cfg.run.get("ckpt_dir", "checkpoints")) / self.run_name

        # Best-metric tracking config.
        ckpt_cfg = (
            cfg.training.get("checkpoint", {}) if hasattr(cfg, "training") else {}
        )
        self.ckpt_every = ckpt_cfg.get("every_iters", 50)
        self.keep_last = ckpt_cfg.get("keep_last", 3)
        self.monitor = ckpt_cfg.get("monitor", "eval/win_rate")
        self.monitor_mode = ckpt_cfg.get("mode", "max")
        self._best_metric = (
            -float("inf") if self.monitor_mode == "max" else float("inf")
        )
        self._eval_history: list[dict] = []

    # --- W&B -------------------------------------------------------------
    def _resolve_wandb_key(self, wandb) -> bool:
        """Ensure we have a W&B API key, without ever crashing on a missing one.

        Resolution order: explicit ``--wandb-key`` (stashed on ``self._wandb_key``)
        -> ``WANDB_API_KEY`` env var -> a key already saved in the container's
        ``~/.netrc`` (from a previous run). If none is found we *prompt* (when a TTY
        is attached): the user can paste a key or press Enter to disable W&B for this
        run. A pasted key is persisted with ``wandb.login`` so it lives in the
        container's ``~/.netrc`` and survives ``docker exec``/restart (until the image
        is rebuilt). Returns True if authenticated, False if W&B should be disabled.
        """
        key = getattr(self, "_wandb_key", None) or os.environ.get("WANDB_API_KEY")
        if not key:
            # Already logged in (netrc from a prior run in this container)?
            try:
                if wandb.api.api_key:
                    return True
            except Exception:
                pass
        if key:
            # Persist an explicitly-provided key so future runs need no flag/env.
            try:
                wandb.login(key=key, relogin=True)
                self.console(
                    "[wandb] key saved to the container's ~/.netrc "
                    "(persists until the image is rebuilt)."
                )
            except Exception as e:
                self.console(f"[wandb] login failed ({e}); disabling W&B for this run.")
                return False
            return True

        # No key anywhere — ask, but never crash.
        if not sys.stdin.isatty():
            self.console(
                "[wandb] no API key found (checked --wandb-key / WANDB_API_KEY "
                "/ ~/.netrc) and no TTY to prompt; disabling W&B for this run. "
                "Re-run with -it, pass --wandb-key, or use --no-wandb."
            )
            return False
        print(
            "\n[wandb] No Weights & Biases API key found "
            "(checked --wandb-key, WANDB_API_KEY, and ~/.netrc)."
        )
        ans = input(
            "[wandb] Paste your API key to enable logging, or press Enter to "
            "disable W&B for this run: "
        ).strip()
        if not ans:
            self.console("[wandb] disabled for this run.")
            return False
        try:
            wandb.login(key=ans, relogin=True)
            self.console(
                "[wandb] key saved to the container's ~/.netrc "
                "(persists until the image is rebuilt)."
            )
        except Exception as e:
            self.console(f"[wandb] login failed ({e}); disabling W&B for this run.")
            return False
        return True

    def init_wandb(self):
        if not self.use_wandb:
            return None  # --no-wandb: log() becomes a no-op (self._wandb stays None)
        if self._wandb is not None:
            return self._wandb
        import wandb

        if not self._resolve_wandb_key(wandb):
            self.use_wandb = False
            return None

        wb = self.cfg.wandb
        self._wandb = wandb.init(
            project=wb.get("project", "micro-rts"),
            entity=wb.get("entity"),
            name=wb.get("run_name"),
            tags=wb.get("tags"),
            config=self._flat_config(),
        )
        return self._wandb

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    # --- console output --------------------------------------------------
    def console(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def start_timer(self) -> None:
        self._t_start = time.perf_counter()

    def progress(self, iters_done: int, iters_total: int, steps_done: int) -> dict:
        """Wall-clock progress that amortizes train *and* eval time.

        ETA is ``avg_iter_time * remaining_iters`` where ``avg_iter_time`` is the
        mean over completed iterations — so the periodic (slower) eval iterations
        are naturally folded into the estimate.
        """
        elapsed = time.perf_counter() - (self._t_start or time.perf_counter())
        avg = elapsed / max(iters_done, 1)
        eta = avg * max(iters_total - iters_done, 0)
        return {
            "elapsed": elapsed,
            "eta": eta,
            "sps": steps_done / elapsed if elapsed > 0 else 0.0,
            "elapsed_str": fmt_duration(elapsed),
            "eta_str": fmt_duration(eta),
        }

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None

    def _flat_config(self) -> dict:
        return {
            "run": self.cfg.run,
            "model": self.cfg.model,
            "training": self.cfg.training,
        }

    # --- checkpoints -----------------------------------------------------
    def save_checkpoint(
        self,
        policy,
        optimizer,
        step,
        metrics=None,
        tag="latest",
        scheduler=None,
        extra=None,
    ) -> Path:
        """Atomically write a full checkpoint and rotate old ``step_*`` ones.

        ``model`` holds the entire policy state_dict, which contains the encoder,
        neck, actor, and critic parameters under their submodule prefixes.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(step),
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics or {},
            "monitor": self.monitor,
            "best_metric": self._best_metric,
            "config": self._flat_config(),
        }
        payload.update(extra or {})
        path = self.run_dir / f"{tag}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        if tag.startswith("step_"):
            self._rotate_step_checkpoints()
        return path

    def _rotate_step_checkpoints(self) -> None:
        ckpts = sorted(
            self.run_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        for stale in ckpts[: max(0, len(ckpts) - self.keep_last)]:
            stale.unlink(missing_ok=True)

    def load_checkpoint(
        self, policy, optimizer=None, tag="latest", scheduler=None
    ) -> dict:
        path = Path(tag)
        if path.suffix != ".pt" or not path.is_file():
            path = self.run_dir / f"{tag}.pt"
        payload = torch.load(path, map_location=self.device, weights_only=False)
        policy.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if payload.get("best_metric") is not None:
            self._best_metric = float(payload["best_metric"])
        return payload

    def load_weights_from(self, policy, path) -> dict:
        """Warm-start: load *only* the model weights from an arbitrary checkpoint
        (typically the best.pt of a *different* run). Unlike ``load_checkpoint`` this
        does not touch the optimizer, step, or best-metric — the new run starts fresh
        at step 0 with its own LR schedule and curriculum, standing on the loaded
        policy. ``path`` may be absolute or relative to the process CWD (repo root)."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"init_from checkpoint not found: {p}")
        payload = torch.load(p, map_location=self.device, weights_only=False)
        policy.load_state_dict(payload["model"])
        return payload

    # --- eval history + best tracking ------------------------------------
    def _is_improvement(self, value: float) -> bool:
        if value != value:  # NaN never improves
            return False
        if self.monitor_mode == "max":
            return value > self._best_metric
        return value < self._best_metric

    def record_eval(
        self,
        step,
        eval_metrics,
        policy=None,
        optimizer=None,
        scheduler=None,
        extra=None,
    ) -> bool:
        """Append an eval record, update best, optionally save ``best.pt``.

        Returns whether this eval is a new best on the monitored metric. The
        full eval history is persisted to ``eval_history.json`` so each
        checkpoint's eval result is inspectable after the run.
        """
        value = eval_metrics.get(self.monitor)
        is_best = value is not None and self._is_improvement(float(value))
        if is_best:
            self._best_metric = float(value)

        record = {
            "step": int(step),
            "metrics": {k: float(v) for k, v in eval_metrics.items()},
            "is_best": bool(is_best),
            "monitor": self.monitor,
        }
        self._eval_history.append(record)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "eval_history.json", "w") as f:
            json.dump(self._eval_history, f, indent=2)

        if is_best and policy is not None:
            self.save_checkpoint(
                policy,
                optimizer,
                step,
                eval_metrics,
                tag="best",
                scheduler=scheduler,
                extra=extra,
            )
            with open(self.run_dir / "best.json", "w") as f:
                json.dump(record, f, indent=2)
        return is_best

    # --- generic collapse guard ------------------------------------------
    def guard_finite(self, metrics: dict, step: int) -> None:
        """Raise ``CollapseError`` if any reported metric is non-finite."""
        if not is_finite(metrics):
            raise CollapseError(f"non-finite metrics at step {step}: {metrics}")

    # --- to implement ----------------------------------------------------
    def train(self):
        raise NotImplementedError

    def smoke_test(self):
        raise NotImplementedError
