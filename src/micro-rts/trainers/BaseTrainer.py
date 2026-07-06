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
        self.verbose = cfg.run.get("verbose", True)  # console prints
        self._t_start = None

        # Run / checkpoint directory.
        self.run_name = cfg.run.get("name", "run")
        self.run_dir = Path(cfg.run.get("ckpt_dir", "checkpoints")) / self.run_name

        # Best-metric tracking config.
        ckpt_cfg = cfg.training.get("checkpoint", {}) if hasattr(cfg, "training") else {}
        self.ckpt_every = ckpt_cfg.get("every_iters", 50)
        self.keep_last = ckpt_cfg.get("keep_last", 3)
        self.monitor = ckpt_cfg.get("monitor", "eval/win_rate")
        self.monitor_mode = ckpt_cfg.get("mode", "max")
        self._best_metric = -float("inf") if self.monitor_mode == "max" else float("inf")
        self._eval_history: list[dict] = []

    # --- W&B -------------------------------------------------------------
    def init_wandb(self):
        if not self.use_wandb:
            return None  # --no-wandb: log() becomes a no-op (self._wandb stays None)
        if self._wandb is not None:
            return self._wandb
        import wandb

        if not os.environ.get("WANDB_API_KEY"):
            wandb.login()

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
        return {"run": self.cfg.run, "model": self.cfg.model, "training": self.cfg.training}

    # --- checkpoints -----------------------------------------------------
    def save_checkpoint(self, policy, optimizer, step, metrics=None, tag="latest") -> Path:
        """Atomically write a full checkpoint and rotate old ``step_*`` ones.

        ``model`` holds the entire policy state_dict, which contains the encoder,
        neck, actor, and critic parameters under their submodule prefixes.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(step),
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics or {},
            "monitor": self.monitor,
            "best_metric": self._best_metric,
            "config": self._flat_config(),
        }
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

    def load_checkpoint(self, policy, optimizer=None, tag="latest") -> dict:
        payload = torch.load(self.run_dir / f"{tag}.pt", map_location=self.device)
        policy.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    # --- eval history + best tracking ------------------------------------
    def _is_improvement(self, value: float) -> bool:
        if value != value:  # NaN never improves
            return False
        if self.monitor_mode == "max":
            return value > self._best_metric
        return value < self._best_metric

    def record_eval(self, step, eval_metrics, policy=None, optimizer=None) -> bool:
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
            self.save_checkpoint(policy, optimizer, step, eval_metrics, tag="best")
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
