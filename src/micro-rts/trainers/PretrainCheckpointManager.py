"""Config-driven checkpoint lifecycle shared by offline pretraining phases."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch


def _scalar_metrics(metrics):
    return {
        key: float(value.detach()) if torch.is_tensor(value) else float(value)
        for key, value in (metrics or {}).items()
    }


class PretrainCheckpointManager:
    """One checkpoint authority for a pretraining process.

    The manager delegates atomic IO, best tracking, and last-K rotation to the
    supplied :class:`BaseTrainer`.  It additionally owns scheduler state,
    automatic resume, phase metadata, and the legacy ``training.out`` export.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            raise RuntimeError(
                "only one PretrainCheckpointManager may exist per process"
            )
        obj = super().__new__(cls)
        cls._instance = obj
        return obj

    def __init__(self, trainer, model, optimizer, scheduler, metadata, default_monitor):
        self.trainer = trainer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metadata = dict(metadata)
        cfg = trainer.cfg.training.get("checkpoint", {})
        self.every = int(
            cfg.get("every_steps", trainer.cfg.training.get("save_every", 0))
        )
        self.trainer.keep_last = int(cfg.get("keep_last", 3))
        self.trainer.monitor = cfg.get("monitor", default_monitor)
        self.trainer.monitor_mode = cfg.get("mode", "min")
        self.trainer._best_metric = (
            float("inf") if self.trainer.monitor_mode == "min" else -float("inf")
        )
        self.resume = bool(cfg.get("resume", False))
        self.resume_from = cfg.get("resume_from", "latest")

    def close(self):
        type(self)._instance = None

    def load_if_requested(self, resume=None, resume_from=None) -> int:
        enabled = self.resume if resume is None else bool(resume)
        source = resume_from or self.resume_from
        if not enabled and not resume_from:
            return 0
        payload = self.trainer.load_checkpoint(
            self.model, self.optimizer, tag=source, scheduler=self.scheduler
        )
        step = int(payload.get("step", 0))
        print(f"[checkpoint] resumed {source} at step {step}", flush=True)
        return step

    def _save(self, step, metrics, tag):
        return self.trainer.save_checkpoint(
            self.model,
            self.optimizer,
            step,
            _scalar_metrics(metrics),
            tag=tag,
            scheduler=self.scheduler,
            extra=self.metadata,
        )

    def save_periodic(self, step, metrics):
        if not self.every or step % self.every:
            return
        self._save(step, metrics, "latest")
        self._save(step, metrics, f"step_{step}")

    def record_eval(self, step, metrics):
        return self.trainer.record_eval(
            step,
            metrics,
            self.model,
            self.optimizer,
            scheduler=self.scheduler,
            extra=self.metadata,
        )

    def finish(self, step, metrics, export_path=None):
        latest = self._save(step, metrics, "latest")
        self._save(step, metrics, f"step_{step}")
        if export_path:
            target = Path(export_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.resolve() != latest.resolve():
                tmp = target.with_suffix(target.suffix + ".tmp")
                shutil.copy2(latest, tmp)
                os.replace(tmp, target)
        self.close()
        return latest
