"""``AbstractDreamerTrainer`` — the env-agnostic DreamerV4 model-based RL loop.

This holds the *algorithm* of Dreamer 4 training, independent of any particular
environment or logging/checkpoint backend:

1. **collect** real transitions into a sequence replay buffer,
2. **learn the world model** on sampled ``(B, T)`` sequences (one ``world``
   optimizer step per sampled batch),
3. **imagine** rollouts from the encoded start latents and **improve the policy**
   (``actor`` + ``critic`` optimizer steps, then EMA-update the target critic).

Everything that *is* environment- or infrastructure-specific is a hook a subclass
fills in: how to build the env / policy / collector, how to compute the two loss
groups, how to evaluate, and how to log / checkpoint / print. The trainer only
assumes the **DreamerV4 model contract** — ``build_optimizers()``, ``imagine()``,
and the ``tokenizer`` / ``world_model`` / ``action_expert`` submodules — which is
shared across environments; the observation/action spaces are not.

A concrete trainer for a new environment therefore only implements the hooks with
that env's spaces and its own logging, and inherits the whole training loop.
"""

from __future__ import annotations

import abc

import torch


class AbstractDreamerTrainer(abc.ABC):
    def configure(self, d: dict | None = None) -> None:
        """Read the (env-agnostic) loop hyperparameters from a plain dict.

        Called by concrete ``__init__``. Kept separate from ``__init__`` so a
        subclass can mix this in alongside another base class (e.g. a project's
        ``BaseTrainer``) without MRO juggling.
        """
        d = dict(d or {})
        self.horizon = d.get("horizon", 32)
        self.seq_len = d.get("seq_len", 16)
        self.batch_seqs = d.get("batch_seqs", 16)
        self.wm_updates = d.get("wm_updates", 1)
        self.train_ratio = d.get("train_ratio", None)
        self.buffer_capacity = d.get("buffer_capacity", 256)
        self.warmup_steps = d.get("warmup_steps", self.seq_len)
        self.imagine_horizon = d.get("imagine_horizon", None)
        self.imagine_seeds = d.get("imagine_seeds", 256)
        self.grad_clip = d.get("grad_clip", 1000.0)

    # --- generic optimizer mechanics ------------------------------------
    @staticmethod
    def _params_of(opt) -> list:
        return [p for g in opt.param_groups for p in g["params"]]

    def _optimize(self, loss, opt, clip, *, retain_graph=False) -> float:
        """Backprop + clip (over ``opt``'s own params) + step. Model-agnostic: the
        param partition is whatever ``build_optimizers`` assigned to this optimizer,
        so no subclass-specific submodule names leak into the loop."""
        opt.zero_grad(set_to_none=True)
        loss.backward(retain_graph=retain_graph)
        gn = torch.nn.utils.clip_grad_norm_(self._params_of(opt), clip)
        opt.step()
        return float(gn)

    def world_update(self, policy, opts, batch) -> tuple[dict, torch.Tensor]:
        """One world-model optimizer step; returns (metrics, detached latents ``z``)."""
        loss, metrics, z = self.compute_world_loss(policy, batch)
        metrics["wm/grad_norm"] = self._optimize(loss, opts["world"], self.grad_clip)
        return metrics, z.detach()

    def encode_batch(self, policy, batch) -> torch.Tensor:
        """Latents for a batch when the world group is frozen (no world update)."""
        with torch.no_grad():
            return policy.tokenizer.encode(batch["obs"])

    def imagine_rollout(self, policy, z, batch):
        """Seed imagination from encoded replay latents. Default: every frame is an
        independent single-frame seed (subsampled to ``imagine_seeds``). Trainers
        whose world model consumes history override this to seed with context."""
        B, T = z.shape[:2]
        z0 = z.reshape(B * T, *z.shape[2:])
        if z0.shape[0] > self.imagine_seeds:
            idx = torch.randperm(z0.shape[0], device=z0.device)[: self.imagine_seeds]
            z0 = z0[idx]
        return policy.imagine(z0, self.imagine_horizon)

    def actor_critic_update(self, policy, opts, z, batch=None) -> dict:
        """Imagine from latents ``z`` and take the (unfrozen) actor/critic steps."""
        imagined = self.imagine_rollout(policy, z, batch)
        actor_loss, critic_loss, metrics = self.compute_actor_critic_losses(policy, imagined)

        # Actor and critic own disjoint params (per build_optimizers) but share the
        # imagined-value graph, so retain it across the actor backward. Either may
        # be frozen (missing from ``opts``) in pretrain/finetune configurations.
        if "actor" in opts:
            metrics["ac/actor_grad_norm"] = self._optimize(
                actor_loss, opts["actor"], self.grad_clip, retain_graph="critic" in opts
            )
        if "critic" in opts:
            metrics["ac/critic_grad_norm"] = self._optimize(
                critic_loss, opts["critic"], self.grad_clip)
            policy.action_expert.update_target()
        return metrics

    def train_step(self, policy, opts, buffer) -> dict:
        """One learning step: sample -> world update -> actor/critic update.

        Frozen groups (keys absent from ``opts``) are skipped: a frozen world
        model still encodes latents for imagination; frozen actor+critic skips
        imagination entirely (world-model pretrain)."""
        log: dict = {}
        wm_metrics: dict = {}
        ac_metrics: dict = {}
        for _ in range(self.wm_updates):
            batch = buffer.sample(self.batch_seqs, self.seq_len)
            if "world" in opts:
                wm_metrics, z = self.world_update(policy, opts, batch)
            else:
                z = self.encode_batch(policy, batch)
            if "actor" in opts or "critic" in opts:
                ac_metrics = self.actor_critic_update(policy, opts, z, batch)
        log.update(wm_metrics)
        log.update(ac_metrics)
        return log

    # --- main loop -------------------------------------------------------
    def run(self, iters: int) -> None:
        env = self.build_env()
        policy = self.build_policy(env)
        opts = policy.build_optimizers()
        collector = self.build_collector(env, policy)

        steps_per_iter = self.horizon * env.num_envs
        if self.train_ratio is not None:
            update_steps = max(1, self.batch_seqs * self.seq_len)
            self.wm_updates = max(1, round(float(self.train_ratio) * steps_per_iter / update_steps))
        self.effective_train_ratio = (
            self.wm_updates * self.batch_seqs * self.seq_len / max(steps_per_iter, 1)
        )
        self.on_train_start(env, policy)

        global_step = 0
        for it in range(iters):
            buffer = collector.collect()
            warmup = getattr(buffer, "num_transitions", buffer.size)
            if not buffer.can_sample(self.seq_len) or warmup < self.warmup_steps:
                global_step += steps_per_iter
                continue
            log = self.train_step(policy, opts, buffer)
            self.on_step_end(it, global_step, policy, log)
            global_step += steps_per_iter
        self.on_train_end(global_step, policy)

    # --- hooks a concrete env/trainer must implement --------------------
    @abc.abstractmethod
    def build_env(self): ...

    @abc.abstractmethod
    def build_policy(self, env): ...

    @abc.abstractmethod
    def build_collector(self, env, policy): ...

    @abc.abstractmethod
    def compute_world_loss(self, policy, batch) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """Return (scalar loss, metrics, encoded latents ``z``)."""

    @abc.abstractmethod
    def compute_actor_critic_losses(self, policy, imagined) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Return (actor_loss, critic_loss, metrics) for an imagined rollout."""

    # --- lifecycle hooks (default no-ops; override for logging/ckpt) -----
    def on_train_start(self, env, policy) -> None: ...

    def on_step_end(self, it, global_step, policy, log) -> None: ...

    def on_train_end(self, global_step, policy) -> None: ...
