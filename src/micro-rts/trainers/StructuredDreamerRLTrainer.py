"""Guarded online Dreamer training for the pretrained structured world model."""

from __future__ import annotations

import contextlib
import glob

import torch

from collectors.dream_collector import DreamCollector
from collectors.offline_data import build_mrts_loader, cycle, to_device
from core.registry import build, register
from environments.dream_env import DreamEnv
from environments.microrts_env import EnvConfig, MicroRTSVecEnv
from evaluation.matchplay import build_report, run_match_eval
from loss.structured_dreamer import (
    structured_agent_head_loss,
    structured_online_world_loss,
    structured_pmpo_losses,
    structured_world_metrics,
)
from models.dreamer_v2.dynamics import structured_causal_paired_loss
from rewards.rewards import reward_weight
from shared.dreamerv4.trainer import AbstractDreamerTrainer
from trainers.BaseTrainer import BaseTrainer

import models.dreamer_v2  # noqa: F401 - registry side effect


class _OptimizerBundle:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def state_dict(self):
        return {name: opt.state_dict() for name, opt in self.optimizers.items()}

    def load_state_dict(self, state):
        for name, value in state.items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(value)


@register("trainer", "structured_dreamer")
class StructuredDreamerRLTrainer(BaseTrainer, AbstractDreamerTrainer):
    """Phase-2 heads + PMPO imagination + guarded real-replay WM adaptation."""

    def __init__(self, cfg, device=None):
        BaseTrainer.__init__(self, cfg, device)
        d = cfg.training.get("dreamer", {})
        self.configure(d)
        self.mode = d.get("mode", "hybrid")
        if self.mode not in ("hybrid", "imagination"):
            raise ValueError("structured Dreamer mode must be hybrid or imagination")
        self.init_from = d.get("init_from")
        if not self.init_from:
            raise ValueError("structured Dreamer requires training.dreamer.init_from")
        self.amp = bool(d.get("amp", True))
        self.buffer_device = d.get("buffer_device", "cpu")
        self.iters = int(cfg.training.get("iters", 1000))
        self.num_envs = int(d.get("num_envs", 16))
        self.map_path = d.get("map_path", "maps/16x16/basesWorkers16x16.xml")
        self.max_steps = int(d.get("max_steps", 2000))
        self.bots = tuple(d.get("bots", ("randomBiasedAI",)))
        self.reward_preset = d.get("reward", "dense_shaped")
        self.agent_cfg = d.get("agent_heads", {}) or {}
        self.world_cfg = d.get("world_update", {}) or {}
        self.guard_cfg = d.get("world_guard", {}) or {}
        self.agent_gate_cfg = d.get("agent_gate", {}) or {}
        self.agent_warmup_updates = int(d.get("agent_warmup_updates", 1000))
        self.world_finetune_after = int(d.get("world_finetune_after", self.agent_warmup_updates))
        self._updates = 0
        self._actor_ready = False
        self._world_updates_enabled = self.mode == "hybrid"
        self._guard_bad = 0
        self._guard_baseline = None
        self._last_guard = {}
        self._agent_health = {}
        self._agent_iter = self._world_iter = None
        self._guard_batch = self._guard_paired_batch = None
        self._collector = self._eval_env = None
        self._opts = None
        self._smoke = False

    def _amp_ctx(self):
        if self.amp and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
        return contextlib.nullcontext()

    def build_env(self):
        return DreamEnv(MicroRTSVecEnv(EnvConfig(
            num_envs=self.num_envs,
            map_path=self.map_path,
            max_steps=self.max_steps,
            mode="bot",
            bots=self.bots,
            reward_weight=reward_weight(self.reward_preset),
            gridnet=True,
            opponent_action=True,
            full_state=True,
        )))

    def build_policy(self, env):
        model_cfg = {k: v for k, v in self.cfg.model.items() if k != "type"}
        policy = build(
            "model", type="structured_dreamer", obs_shape=env.obs_shape,
            action_nvec=env.action_nvec, device=str(self.device), **model_cfg,
        )
        self._load_init(policy)
        if self.mode == "imagination":
            policy.cfg.freeze.dynamics = True
            policy._apply_agent_freeze()
            self._world_updates_enabled = False
        self.grad_clip = policy.cfg.grad_clip
        return policy

    def _load_init(self, policy):
        ckpt = torch.load(self.init_from, map_location=self.device, weights_only=False)
        configured = policy.cfg.to_dict()
        for section, checkpoint_key in (
            ("tokenizer", "tokenizer_cfg"),
            ("dynamics", "dynamics_cfg"),
        ):
            saved = ckpt.get(checkpoint_key)
            if not saved:
                continue
            mismatches = {
                name: (value, configured[section].get(name))
                for name, value in saved.items()
                if configured[section].get(name) != value
            }
            if mismatches:
                raise RuntimeError(
                    f"structured init {section} config mismatch: {mismatches}"
                )
        state = ckpt.get("model", ckpt)
        missing, unexpected = policy.load_state_dict(state, strict=False)
        allowed = ("action_expert.", "scalar_heads.")
        stray = [name for name in missing if not name.startswith(allowed)]
        if stray or unexpected:
            raise RuntimeError(
                f"structured init mismatch: missing={stray[:8]} unexpected={list(unexpected)[:8]}"
            )
        # Actor starts exactly at the random prior; it is copied again after BC warmup.
        policy.action_expert.sync_actor_from_prior()
        self.console(
            f"[structured-dreamer] init={self.init_from} step={ckpt.get('step')} "
            f"monitor={ckpt.get('monitor')}"
        )

    def _restore_pretrained_world(self, policy):
        ckpt = torch.load(self.init_from, map_location=self.device, weights_only=False)
        state = ckpt.get("model", ckpt)
        own = policy.state_dict()
        for name, value in state.items():
            if name.startswith(("tokenizer.", "dynamics.")) and name in own:
                own[name].copy_(value)
        self._world_updates_enabled = False

    def build_collector(self, env, policy):
        self._collector = DreamCollector(
            env, policy, self.horizon, capacity=self.buffer_capacity,
            device=str(self.device), storage_device=self.buffer_device,
            terminal_splice=True,
        )
        return self._collector

    @staticmethod
    def _resolve_data(pattern):
        matches = sorted(glob.glob(str(pattern)))
        if not matches:
            raise FileNotFoundError(f"structured Dreamer data matched no files: {pattern!r}")
        return matches[-1]

    def _build_offline_loaders(self):
        data = self._resolve_data(self.agent_cfg.get("data", self.world_cfg.get("data", "")))
        self._agent_iter = cycle(build_mrts_loader(
            data, task="structured_rl",
            seq_len=int(self.agent_cfg.get("seq_len", 8)),
            batch_size=int(self.agent_cfg.get("batch_seqs", self.batch_seqs)),
            num_workers=int(self.agent_cfg.get("num_workers", 2)),
            shuffle=True, locking=False,
        ))
        if self.mode == "hybrid":
            world_data = self._resolve_data(self.world_cfg.get("data", data))
            self._world_iter = cycle(build_mrts_loader(
                world_data, task="structured_dynamics_paired", seq_len=1,
                batch_size=int(self.world_cfg.get("anchor_batch", 32)),
                num_workers=int(self.world_cfg.get("num_workers", 2)),
                shuffle=True, locking=False,
                paired_batch_fraction=float(self.world_cfg.get("paired_fraction", 0.5)),
            ))
            guard_loader = build_mrts_loader(
                world_data, task="structured_dynamics_eval", seq_len=1,
                batch_size=int(self.guard_cfg.get("batch", 32)), num_workers=0,
                shuffle=False, locking=False, drop_last=True,
                fixed_chunk_batches=1,
                fixed_chunk_seed=int(self.guard_cfg.get("seed", 0)),
            )
            self._guard_batch = to_device(next(iter(guard_loader)), self.device)
            paired_guard_loader = build_mrts_loader(
                world_data, task="structured_dynamics_eval", seq_len=1,
                batch_size=int(self.guard_cfg.get("paired_batch", 32)), num_workers=0,
                shuffle=False, locking=False, drop_last=True,
                paired_batch_fraction=1.0,
            )
            self._guard_paired_batch = to_device(
                next(iter(paired_guard_loader)), self.device
            )

    def _to_dev(self, batch):
        return {key: value.to(self.device) for key, value in batch.items()}

    def _world_coefficients(self):
        names = {
            "factual_coef", "counterfactual_coef", "effect_coef",
            "active_token_boost", "changed_token_boost", "change_threshold",
            "padding_token_weight", "effect_cosine_coef", "effect_norm_coef",
            "canonical_grounding_coef", "canonical_changed_boost",
            "canonical_effect_margin_coef", "canonical_effect_margin",
            "rollout_grounding_coef", "rollout_latent_coef",
            "rollout_horizon", "rollout_discount", "rollout_batch_fraction",
            "residual_correction_coef",
        }
        return {name: self.world_cfg[name] for name in names if name in self.world_cfg}

    def compute_world_loss(self, policy, batch):
        b = self._to_dev(batch)
        coeffs = self._world_coefficients()
        with self._amp_ctx():
            online_loss, online_metrics = structured_online_world_loss(policy, b, **coeffs)
            total = float(self.world_cfg.get("online_coef", 0.25)) * online_loss
            metrics = {f"online/{k}": v for k, v in online_metrics.items()}
            if self._world_iter is not None:
                anchor = to_device(next(self._world_iter), self.device)
                anchor_loss, anchor_metrics = structured_causal_paired_loss(
                    policy, anchor, **coeffs
                )
                total = total + float(self.world_cfg.get("anchor_coef", 1.0)) * anchor_loss
                metrics.update({f"anchor/{k}": v for k, v in anchor_metrics.items()})
            metrics["wm/total"] = total.detach()
        with torch.no_grad():
            z = policy.tokenizer.encode(b["full_state"], b["full_globals"])
        return total, metrics, z

    def encode_batch(self, policy, batch):
        with torch.no_grad(), self._amp_ctx():
            return policy.tokenizer.encode(
                batch["full_state"].to(self.device), batch["full_globals"].to(self.device)
            )

    def imagine_rollout(self, policy, z, batch):
        # Dreamer 4 starts one rollout from each diverse replay context.
        seeds = z[:, -1]
        if seeds.shape[0] > self.imagine_seeds:
            seeds = seeds[torch.randperm(seeds.shape[0], device=seeds.device)[:self.imagine_seeds]]
        with self._amp_ctx():
            return policy.imagine(seeds, self.imagine_horizon)

    def compute_actor_critic_losses(self, policy, imagined):
        with self._amp_ctx():
            return structured_pmpo_losses(policy, imagined)

    def _agent_update(self, policy, opts, online_batch):
        anchor = to_device(next(self._agent_iter), self.device)
        prior_coef = float(self.agent_cfg.get("prior_bc_coef", 1.0)) \
            if not self._actor_ready else 0.0
        kwargs = {
            "reward_coef": float(self.agent_cfg.get("reward_coef", 1.0)),
            "continue_coef": float(self.agent_cfg.get("continue_coef", 1.0)),
            "prior_bc_coef": prior_coef,
            "opponent_bc_coef": float(self.agent_cfg.get("opponent_bc_coef", 0.5)),
        }
        with self._amp_ctx():
            loss, metrics = structured_agent_head_loss(policy, anchor, **kwargs)
            online_coef = float(self.agent_cfg.get("online_coef", 0.25))
            if online_coef:
                online_loss, online_metrics = structured_agent_head_loss(
                    policy, self._to_dev(online_batch), **kwargs
                )
                loss = loss + online_coef * online_loss
                metrics.update({f"online/{k}": v for k, v in online_metrics.items()})
        metrics["agent/grad_norm"] = self._optimize(loss, opts["agent"], self.grad_clip)
        smoothing = float(self.agent_gate_cfg.get("smoothing", 0.05))
        for name in ("agent/reward_mae", "agent/continue_acc"):
            value = float(metrics[name])
            previous = self._agent_health.get(name, value)
            self._agent_health[name] = previous + smoothing * (value - previous)
            metrics[f"gate/{name.removeprefix('agent/')}_ema"] = self._agent_health[name]
        return metrics

    def _agent_gate_open(self):
        if not self._agent_health:
            return False
        return (
            self._agent_health["agent/reward_mae"]
            <= float(self.agent_gate_cfg.get("max_reward_mae", float("inf")))
            and self._agent_health["agent/continue_acc"]
            >= float(self.agent_gate_cfg.get("min_continue_acc", 0.0))
        )

    @torch.no_grad()
    def _evaluate_guard(self, policy):
        metrics = structured_world_metrics(
            policy, self._guard_batch,
            flow_steps=policy.cfg.actor_critic.imagine_flow_steps,
        )
        _, paired = structured_causal_paired_loss(
            policy, self._guard_paired_batch, **self._world_coefficients()
        )
        metrics["guard/paired_cf_mse"] = float(paired["causal/counterfactual"])
        metrics["guard/effect_cosine"] = float(paired["causal/effect_cosine"])
        metrics["guard/effect_norm"] = float(paired["causal/effect_norm_ratio_aggregate"])
        return metrics

    def _check_guard(self, policy):
        if self._guard_batch is None:
            return {}
        current = self._evaluate_guard(policy)
        if self._guard_baseline is None:
            self._guard_baseline = dict(current)
        base = self._guard_baseline
        bad = (
            current["guard/unweighted_mse"]
            > base["guard/unweighted_mse"] * (1.0 + float(self.guard_cfg.get("max_mse_regression", 0.03)))
            or current["guard/paired_cf_mse"]
            > base["guard/paired_cf_mse"] * (1.0 + float(self.guard_cfg.get("max_paired_regression", 0.05)))
            or current["guard/changed_f1"]
            < base["guard/changed_f1"] * float(self.guard_cfg.get("min_changed_f1_retention", 0.98))
        )
        self._guard_bad = self._guard_bad + 1 if bad else 0
        current["guard/regressed"] = float(bad)
        current["guard/bad_checks"] = float(self._guard_bad)
        if self._guard_bad >= int(self.guard_cfg.get("patience", 2)):
            self.console("[structured-dreamer] WORLD GUARD tripped; restoring pretrained dynamics")
            self._restore_pretrained_world(policy)
            current.update({"guard/rollback": 1.0, "guard/world_updates_enabled": 0.0})
        self._last_guard = current
        return current

    def train_step(self, policy, opts, buffer):
        log = {}
        for _ in range(self.wm_updates):
            batch = buffer.sample(self.batch_seqs, self.seq_len)
            log.update(self._agent_update(policy, opts, batch))
            self._updates += 1
            if (
                not self._actor_ready
                and self._updates >= self.agent_warmup_updates
                and self._agent_gate_open()
            ):
                policy.action_expert.sync_actor_from_prior()
                policy.action_expert.behavior_prior.requires_grad_(False)
                policy.collect_with_prior = False
                self._actor_ready = True
                self.console(f"[structured-dreamer] agent warmup complete at update {self._updates}")

            if (
                self._world_updates_enabled
                and self._actor_ready
                and self._updates >= self.world_finetune_after
                and "world" in opts
            ):
                wm_metrics, z = self.world_update(policy, opts, batch)
                log.update(wm_metrics)
            else:
                z = self.encode_batch(policy, batch)

            guard_every = int(self.guard_cfg.get("every_updates", 100))
            if guard_every and self._updates % guard_every == 0:
                log.update(self._check_guard(policy))

            gate_open = self._actor_ready and not bool(self._last_guard.get("guard/regressed", 0.0))
            if gate_open and ("actor" in opts or "critic" in opts):
                ac = self.actor_critic_update(policy, opts, z, batch)
                log.update(ac)
            else:
                log["gate/actor_paused"] = 1.0
        log["stage/updates"] = float(self._updates)
        log["stage/actor_ready"] = float(self._actor_ready)
        log["gate/agent_heads_healthy"] = float(self._agent_gate_open())
        log["stage/world_updates_enabled"] = float(self._world_updates_enabled)
        return log

    def on_train_start(self, env, policy):
        self.policy = policy
        self._build_offline_loaders()
        if self._guard_batch is not None:
            self._guard_baseline = self._evaluate_guard(policy)
            self._last_guard = dict(self._guard_baseline)
        self.init_wandb()
        self.start_timer()
        self.console(
            f"[structured-dreamer] mode={self.mode} envs={env.num_envs} horizon={self.horizon} "
            f"seq={self.seq_len} imag={self.imagine_horizon} train_ratio={self.effective_train_ratio:.3f}"
        )

    def on_step_end(self, it, global_step, policy, log):
        self.guard_finite({k: v for k, v in log.items() if "grad" not in k}, global_step)
        log.update(self._collector.pop_episode_stats())
        eval_cfg = self.cfg.training.get("eval", {})
        if (
            not self._smoke
            and eval_cfg.get("enabled", False)
            and it % int(eval_cfg.get("every_iters", 50)) == 0
        ):
            em = self.evaluate(eval_cfg)
            log.update(em)
            self.record_eval(
                global_step, em, policy, _OptimizerBundle(self._opts),
                extra={"updates": self._updates, "guard_baseline": self._guard_baseline},
            )
        if not self._smoke and it % self.ckpt_every == 0:
            self.save_checkpoint(
                policy, _OptimizerBundle(self._opts), global_step, log,
                extra={"updates": self._updates, "guard_baseline": self._guard_baseline},
            )
        if it % int(self.cfg.training.get("console", {}).get("every_iters", 10)) == 0:
            self.console(
                f"[{it + 1:>5}/{self.iters}] step={global_step:,} "
                f"agent={float(log.get('agent/total', 0)):.3f} "
                f"wm={float(log.get('wm/total', 0)):.3f} "
                f"actor={float(log.get('ac/actor_loss', 0)):+.3f} "
                f"value={float(log.get('ac/critic_loss', 0)):.3f} "
                f"guard={float(log.get('guard/copy_ratio', self._last_guard.get('guard/copy_ratio', 0))):.3f}"
            )
        self.log(log, step=global_step)

    def on_train_end(self, global_step, policy):
        if not self._smoke:
            self.save_checkpoint(
                policy, _OptimizerBundle(self._opts), global_step, tag="final",
                extra={"updates": self._updates, "guard_baseline": self._guard_baseline},
            )
        self.finish()
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None

    def run(self, iters):
        env = self.build_env()
        try:
            policy = self.build_policy(env)
            opts = policy.build_optimizers()
            self._opts = opts
            collector = self.build_collector(env, policy)
            steps_per_iter = self.horizon * env.num_envs
            if self.train_ratio is not None:
                update_steps = max(1, self.batch_seqs * self.seq_len)
                self.wm_updates = max(1, round(float(self.train_ratio) * steps_per_iter / update_steps))
            self.effective_train_ratio = self.wm_updates * self.batch_seqs * self.seq_len / steps_per_iter
            self.on_train_start(env, policy)
            global_step = 0
            for it in range(iters):
                buffer = collector.collect()
                if buffer.can_sample(self.seq_len) and buffer.num_transitions >= self.warmup_steps:
                    log = self.train_step(policy, opts, buffer)
                    self.on_step_end(it, global_step, policy, log)
                global_step += steps_per_iter
            self.on_train_end(global_step, policy)
        finally:
            env.close()

    def train(self):
        self.run(self.iters)

    @torch.no_grad()
    def evaluate(self, cfg):
        if self._eval_env is None:
            bots = list(cfg.get("bots", self.bots))
            per = int(cfg.get("envs_per_bot", 2))
            n = per * len(bots)
            self._eval_env = MicroRTSVecEnv(EnvConfig(
                num_envs=n, mode="bot", bots=tuple(bots), map_path=self.map_path,
                max_steps=int(cfg.get("max_steps", self.max_steps)),
                reward_weight=reward_weight("win_loss"), gridnet=True,
                full_state=True,
            ))
            self._eval_bots = [bots[i % len(bots)] for i in range(n)]
        stats = run_match_eval(
            self.policy, self._eval_env, self._eval_bots,
            games=int(cfg.get("games", 3)), max_steps=int(cfg.get("max_steps", self.max_steps)),
            device=self.device, deterministic=bool(cfg.get("greedy", True)),
        )
        overall = build_report(stats)["overall"]
        return {"eval/win_rate": overall["win_rate"], "eval/elo": overall["elo"]}

    def smoke_test(self):
        old = (self.num_envs, self.horizon, self.seq_len, self.batch_seqs,
               self.imagine_horizon, self.iters, self.agent_warmup_updates,
               self.world_finetune_after, self.agent_gate_cfg, self.use_wandb,
               self._smoke, self.warmup_steps, self.agent_cfg, self.guard_cfg,
               self._world_updates_enabled)
        self.num_envs, self.horizon, self.seq_len, self.batch_seqs = 2, 4, 3, 2
        self.imagine_horizon, self.iters = 2, 1
        self.agent_warmup_updates = self.world_finetune_after = 1
        self.agent_gate_cfg = {}
        self.use_wandb = False
        self._smoke = True
        self.warmup_steps = 0
        self.agent_cfg = {**self.agent_cfg, "seq_len": 2, "batch_seqs": 1,
                          "num_workers": 0}
        self.guard_cfg = {**self.guard_cfg, "batch": 1, "paired_batch": 1}
        # Full-size WM backward is covered by focused tests; smoke exercises the
        # real environment, checkpoint, replay, heads, and imagined actor path.
        self._world_updates_enabled = False
        self.wm_updates = 1
        try:
            self.run(1)
            return {"ok": 1.0, **self._last_guard}
        finally:
            (self.num_envs, self.horizon, self.seq_len, self.batch_seqs,
             self.imagine_horizon, self.iters, self.agent_warmup_updates,
             self.world_finetune_after, self.agent_gate_cfg, self.use_wandb,
             self._smoke, self.warmup_steps, self.agent_cfg, self.guard_cfg,
             self._world_updates_enabled) = old
