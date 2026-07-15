"""``AtariDreamTrainer`` — Atari implementation of the shared DreamerV4 loop.

Fills in ``shared.dreamerv4.trainer.AbstractDreamerTrainer``'s hooks with Atari
specifics: an ``AtariVecEnv``, the registered ``atari_dreamerv4`` model, the Atari
sequence collector, and the no-mask world-model / actor-critic losses. Logging,
device, checkpointing and a greedy episode-return eval are implemented inline so the
package stays independent of the MicroRTS trainer infra (only ``shared`` is shared).

Distinct from a value-based Atari agent: three optimizers, a replay buffer of
sequences, and returns produced by imagination rather than from stored rollouts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from core.registry import build
import models.dreamer  # noqa: F401  registry side effect

from environments.atari_env import AtariVecEnv, AtariEnvConfig
from collectors.dream_collector import DreamCollector
from loss.dreamer import (
    PercentileReturnNormalizer,
    world_model_loss,
    shortcut_forcing_loss,
    actor_critic_losses,
)

from shared.dreamerv4.trainer import AbstractDreamerTrainer


def resolve_device(spec: str = "auto") -> torch.device:
    if spec in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class AtariDreamTrainer(AbstractDreamerTrainer):
    def __init__(self, cfg, device: str | None = None) -> None:
        self.cfg = cfg
        self.device = resolve_device(device or cfg.run.get("device", "auto"))
        self.verbose = cfg.run.get("verbose", True)
        self.run_name = cfg.run.get("name", "atari-dream")
        self.run_dir = Path(cfg.run.get("ckpt_dir", "checkpoints")) / self.run_name
        self.use_wandb = True
        self._wandb = None

        d = cfg.training.get("dreamer", {})
        self.configure(d)
        self.use_flow = d.get("use_flow", True)
        self.flow_coef = d.get("flow_coef", 1.0)
        self.recon_dynamic_coef = d.get("recon_dynamic_coef", 0.0)
        self.reward_balance_nonzero = d.get("reward_balance_nonzero", False)
        self.iters = cfg.training.get("iters", 1000)
        self.return_normalizer = None

        e = cfg.training.get("env", {})
        self.env_cfg = AtariEnvConfig(
            game=e.get("game", "pong"), num_envs=e.get("num_envs", 16),
            frameskip=e.get("frameskip", 4), resize=e.get("resize", 64),
            grayscale=e.get("grayscale", True), frame_stack=e.get("frame_stack", 1),
            max_steps=e.get("max_steps", 27000),
            noop_max=e.get("noop_max", 30),
            repeat_action_probability=e.get("sticky", 0.0),
            full_action_space=e.get("full_action_space", False),
            seed=cfg.run.get("seed", 0),
            clip_reward=e.get("clip_reward", False),
            reward_scale=e.get("reward_scale", 1.0),
            dense_reward=e.get("dense_reward", "none"),
            dense_reward_coef=e.get("dense_reward_coef", 0.0),
            dense_reward_gamma=e.get("dense_reward_gamma", 0.997),
            dense_reward_clip=e.get("dense_reward_clip", 0.05),
        )
        self.ckpt_every = cfg.training.get("checkpoint", {}).get("every_iters", 50)
        self.console_every = cfg.training.get("console", {}).get("every_iters", 10)
        self.policy = None
        self._eval_env = None
        self._t0 = None

    # --- console / wandb / checkpoint -----------------------------------
    def console(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def init_wandb(self):
        if not self.use_wandb:
            return
        try:
            import wandb
        except Exception:
            self.use_wandb = False
            return
        wb = self.cfg.wandb
        self._wandb = wandb.init(project=wb.get("project", "atari-dreamer"),
                                 entity=wb.get("entity"), name=self.run_name,
                                 config={"run": self.cfg.run, "model": self.cfg.model,
                                         "training": self.cfg.training})

    def log(self, metrics, step):
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def save_checkpoint(self, policy, step, metrics=None, tag="latest"):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"step": int(step), "model": policy.state_dict(),
                    "metrics": metrics or {}, "config": {"model": self.cfg.model}},
                   self.run_dir / f"{tag}.pt")

    @staticmethod
    def _finite(metrics):
        return all(float(v) == float(v) and abs(float(v)) != float("inf")
                   for v in metrics.values())

    # --- AbstractDreamerTrainer build hooks -----------------------------
    def build_env(self):
        return AtariVecEnv(self.env_cfg)

    def build_policy(self, env):
        model_cfg = {k: v for k, v in self.cfg.model.items() if k != "type"}
        policy = build("model", type=self.cfg.model.get("type", "atari_dreamerv4"),
                       obs_shape=env.obs_shape, num_actions=env.num_actions,
                       device=str(self.device), **model_cfg)
        self.grad_clip = policy.cfg.grad_clip
        self.policy = policy
        ac = policy.cfg.actor_critic
        if str(ac.return_norm).lower() in ("percentile", "perc"):
            self.return_normalizer = PercentileReturnNormalizer(
                rate=ac.return_norm_rate,
                limit=ac.return_norm_limit,
                perclo=ac.return_norm_low,
                perchi=ac.return_norm_high,
            )
        else:
            self.return_normalizer = None
        return policy

    def build_collector(self, env, policy):
        return DreamCollector(env, policy, self.horizon,
                              capacity=self.buffer_capacity, device=str(self.device))

    # --- AbstractDreamerTrainer loss hooks ------------------------------
    def compute_world_loss(self, policy, batch):
        obs = batch["obs"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        cont = batch["cont"].to(self.device)
        is_first = batch.get("is_first", None)
        is_first = is_first.to(self.device) if is_first is not None else None
        loss, metrics, z = world_model_loss(
            policy, obs, action, reward, cont, is_first,
            recon_dynamic_coef=self.recon_dynamic_coef,
            reward_balance_nonzero=self.reward_balance_nonzero,
        )
        if self.use_flow:
            floss, fm = shortcut_forcing_loss(policy, z.detach(), action, is_first)
            loss = loss + self.flow_coef * floss
            metrics.update(fm)
        return loss, metrics, z

    def compute_actor_critic_losses(self, policy, imagined):
        ac = policy.cfg.actor_critic
        return actor_critic_losses(policy, imagined, gamma=ac.gamma, lam=ac.lam,
                                   entropy_coef=ac.entropy_coef,
                                   normalize_adv=ac.normalize_adv,
                                   return_normalizer=self.return_normalizer)

    # --- lifecycle -------------------------------------------------------
    def on_train_start(self, env, policy):
        self.init_wandb()
        n_params = sum(p.numel() for p in policy.parameters())
        self.console(f"[atari-dream] run={self.run_name} game={self.env_cfg.game} "
                     f"device={self.device} params={n_params/1e6:.1f}M | iters={self.iters} "
                     f"envs={env.num_envs} horizon={self.horizon} seq_len={self.seq_len} "
                     f"updates={self.wm_updates} train_ratio={self.effective_train_ratio:.1f}")
        self._t0 = time.perf_counter()

    def on_step_end(self, it, global_step, policy, log):
        if not self._finite({k: v for k, v in log.items() if "grad" not in k}):
            raise RuntimeError(f"non-finite metrics at step {global_step}: {log}")
        eval_cfg = self.cfg.training.get("eval", {})
        if eval_cfg.get("enabled", False) and it % eval_cfg.get("every_iters", 50) == 0:
            log.update(self.evaluate(eval_cfg))
        if it % self.ckpt_every == 0:
            self.save_checkpoint(policy, global_step, log, tag="latest")
        if it % self.console_every == 0:
            self._print(it, global_step, log)
        self.log(log, step=global_step)

    def on_train_end(self, global_step, policy):
        self.save_checkpoint(policy, global_step, tag="final")
        self.console(f"[done] {self.iters} iters, {global_step:,} steps -> {self.run_dir}")
        if self._wandb is not None:
            self._wandb.finish()

    def _print(self, it, step, log):
        sps = step / max(time.perf_counter() - (self._t0 or time.perf_counter()), 1e-6)
        self.console(
            f"[{it + 1:>5}/{self.iters}] step {step:>10,} {sps/1000:5.1f}k sps | "
            f"recon {log.get('wm/recon', 0):.3f}/{log.get('wm/recon_pixel', log.get('wm/recon', 0)):.3f} "
            f"base {log.get('wm/recon_mean_baseline', 0):.4f} "
            f"dyn {log.get('wm/dynamics', 0):.2e} "
            f"copy {log.get('wm/dynamics_copy_baseline', 0):.2e} "
            f"xcopy {log.get('wm/dynamics_vs_copy', 0):.2f} "
            f"zΔ {log.get('wm/latent_delta_abs', 0):.4f} "
            f"rew {log.get('wm/reward', 0):.4f}/{log.get('wm/reward_nonzero', 0):.3f} "
            f"rnz {log.get('wm/reward_nonzero_frac', 0):.3f} "
            f"flow {log.get('flow/matching', 0):.3f} | "
            f"actor {log.get('ac/actor_loss', 0):+.3f} critic {log.get('ac/critic_loss', 0):.3f} "
            f"ret {log.get('ac/return_mean', 0):+.3f} "
            f"rscale {log.get('ac/return_norm_scale', 0):.2f} "
            f"ent {log.get('ac/entropy', 0):.3f}"
            + (f" | eval_ret {log['eval/episode_return']:+.1f}" if 'eval/episode_return' in log else ""))

    @torch.no_grad()
    def evaluate(self, eval_cfg):
        if self._eval_env is None:
            ec = AtariEnvConfig(**{**self.env_cfg.__dict__,
                                   "num_envs": eval_cfg.get("num_envs", 8),
                                   "seed": self.env_cfg.seed + 999})
            self._eval_env = AtariVecEnv(ec)
        env = self._eval_env
        td = env.reset()
        running = np.zeros(env.num_envs)
        completed = []
        for _ in range(eval_cfg.get("steps", 1000)):
            out = self.policy.step(td["obs"].to(self.device), deterministic=eval_cfg.get("greedy", True))
            td = env.step(out["action"])
            score_reward = td.get("raw_reward", td["reward"])
            running += score_reward.numpy()
            done = td["done"].numpy()
            for i in np.where(done)[0]:
                completed.append(running[i]); running[i] = 0.0
        ret = float(np.mean(completed)) if completed else float(running.mean())
        return {"eval/episode_return": ret, "eval/episodes": float(len(completed))}

    # --- entrypoints -----------------------------------------------------
    def train(self):
        self.run(self.iters)

    def smoke_test(self):
        env = AtariVecEnv(AtariEnvConfig(game=self.env_cfg.game, num_envs=4, max_steps=200))
        try:
            policy = self.build_policy(env)
            opts = policy.build_optimizers()
            self.horizon, self.seq_len, self.batch_seqs = 6, 4, 4
            self.imagine_horizon, self.imagine_seeds = 4, 8
            collector = DreamCollector(env, policy, self.horizon, capacity=32, device=str(self.device))
            buf = collector.collect()
            batch = buf.sample(self.batch_seqs, self.seq_len)
            wm_metrics, z = self.world_update(policy, opts, batch)
            ac_metrics = self.actor_critic_update(policy, opts, z)
            out = {**wm_metrics, **ac_metrics}
            assert self._finite(out), "non-finite smoke metrics"
            return out
        finally:
            env.close()
