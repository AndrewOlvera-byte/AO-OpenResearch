"""``PPOTrainer`` — owns the full PPO training loop.

Two construction modes:

- **Legacy / component mode** — ``PPOTrainer(policy, epochs=..., minibatches=...)``.
  A thin optimizer + minibatch updater; ``update(buffer)`` runs the epochs x
  minibatches PPO step over a filled ``RolloutBuffer``. Used by the throughput
  ``Runner`` and its tests.
- **Config mode** — ``PPOTrainer(cfg)``. Subclasses ``BaseTrainer`` and drives the
  whole run from config: build curriculum -> env -> policy -> optimizer, then loop
  collect -> GAE -> PPO update -> curriculum step -> eval -> W&B logging. ``train``
  runs the real run; ``smoke_test`` does one collect + one update with no W&B / no
  save.

The PPO math lives in ``loss/ppo.py``; this class is orchestration only.
"""

from __future__ import annotations

import contextlib

import torch

from collectors.buffer import RolloutBuffer
from collectors.collector import Collector
from collectors.selfplay_collector import SelfPlayCollector
from environments.microrts_env import EnvConfig, MicroRTSVecEnv
from evaluation.matchplay import build_report, run_match_eval
from loss.ppo import ppo_loss
from rewards.rewards import reward_weight

from core.registry import build

from .BaseTrainer import BaseTrainer, CollapseError
from .guards import entropy_fraction, explained_variance, max_entropy


class PPOTrainer(BaseTrainer):
    def __init__(self, policy=None, cfg=None, *, lr=2.5e-4, epochs=4, minibatches=4,
                 clip=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
                 clip_vloss=True, gamma=0.99, lam=0.95, target_kl=None, anneal_lr=False):
        if cfg is not None:
            super().__init__(cfg)
            hp = cfg.training.get("ppo", {})
            lr = hp.get("lr", lr)
            epochs = hp.get("epochs", epochs)
            minibatches = hp.get("minibatches", minibatches)
            clip = hp.get("clip", clip)
            vf_coef = hp.get("vf_coef", vf_coef)
            ent_coef = hp.get("ent_coef", ent_coef)
            max_grad_norm = hp.get("max_grad_norm", max_grad_norm)
            clip_vloss = hp.get("clip_vloss", clip_vloss)
            gamma = hp.get("gamma", gamma)
            lam = hp.get("lam", lam)
            target_kl = hp.get("target_kl", target_kl)
            anneal_lr = hp.get("anneal_lr", anneal_lr)
            self.amp = hp.get("amp", False)
            g = cfg.training.get("guards", {})
            self.entropy_floor_frac = g.get("entropy_floor_frac", 0.05)
            self.entropy_patience = g.get("entropy_patience", 30)
            self.guards_fatal = g.get("fatal", False)
        else:
            self.cfg = None
            self.device = torch.device("cpu")
            self._wandb = None
            self.amp = False
            self.entropy_floor_frac, self.entropy_patience, self.guards_fatal = 0.05, 30, False

        self.epochs, self.minibatches = epochs, minibatches
        self.clip, self.vf_coef, self.ent_coef = clip, vf_coef, ent_coef
        self.max_grad_norm, self.clip_vloss = max_grad_norm, clip_vloss
        self.gamma, self.lam, self.lr, self.target_kl = gamma, lam, lr, target_kl
        self.anneal_lr = anneal_lr

        self._entropy_collapse_steps = 0
        self._eval_env = None
        self.policy = None
        self.opt = None
        self._resume_from = None          # set by the entrypoint's --resume flag
        self._start_step, self._start_it = 0, 0
        if policy is not None:
            self._attach(policy)

    def _attach(self, policy) -> None:
        self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=self.lr, eps=1e-5)

    def _amp_ctx(self):
        """bf16 autocast on CUDA when enabled; a no-op otherwise. bf16 (not fp16)
        needs no GradScaler — its dynamic range matches fp32 — and it also *lowers*
        activation VRAM, so it speeds up the update without a memory-safety cost."""
        if getattr(self, "amp", False) and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # --- the PPO update over a filled buffer (shared by both modes) -------
    def update(self, buffer) -> float:
        metrics = self._update(buffer)
        return metrics["total_loss"]

    def _update(self, buffer) -> dict:
        agg: dict[str, float] = {}
        n, skipped = 0, 0
        for _ in range(self.epochs):
            stop = False
            for mb in buffer.minibatches(self.minibatches):
                with self._amp_ctx():
                    loss, m = ppo_loss(
                        mb, self.policy, self.clip, self.vf_coef, self.ent_coef, self.clip_vloss
                    )
                # KL-blowup guard: an exploded KL means a bad minibatch — drop the
                # whole update epoch loop rather than take a destructive step.
                if self.target_kl is not None and m["approx_kl"] > 1.5 * self.target_kl:
                    stop = True
                    break
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                # Non-finite guard: never let a NaN/Inf loss or grad corrupt weights.
                if not (torch.isfinite(loss) and torch.isfinite(grad_norm)):
                    self.opt.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
                self.opt.step()
                m["grad_norm"] = float(grad_norm)
                for k, v in m.items():
                    agg[k] = agg.get(k, 0.0) + v
                n += 1
            if stop:
                break
        out = {k: v / max(n, 1) for k, v in agg.items()}
        out["updates_skipped"] = float(skipped)
        return out

    # --- config-mode building blocks -------------------------------------
    def _build_policy(self, env):
        model_cfg = {k: v for k, v in self.cfg.model.items() if k != "type"}
        return build(
            "model", type=self.cfg.model["type"],
            obs_shape=env.obs_shape, action_nvec=env.action_nvec,
            device=str(self.device), **model_cfg,
        )

    def _build_env(self, env_config, reward_weight_vec):
        """Build base env -> ShapeReward (annealed weights) -> optional NormalizeReward.

        Keeps a handle to the shaper (``self._shaper``) so the training loop can
        update the reward weights each iteration without rebuilding the env.
        """
        from environments.shape_reward import ShapeReward

        base = MicroRTSVecEnv(env_config)
        self._shaper = ShapeReward(base, weight=reward_weight_vec)
        env = self._shaper
        coll_cfg = self.cfg.training.get("collector", {})
        if coll_cfg.get("normalize_reward", False):
            from environments.normalize import NormalizeReward
            env = NormalizeReward(
                env, gamma=self.gamma, clip=coll_cfg.get("reward_clip", 10.0)
            )
        return env

    def _make_collector(self, env, phase, horizon):
        if phase == "selfplay":
            opponent = self._build_policy(env)
            return SelfPlayCollector(env, self.policy, opponent, horizon, str(self.device))
        return Collector(env, self.policy, horizon, str(self.device))

    @staticmethod
    def _rollout_stats(buf) -> dict:
        d = buf.data
        return {
            "rollout/reward_mean": float(d["reward"].mean()),
            "rollout/return_mean": float(d["return"].mean()),
            "rollout/value_mean": float(d["value"].mean()),
            "rollout/advantage_std": float(d["advantage"].std()),
            "rollout/episodes": float(d["done"].sum()),
            # explained variance of the (pre-update) critic vs realized returns;
            # drifting toward 0/negative is the classic value-collapse signal.
            "rollout/explained_variance": explained_variance(d["value"], d["return"]),
        }

    def _entropy_guard(self, entropy: float, action_nvec) -> dict:
        """Track entropy-as-fraction-of-max; flag sustained policy collapse."""
        frac = entropy_fraction(entropy, action_nvec)
        if frac < self.entropy_floor_frac:
            self._entropy_collapse_steps += 1
        else:
            self._entropy_collapse_steps = 0
        collapsed = self._entropy_collapse_steps >= self.entropy_patience
        return {"guards/entropy_frac": frac, "guards/entropy_collapse": float(collapsed)}

    # --- full run --------------------------------------------------------
    def train(self) -> None:
        assert self.cfg is not None, "train() requires config-mode construction"
        cur = build("curriculum", **self.cfg.training["curriculum"])
        coll_cfg = self.cfg.training.get("collector", {})
        horizon = coll_cfg.get("horizon", 128)
        iters = self.cfg.training.get("iters", 1000)
        eval_cfg = self.cfg.training.get("eval", {})

        console_every = self.cfg.training.get("console", {}).get("every_iters", 10)
        steps_per_iter = horizon * self.cfg.training["curriculum"].get("num_envs", 256)

        # Resume: load the full state (model + optimizer + step + best_metric) and
        # continue from the saved step. The env / collector are built for the resumed
        # curriculum phase, and LR annealing / schedules pick up at the right iteration.
        start_step = 0
        resume_tag = getattr(self, "_resume_from", None)
        env = self._build_env(cur.env_config(0), cur.reward_weight_vec(0))
        self._attach(self._build_policy(env))
        init_from = self.cfg.run.get("init_from")
        if resume_tag is not None:
            payload = self.load_checkpoint(self.policy, self.opt, tag=resume_tag)
            start_step = int(payload["step"])
            self._best_metric = payload.get("best_metric", self._best_metric)
            if start_step > 0:  # rebuild env for the phase we're resuming into
                env = self._build_env(cur.env_config(start_step), cur.reward_weight_vec(start_step))
            self.console(f"[resume] loaded '{resume_tag}.pt' @ step {start_step:,} "
                         f"(best {self.monitor}={self._best_metric:+.4f})")
        elif init_from:
            # Warm-start a fresh run from another run's checkpoint (weights only):
            # step/optimizer/best reset, LR schedule + curriculum start at 0.
            payload = self.load_weights_from(self.policy, init_from)
            self.console(f"[init] warm-started weights from {init_from} "
                         f"(source step {int(payload.get('step', 0)):,}); "
                         f"fresh optimizer, run starts at step 0")
        collector = self._make_collector(env, cur.phase(start_step), horizon)
        self.init_wandb()

        self._start_step = start_step
        self._start_it = start_it = start_step // steps_per_iter
        self.console(
            f"[train] run={self.run_name} device={self.device} amp={self.amp} | "
            f"iters={iters} (from {start_it}) envs={env.num_envs} horizon={horizon} "
            f"steps/iter={steps_per_iter} target_steps={iters * steps_per_iter:,}"
        )
        self.start_timer()

        global_step, prev_step = start_step, start_step
        for it in range(start_it, iters):
            events = cur.on_step(global_step, self.policy)

            # Linear LR decay to 0 over the run (canonical PPO; stabilizes late training
            # once the batch is large and the policy near-converged).
            lr_now = self.lr * (1.0 - it / iters) if self.anneal_lr else self.lr
            if self.anneal_lr:
                for group in self.opt.param_groups:
                    group["lr"] = lr_now

            if it > 0 and cur.should_rebuild_env(prev_step, global_step):
                # NB: do NOT close the old env here. gym_microrts shares one JVM
                # per process and JPype cannot restart it, so closing would kill
                # the JVM. We drop the old env (its games simply stop being
                # stepped) and build the new one in the same JVM.
                env = self._build_env(cur.env_config(global_step), cur.reward_weight_vec(global_step))
                collector = self._make_collector(env, events["phase"], horizon)

            # Anneal the reward shaping toward the win objective (no env rebuild).
            rw = cur.reward_weight_vec(global_step)
            self._shaper.set_weight(rw)

            # Refresh the frozen opponent from the pool each self-play rollout.
            if isinstance(collector, SelfPlayCollector):
                snap = cur.pool.sample()
                if snap is not None:
                    collector.opponent.load_state_dict(snap)

            buf = collector.collect()
            metrics = self._update(buf)

            log = {f"ppo/{k}": v for k, v in metrics.items()}
            log.update(self._rollout_stats(buf))
            log.update(self._entropy_guard(metrics.get("entropy", 0.0), env.action_nvec))
            log.update({"curriculum/phase_bot": float(events["phase"] == "bot"),
                        "curriculum/encoder_frozen": float(events["encoder_frozen"]),
                        "curriculum/pool_size": float(events["pool_size"]),
                        "ppo/lr": lr_now,
                        # Reward-anneal visibility: win weight vs total dense weight.
                        "reward/w_win": float(rw[0]),
                        "reward/w_dense_sum": float(sum(rw[1:]))})

            # Generic collapse guard: never continue on non-finite stats.
            self.guard_finite(metrics, global_step)
            if log["guards/entropy_collapse"] and self.guards_fatal:
                self.save_checkpoint(self.policy, self.opt, global_step, metrics, tag="collapse")
                raise CollapseError(f"entropy collapsed for {self.entropy_patience} iters")

            if eval_cfg.get("enabled", False) and it % eval_cfg.get("every_iters", 50) == 0:
                eval_metrics = self.evaluate(cur, eval_cfg)
                log.update(eval_metrics)
                is_best = self.record_eval(global_step, eval_metrics, self.policy, self.opt)
                self._print_eval(global_step, eval_metrics, is_best)

            # Periodic full checkpoint (model + optimizer + metrics).
            if it % self.ckpt_every == 0:
                self.save_checkpoint(self.policy, self.opt, global_step, metrics, tag="latest")
                self.save_checkpoint(self.policy, self.opt, global_step, metrics,
                                     tag=f"step_{global_step}")

            # Periodic console stats line (with ETA folding in eval cost).
            if it % console_every == 0 or it == iters - 1:
                self._print_stats(it, iters, global_step, steps_per_iter, log, events)

            self.log(log, step=global_step)

            prev_step = global_step
            global_step += horizon * env.num_envs

        self.save_checkpoint(self.policy, self.opt, global_step, tag="final")
        # Progress is measured since (re)start so sps/ETA stay accurate after a resume.
        prog = self.progress(iters - self._start_it, iters - self._start_it,
                             global_step - self._start_step)
        self.console(
            f"[done] {iters} iters, {global_step:,} steps in {prog['elapsed_str']} "
            f"({prog['sps']/1000:.1f}k sps) | best {self.monitor}={self._best_metric:+.4f} "
            f"| checkpoints -> {self.run_dir}"
        )
        self.finish()

    def _print_stats(self, it, iters, global_step, steps_per_iter, log, events) -> None:
        done_iters = it + 1 - self._start_it
        prog = self.progress(done_iters, iters - self._start_it, done_iters * steps_per_iter)
        self.console(
            f"[{it + 1:>5}/{iters}] step {global_step:>11,} | "
            f"{prog['sps'] / 1000:5.1f}k sps | el {prog['elapsed_str']} eta {prog['eta_str']} | "
            f"{events['phase']:>8} | "
            f"pg {log['ppo/policy_loss']:+.4f} vf {log['ppo/value_loss']:.4f} "
            f"ent {log['ppo/entropy']:.2f}({log['guards/entropy_frac'] * 100:.0f}%) "
            f"kl {log['ppo/approx_kl']:.4f} ev {log['rollout/explained_variance']:+.2f} "
            f"rew {log['rollout/reward_mean']:+.4f} gN {log['ppo/grad_norm']:.1f}"
        )

    def _print_eval(self, step, eval_metrics, is_best) -> None:
        parts = "  ".join(f"{k}={v:+.4f}" for k, v in eval_metrics.items())
        self.console(f"   >> EVAL @ step {step:,} | {parts}{'   [NEW BEST]' if is_best else ''}")

    @torch.no_grad()
    def evaluate(self, cur, eval_cfg) -> dict:
        """Play full games vs fixed bots and report win rate / draw rate / Elo.

        This is the *unhackable* performance signal: it scores actual game
        outcomes (win-loss reward), not the shaped reward the policy optimizes —
        so best-checkpoint selection can't be gamed by farming dense reward. The
        eval env is built once and reused (closing it would shut the shared JVM).
        """
        if self._eval_env is None:
            bots = list(cur.eval_bots)
            envs_per_bot = eval_cfg.get("envs_per_bot", 4)
            n = envs_per_bot * len(bots)
            self._eval_env = MicroRTSVecEnv(EnvConfig(
                num_envs=n, mode="bot", bots=tuple(bots),
                max_steps=eval_cfg.get("max_steps", cur.max_steps),
                reward_weight=reward_weight("win_loss"),
                gridnet=getattr(cur, "gridnet", False),
            ))
            self._eval_bots = [bots[i % len(bots)] for i in range(n)]

        stats = run_match_eval(
            self.policy, self._eval_env, self._eval_bots,
            games=eval_cfg.get("games", 5),
            max_steps=eval_cfg.get("max_steps", cur.max_steps),
            device=self.device, deterministic=eval_cfg.get("greedy", True),
        )
        o = build_report(stats)["overall"]
        return {"eval/win_rate": o["win_rate"], "eval/draw_rate": o["draw_rate"],
                "eval/elo": o["elo"]}

    # --- smoke test ------------------------------------------------------
    def smoke_test(self) -> dict:
        """One collect + one PPO update on a tiny env. No W&B, no checkpoint."""
        assert self.cfg is not None, "smoke_test() requires config-mode construction"
        from environments.microrts_env import EnvConfig

        gridnet = self.cfg.training.get("curriculum", {}).get("gridnet", False)
        env = MicroRTSVecEnv(EnvConfig(num_envs=4, max_steps=200, mode="bot",
                                       bots=("randomBiasedAI",), gridnet=gridnet))
        try:
            self._attach(self._build_policy(env))
            collector = Collector(env, self.policy, horizon=8, device=str(self.device))
            buf = collector.collect()
            metrics = self._update(buf)
            assert all(map(lambda v: v == v, metrics.values())), "non-finite smoke metrics"
            return metrics
        finally:
            env.close()
