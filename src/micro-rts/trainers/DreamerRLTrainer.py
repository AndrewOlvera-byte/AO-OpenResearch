"""``DreamerRLTrainer`` — the MicroRTS implementation of the DreamerV4 RL loop.

The env-agnostic algorithm (collect -> learn world model -> imagine -> improve
policy) lives in ``shared.dreamerv4.trainer.AbstractDreamerTrainer``. This class
supplies the MicroRTS specifics by implementing that abstract's hooks:

- ``build_env`` — an in-process ``MicroRTSVecEnv`` (gridnet action space) wrapped
  in ``DreamEnv`` for the ``is_first`` collation the world model needs,
- ``build_policy`` — the registered ``dreamerv4`` model built for the env's
  spaces (freezing per ``model.freeze`` for pretrain-then-RL staging),
- ``build_collector`` — a ``DreamCollector`` filling the sequence replay buffer
  and maintaining a live ``WorldModelMemory`` context window,
- ``compute_world_loss`` — tokenizer recon + predicted-mask BCE + arrive-aligned
  reward/continue, plus the Dreamer 4 shortcut-forcing flow loss,
- ``imagine_rollout`` — **context-seeded** imagination: each sampled replay
  sequence contributes its trailing ``imagine_context`` frames (latents, actions,
  is_first) as world-model history before the actor takes over,
- ``compute_actor_critic_losses`` — imagined lambda-return critic + REINFORCE
  actor with DreamerV3 percentile return normalization,
- ``analyze_dynamics`` — periodic world-model health metrics (open-loop latent
  error vs a copy-last-frame baseline, reward correlation, predicted-mask
  accuracy, imagined latent motion) for smoke/dynamics-analysis runs,
- eval vs scripted bots, and the console / W&B / checkpoint lifecycle hooks (via
  the project's ``BaseTrainer``).
"""

from __future__ import annotations

import contextlib
import glob

import torch

from environments.microrts_env import EnvConfig, MicroRTSVecEnv
from environments.dream_env import DreamEnv
from collectors.dream_collector import DreamCollector, DreamLeagueCollector
from evaluation.matchplay import build_report, run_match_eval
from loss.dreamer import (
    ReturnNormalizer, actor_critic_losses, cell_weights, dynamics_loss,
    mask_actions_to_sources, real_actor_critic_losses,
)
from collectors.offline_data import build_mrts_loader, cycle, to_device
from rewards.rewards import reward_weight

from core.registry import build

# Registry side effect: make the ``dreamerv4`` model type buildable.
import models.dreamer  # noqa: F401
from models.dreamer.memory import WorldModelMemory

from shared.dreamerv4.trainer import AbstractDreamerTrainer
from trainers.BaseTrainer import BaseTrainer


MODES = ("imagination", "hybrid", "online")


class DreamerRLTrainer(BaseTrainer, AbstractDreamerTrainer):
    def __init__(self, cfg, device: str | None = None) -> None:
        BaseTrainer.__init__(self, cfg, device)
        d = cfg.training.get("dreamer", {})
        self.configure(d)
        # Where the actor/critic gradient comes from:
        #   imagination — imagined rollouts on a FROZEN pretrained world model
        #                 (the full Dreamer 4 pretrain-then-RL recipe),
        #   hybrid      — imagined rollouts while the world model keeps training
        #                 on the fresh replay (classic online Dreamer),
        #   online      — real replay sequences, no world model in the loop
        #                 (the model-free sample-efficiency baseline).
        self.mode = d.get("mode", "hybrid")
        if self.mode not in MODES:
            raise ValueError(f"training.dreamer.mode must be one of {MODES}, got {self.mode!r}")
        self.init_from = d.get("init_from", None)   # pretrained ckpt to warm-start from
        self.amp = d.get("amp", False)              # bf16 autocast on CUDA
        self.buffer_device = d.get("buffer_device", "cpu")
        self.use_flow = d.get("use_flow", True)
        self.flow_coef = d.get("flow_coef", 1.0)
        self.flow_self_frac = d.get("flow_self_frac", 0.25)
        self.analyze_every = d.get("analyze_every", 0)   # 0 = off
        self.opp_bc_coef = float(d.get("opp_bc_coef", 0.3))
        self.anchor_cfg = d.get("anchor", {}) or {}
        self.health_gate = d.get("health_gate", {}) or {}
        self._anchor_iter = None
        self._dynamics_gate_open = not bool(self.health_gate.get("enabled", False))
        self._gate_checks = 0
        self.iters = cfg.training.get("iters", 1000)
        self.num_envs = d.get("num_envs", 16)
        self.map_path = d.get("map_path", "maps/16x16/basesWorkers16x16.xml")
        self.max_steps = d.get("max_steps", 2000)
        self.bots = tuple(d.get("bots", ("randomBiasedAI",)))
        self.reward_preset = d.get("reward", "dense_shaped")
        # Bot/self-play league collection (mirrors the PPO league recipe):
        # alternate bot blocks and self-play blocks vs an OpponentPool of frozen
        # past selves. Only affects where real transitions come from.
        self.league = d.get("league", {}) or {}
        self.return_norm = None
        self._collector = None
        self._eval_env = None

    def _amp_ctx(self):
        """bf16 autocast on CUDA when enabled; a no-op otherwise. bf16 needs no
        GradScaler and halves activation VRAM (see PPOTrainer._amp_ctx).

        ``cache_enabled=False``: the AC losses run the critic under ``no_grad``
        (values/targets) before the grad-requiring logits pass; autocast's weight
        cache would reuse the no-grad bf16 cast and silently detach the critic
        graph ("element 0 of tensors does not require grad")."""
        if self.amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                  cache_enabled=False)
        return contextlib.nullcontext()

    # --- AbstractDreamerTrainer hooks: build ----------------------------
    def build_env(self):
        base = MicroRTSVecEnv(EnvConfig(
            num_envs=self.num_envs, map_path=self.map_path, max_steps=self.max_steps,
            mode="bot", bots=self.bots, reward_weight=reward_weight(self.reward_preset),
            gridnet=True, opponent_action=(self.mode == "hybrid"),
        ))
        # The env's weighted scalar reward is exactly what the world model's reward
        # head predicts, so shaping is folded into the env weights (no wrapper).
        return DreamEnv(base)

    def build_policy(self, env):
        model_cfg = {k: v for k, v in self.cfg.model.items() if k != "type"}
        policy = build("model", type=self.cfg.model.get("type", "dreamerv4"),
                       obs_shape=env.obs_shape, action_nvec=env.action_nvec,
                       device=str(self.device), **model_cfg)
        if self.init_from:
            self._load_init(policy)
        # Imagination mode means "the pretrained world model IS the environment":
        # a trainable tokenizer/world model there would silently turn the run into
        # hybrid, so the toggle enforces the freeze rather than trusting the yaml.
        if self.mode == "imagination":
            fz = policy.cfg.freeze
            if not (fz.tokenizer and fz.world_model):
                self.console("[dream] mode=imagination: forcing tokenizer + world_model frozen")
                fz.tokenizer = fz.world_model = True
                policy._apply_freeze()
        self.grad_clip = policy.cfg.grad_clip
        ac = policy.cfg.actor_critic
        if ac.return_norm:
            self.return_norm = ReturnNormalizer(
                rate=ac.return_norm_rate, limit=ac.return_norm_limit,
                low=ac.return_norm_low, high=ac.return_norm_high)
        return policy

    def _load_init(self, policy) -> None:
        """Warm-start from a pretraining checkpoint (``train_dreamer_dynamics``
        format: ``{"model": state_dict, ...}`` — includes the frozen tokenizer and
        the world model's ``latent_scale`` buffer). Arch must match the yaml."""
        ckpt = torch.load(self.init_from, map_location=self.device)
        sd = ckpt.get("model", ckpt)
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        # The pretrain phases never touch the action expert, so its params being
        # "missing" from the checkpoint is expected; anything else is an arch drift.
        stray = [k for k in missing if not k.startswith("action_expert.")]
        if stray or unexpected:
            raise RuntimeError(
                f"init_from={self.init_from}: state dict mismatch — the model: block "
                f"must be identical to the pretrain config. missing={stray[:6]} "
                f"unexpected={list(unexpected)[:6]}")
        self.console(f"[dream] init_from={self.init_from} "
                     f"(phase={ckpt.get('phase')}, step={ckpt.get('step')}, "
                     f"latent_scale={float(policy.world_model.latent_scale):.4f})")

    def build_collector(self, env, policy):
        memory = WorldModelMemory(policy.cfg.actor_critic.imagine_context)
        # Terminal splice feeds arrive-aligned cont=0 / win-reward rows to the
        # world-model loss — only wanted when the world model actually trains on
        # the buffer; in online mode the spliced (obs, action) pair would corrupt
        # the on-policy actor loss instead.
        splice = self.mode == "hybrid"
        common = dict(capacity=self.buffer_capacity, device=str(self.device),
                      memory=memory, storage_device=self.buffer_device,
                      terminal_splice=splice)
        if self.league.get("enabled", False):
            sp_env = DreamEnv(MicroRTSVecEnv(EnvConfig(
                num_envs=2 * env.num_envs, mode="selfplay", map_path=self.map_path,
                max_steps=self.max_steps,
                reward_weight=reward_weight(self.reward_preset), gridnet=True)))
            self._collector = DreamLeagueCollector(
                env, sp_env, policy, self._build_opponent(env), self.horizon,
                bot_steps=self.league.get("bot_steps", 0),
                mix_bot_block=self.league.get("mix_bot_block", 0),
                mix_selfplay_block=self.league.get("mix_selfplay_block", 0),
                snapshot_every=self.league.get("snapshot_every", 250_000),
                pool_capacity=self.league.get("pool_capacity", 8),
                recency_bias=self.league.get("recency_bias", 2.0),
                **common)
        else:
            self._collector = DreamCollector(env, policy, self.horizon, **common)
        return self._collector

    def _build_opponent(self, env):
        """Frozen self-play opponent: same arch, warm-started like the learner
        (its tokenizer must be sensible before the first pool snapshot lands);
        stepping only uses tokenizer.encode -> action_expert.act, never the WM."""
        model_cfg = {k: v for k, v in self.cfg.model.items() if k != "type"}
        opp = build("model", type=self.cfg.model.get("type", "dreamerv4"),
                    obs_shape=env.obs_shape, action_nvec=env.action_nvec,
                    device=str(self.device), **model_cfg)
        if self.init_from:
            ckpt = torch.load(self.init_from, map_location=self.device)
            opp.load_state_dict(ckpt.get("model", ckpt), strict=False)
        opp.requires_grad_(False)
        opp.eval()
        return opp

    # --- AbstractDreamerTrainer hooks: losses ---------------------------
    def _to_dev(self, batch):
        keys = ("obs", "action", "reward", "cont", "mask", "is_first",
                "opponent_action", "opponent_valid")
        keys = tuple(k for k in keys if k in batch.keys())
        b = {k: batch[k].to(self.device) for k in keys}
        # v4.2: replay actions are raw GridNet output over all cells; NOOP the
        # ~97% the engine never executed before any world-model consumer (see
        # loss.dreamer.mask_actions_to_sources). The actor-loss path is fine
        # either way: evaluate() scores under the engine mask.
        if getattr(self.policy.cfg.dynamics, "mask_junk_actions", False):
            b["action"] = mask_actions_to_sources(b["action"], b["obs"])
        return b

    def compute_world_loss(self, policy, batch):
        b = self._to_dev(batch)
        with self._amp_ctx():
            z = policy.tokenizer.encode(b["obs"])
            cw = cell_weights(
                b["obs"],
                occ_boost=policy.cfg.dynamics.cell_occ_boost,
                changed_boost=policy.cfg.dynamics.cell_changed_boost,
                floor=policy.cfg.dynamics.cell_weight_floor,
                downsample=policy.tokenizer.cfg.downsample)
            opp = b.get("opponent_action")
            opp_valid = b.get("opponent_valid")
            loss, metrics = dynamics_loss(
                policy, z, b["action"], b["reward"], b["cont"], b["is_first"],
                opponent_action=opp, opponent_valid=opp_valid,
                cell_weight=cw, obs=b["obs"], flow_coef=self.flow_coef if self.use_flow else 0.0,
                opp_bc_coef=self.opp_bc_coef, self_frac=self.flow_self_frac)

            # Rehearse the diverse offline joint-action corpus during online
            # finetuning.  This anchors the conditional model and BC head while
            # the live distribution moves with the league.
            if self._anchor_iter is not None:
                ab = to_device(next(self._anchor_iter), self.device)
                if getattr(policy.cfg.dynamics, "mask_junk_actions", False):
                    ab["action"] = mask_actions_to_sources(ab["action"], ab["obs"])
                with torch.no_grad():
                    az = policy.tokenizer.encode(ab["obs"])
                acw = cell_weights(
                    ab["obs"], occ_boost=policy.cfg.dynamics.cell_occ_boost,
                    changed_boost=policy.cfg.dynamics.cell_changed_boost,
                    floor=policy.cfg.dynamics.cell_weight_floor,
                    downsample=policy.tokenizer.cfg.downsample)
                aloss, am = dynamics_loss(
                    policy, az, ab["action"], ab["reward"], ab["cont"], ab["is_first"],
                    opponent_action=ab["opponent_action"],
                    opponent_valid=torch.ones_like(ab["is_first"], dtype=torch.bool),
                    cell_weight=acw, obs=ab["obs"], flow_coef=self.flow_coef,
                    opp_bc_coef=self.opp_bc_coef, self_frac=self.flow_self_frac)
                coef = float(self.anchor_cfg.get("coef", 0.5))
                loss = loss + coef * aloss
                metrics.update({f"anchor/{k.replace('/', '_')}": v for k, v in am.items()})
                metrics["anchor/coef"] = coef
        return loss, metrics, z

    def encode_batch(self, policy, batch):
        with torch.no_grad(), self._amp_ctx():
            return policy.tokenizer.encode(batch["obs"].to(self.device))

    def imagine_rollout(self, policy, z, batch):
        """Seed imagination with a trailing context window from each replay
        sequence — the transformer world model's memory — instead of bare frames."""
        with self._amp_ctx():
            if batch is None:
                return policy.imagine(z[:, -1], self.imagine_horizon)
            Tc = max(1, min(policy.cfg.actor_critic.imagine_context, z.shape[1]))
            ctx_action = batch["action"].to(self.device)[:, -Tc:]
            if getattr(policy.cfg.dynamics, "mask_junk_actions", False):
                ctx_action = mask_actions_to_sources(
                    ctx_action, batch["obs"].to(self.device)[:, -Tc:])
            return policy.imagine(
                z[:, -Tc:], self.imagine_horizon,
                ctx_action=ctx_action,
                ctx_is_first=batch["is_first"].to(self.device)[:, -Tc:],
                ctx_opponent_action=(batch["opponent_action"].to(self.device)[:, -Tc:]
                                     if "opponent_action" in batch.keys() and
                                     ("opponent_valid" not in batch.keys() or
                                      bool(batch["opponent_valid"][:, -Tc:].all()))
                                     else None),
            )

    def compute_actor_critic_losses(self, policy, imagined):
        ac = policy.cfg.actor_critic
        with self._amp_ctx():
            return actor_critic_losses(policy, imagined, gamma=ac.gamma, lam=ac.lam,
                                       entropy_coef=ac.entropy_coef,
                                       return_normalizer=self.return_norm)

    def actor_critic_update(self, policy, opts, z, batch=None):
        """Online mode swaps the actor/critic gradient source: real replay
        sequences (env rewards, engine masks, ``cont`` cutting the bootstrap)
        instead of imagined rollouts. Imagination modes defer to the base loop."""
        if self.mode != "online" and not self._dynamics_gate_open:
            return {"gate/actor_paused": 1.0}
        if self.mode != "online":
            return super().actor_critic_update(policy, opts, z, batch)
        ac = policy.cfg.actor_critic
        b = self._to_dev(batch)
        with self._amp_ctx():
            actor_loss, critic_loss, metrics = real_actor_critic_losses(
                policy, z, b, gamma=ac.gamma, lam=ac.lam,
                entropy_coef=ac.entropy_coef, return_normalizer=self.return_norm)
        if "actor" in opts:
            metrics["ac/actor_grad_norm"] = self._optimize(
                actor_loss, opts["actor"], self.grad_clip, retain_graph="critic" in opts)
        if "critic" in opts:
            metrics["ac/critic_grad_norm"] = self._optimize(
                critic_loss, opts["critic"], self.grad_clip)
            policy.action_expert.update_target()
        return metrics

    # --- dynamics analysis ------------------------------------------------
    @torch.no_grad()
    def analyze_dynamics(self, policy, batch=None) -> dict:
        """World-model health check on a replay batch: is the generative dynamics
        actually predicting motion (vs copying the last frame), does the reward
        head track the env reward, and is the predicted action mask usable?"""
        if batch is None:
            buf = self._collector.buffer if self._collector else None
            if buf is None or not buf.can_sample(self.seq_len):
                return {}
            batch = buf.sample(min(self.batch_seqs, 8), self.seq_len)
        b = self._to_dev(batch)
        z = policy.tokenizer.encode(b["obs"])
        B, T = z.shape[:2]
        ctx_len = max(1, min(policy.cfg.actor_critic.imagine_context, T - 1))

        # Open-loop generation with the real actions vs the real encoded latents.
        opp = b.get("opponent_action")
        if "opponent_valid" in b and not bool(b["opponent_valid"].all()):
            opp = None
        z_pred = policy.open_loop(z, b["action"], b["is_first"], context=ctx_len,
                                  opponent_action=opp)
        z_tgt = z[:, ctx_len:]
        openloop_mse = torch.nn.functional.mse_loss(z_pred, z_tgt)
        copylast_mse = torch.nn.functional.mse_loss(
            z[:, ctx_len - 1:-1].expand_as(z_tgt), z_tgt)
        z_pred_n = policy.world_model.normalize(z_pred)
        z_tgt_n = policy.world_model.normalize(z_tgt)
        copy_n = policy.world_model.normalize(z[:, ctx_len - 1:-1].expand_as(z_tgt))

        # Reward head correlation (arrive-aligned) with the env reward.
        ctx = policy.world_model.contextualize(
            z, b["action"], b["is_first"], opponent_action=opp)
        pred_r = ctx["reward"][:, 1:].flatten().float()
        true_r = b["reward"][:, :-1].flatten().float()
        if pred_r.std() > 1e-6 and true_r.std() > 1e-6:
            corr = float(torch.corrcoef(torch.stack([pred_r, true_r]))[0, 1])
        else:
            corr = 0.0

        # Predicted action-mask accuracy on generated latents (what imagination uses).
        mask_pred = torch.sigmoid(policy.tokenizer.decode_mask(z_pred)) > 0.5
        mask_acc = (mask_pred == b["mask"][:, ctx_len:].bool()).float().mean()
        mask_true = b["mask"][:, ctx_len:].bool()
        tp = (mask_pred & mask_true).sum().float()
        precision = tp / (mask_pred.sum().float().clamp_min(1.0))
        recall = tp / (mask_true.sum().float().clamp_min(1.0))

        # Imagined latent motion: a healthy generative dynamics moves.
        imagined = self.imagine_rollout(policy, z, batch)
        motion = (imagined["z"][:, 1:] - imagined["z"][:, :-1]).abs().mean()

        out = {
            "dyn/openloop_mse": float(openloop_mse),
            "dyn/copylast_mse": float(copylast_mse),
            "dyn/openloop_nmse": float(torch.nn.functional.mse_loss(z_pred_n, z_tgt_n)),
            "dyn/copylast_nmse": float(torch.nn.functional.mse_loss(copy_n, z_tgt_n)),
            "dyn/reward_corr": corr,
            "dyn/mask_acc": float(mask_acc),
            "dyn/mask_precision": float(precision),
            "dyn/mask_recall": float(recall),
            "dyn/z_motion": float(motion),
        }
        # Counterfactual self-action probe (the v3 go/no-go signal). The replay
        # buffer stores no opponent stream, so only the self channel is probed;
        # a ~0 gap means the actor is being trained inside action-insensitive
        # mush and the run should be stopped.
        from entrypoints.probes import counterfactual_action_probe

        out.update(counterfactual_action_probe(
            policy, z, b["action"], b.get("opponent_action"), b["is_first"],
            context=ctx_len))
        return out

    def _update_health_gate(self, dyn: dict) -> dict:
        if not self.health_gate.get("enabled", False):
            return {}
        self._gate_checks += 1
        ratio = dyn["dyn/openloop_mse"] / max(dyn["dyn/copylast_mse"], 1e-8)
        checks = {
            "copy": ratio <= float(self.health_gate.get("max_copy_ratio", 1.0)),
            "self": dyn.get("probe/self_gap_issued", 0.0) >=
                    float(self.health_gate.get("min_self_gap_issued", 0.02)),
            "precision": dyn.get("dyn/mask_precision", 0.0) >=
                         float(self.health_gate.get("min_mask_precision", 0.8)),
            "recall": dyn.get("dyn/mask_recall", 0.0) >=
                      float(self.health_gate.get("min_mask_recall", 0.8)),
        }
        warm = int(self.health_gate.get("warmup_checks", 1))
        self._dynamics_gate_open = self._gate_checks >= warm and all(checks.values())
        return {"gate/open": float(self._dynamics_gate_open),
                "gate/openloop_copy_ratio": ratio,
                **{f"gate/pass_{k}": float(v) for k, v in checks.items()}}

    # --- lifecycle hooks (console / W&B / eval / checkpoint) ------------
    def on_train_start(self, env, policy):
        self.policy = policy
        if self.anchor_cfg.get("enabled", False):
            matches = sorted(glob.glob(str(self.anchor_cfg.get("data", ""))))
            if not matches:
                raise FileNotFoundError(
                    f"dreamer.anchor.data matched no files: {self.anchor_cfg.get('data')!r}")
            loader = build_mrts_loader(
                matches[-1], task="dynamics", seq_len=self.seq_len,
                batch_size=int(self.anchor_cfg.get("batch_seqs", self.batch_seqs)),
                num_workers=int(self.anchor_cfg.get("num_workers", 2)), shuffle=True,
                locking=False)
            self._anchor_iter = cycle(loader)
            self.console(f"[dream] anchor replay={matches[-1]} coef={self.anchor_cfg.get('coef', 0.5)}")
        self.init_wandb()
        frozen = [n for n in ("tokenizer", "world_model", "actor", "critic")
                  if getattr(policy.cfg.freeze, n)]
        self.console(f"[dream] run={self.run_name} mode={self.mode} device={self.device} "
                     f"amp={'bf16' if self.amp else 'off'} buffer={self.buffer_device} | "
                     f"iters={self.iters} envs={env.num_envs} horizon={self.horizon} "
                     f"seq_len={self.seq_len} wm_updates={self.wm_updates}"
                     + (f" | frozen={','.join(frozen)}" if frozen else ""))
        self.start_timer()

    def on_step_end(self, it, global_step, policy, log):
        self.guard_finite({k: v for k, v in log.items() if "grad" not in k}, global_step)
        if self._collector is not None:
            log.update(self._collector.pop_episode_stats())
        if isinstance(self._collector, DreamLeagueCollector):
            log["league/selfplay"] = float(self._collector.phase() == "selfplay")
            log["league/pool_size"] = float(self._collector.pool_size)
        if self.analyze_every and it % self.analyze_every == 0:
            dyn = self.analyze_dynamics(policy)
            log.update(dyn)
            log.update(self._update_health_gate(dyn))
        eval_cfg = self.cfg.training.get("eval", {})
        if eval_cfg.get("enabled", False) and it % eval_cfg.get("every_iters", 50) == 0:
            em = self.evaluate(eval_cfg)
            log.update(em)
            is_best = self.record_eval(global_step, em, policy, None)
            self._print_eval(global_step, em, is_best)
        if it % self.ckpt_every == 0:
            self.save_checkpoint(policy, None, global_step, log, tag="latest")
        if it % self.cfg.training.get("console", {}).get("every_iters", 10) == 0:
            self._print_stats(it, global_step, log)
        self.log(log, step=global_step)

    def on_train_end(self, global_step, policy):
        self.save_checkpoint(policy, None, global_step, tag="final")
        self.console(f"[done] {self.iters} iters, {global_step:,} steps -> {self.run_dir}")
        self.finish()

    # --- entrypoints -----------------------------------------------------
    def train(self):
        self.run(self.iters)

    def _print_stats(self, it, step, log):
        self.console(
            f"[{it + 1:>5}/{self.iters}] step {step:>11,} | "
            f"recon {log.get('wm/recon', 0):.3f} rew {log.get('wm/reward', 0):.3f} "
            f"flow {log.get('flow/matching', 0):.3f} sc {log.get('flow/consistency', 0):.3f} | "
            f"actor {log.get('ac/actor_loss', 0):+.3f} critic {log.get('ac/critic_loss', 0):.3f} "
            f"ret {log.get('ac/return_mean', 0):+.3f} ent {log.get('ac/entropy', 0):.2f}"
            + (f" | ol {log['dyn/openloop_mse']:.4f} corr {log['dyn/reward_corr']:.2f}"
               if "dyn/openloop_mse" in log else "")
            + (f" | env ret {log['collect/ep_return']:+.2f} "
               f"W/L/T {log.get('collect/wins', 0):.0f}/{log.get('collect/losses', 0):.0f}"
               f"/{log.get('collect/timeouts', 0):.0f}"
               if "collect/ep_return" in log else "")
            + (f" | {'selfplay' if log['league/selfplay'] else 'bot':>8}"
               f" pool {log['league/pool_size']:.0f}"
               if "league/selfplay" in log else "")
        )

    def _print_eval(self, step, em, is_best):
        parts = "  ".join(f"{k}={v:+.4f}" for k, v in em.items())
        self.console(f"   >> EVAL @ {step:,} | {parts}{'   [NEW BEST]' if is_best else ''}")

    @torch.no_grad()
    def evaluate(self, eval_cfg):
        if self._eval_env is None:
            bots = list(eval_cfg.get("bots", self.bots))
            envs_per_bot = eval_cfg.get("envs_per_bot", 4)
            n = envs_per_bot * len(bots)
            self._eval_env = MicroRTSVecEnv(EnvConfig(
                num_envs=n, mode="bot", bots=tuple(bots), map_path=self.map_path,
                max_steps=eval_cfg.get("max_steps", self.max_steps),
                reward_weight=reward_weight("win_loss"), gridnet=True,
            ))
            self._eval_bots = [bots[i % len(bots)] for i in range(n)]
        stats = run_match_eval(
            self.policy, self._eval_env, self._eval_bots,
            games=eval_cfg.get("games", 5),
            max_steps=eval_cfg.get("max_steps", self.max_steps),
            device=self.device, deterministic=eval_cfg.get("greedy", True))
        o = build_report(stats)["overall"]
        return {"eval/win_rate": o["win_rate"], "eval/draw_rate": o["draw_rate"], "eval/elo": o["elo"]}

    # --- smoke test ------------------------------------------------------
    def smoke_test(self):
        """One collect + WM update + actor-critic update + dynamics analysis on a
        tiny env — verifies the whole model-based loop end to end."""
        base = MicroRTSVecEnv(EnvConfig(num_envs=4, max_steps=200, mode="bot",
                                        bots=("randomBiasedAI",), gridnet=True))
        env = DreamEnv(base)
        try:
            policy = self.build_policy(env)
            self.policy = policy
            opts = policy.build_optimizers()
            self.horizon, self.seq_len, self.batch_seqs = 8, 6, 4
            self.imagine_horizon, self.imagine_seeds = 4, 8
            collector = self.build_collector(env, policy)
            buf = collector.collect()
            batch = buf.sample(self.batch_seqs, self.seq_len)
            if "world" in opts:
                wm_metrics, z = self.world_update(policy, opts, batch)
            else:
                wm_metrics, z = {}, self.encode_batch(policy, batch)
            ac_metrics = self.actor_critic_update(policy, opts, z, batch)
            dyn = self.analyze_dynamics(policy, batch)
            out = {**wm_metrics, **ac_metrics, **dyn}
            assert all(v == v for v in out.values()), "non-finite smoke metrics"
            return out
        finally:
            env.close()
