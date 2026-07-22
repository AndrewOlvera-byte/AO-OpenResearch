"""Dreamer/PMPO trainer for the frozen incomplete-information v2 belief model."""

from __future__ import annotations

import torch

from environments.dream_env import DreamEnv
from environments.microrts_env import EnvConfig, MicroRTSVecEnv
from evaluation.matchplay import build_report, run_match_eval
from models.incomplete_info import IncompleteBeliefAgentConfig, IncompleteBeliefDreamer
from rewards.rewards import reward_weight
from trainers.StructuredDreamerRLTrainer import StructuredDreamerRLTrainer


class IncompleteBeliefRLTrainer(StructuredDreamerRLTrainer):
    """Reuse structured Dreamer's heads, PMPO, W&B, eval, and checkpoints.

    The only replacements are policy construction, real-history belief encoding,
    and imagined rollout generation.  The pretrained v2 world model is immutable.
    """

    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        if self.mode != "imagination":
            raise ValueError("incomplete-belief RL currently supports imagination mode only")
        self._world_updates_enabled = False

    def build_policy(self, env):
        values = {k: v for k, v in self.cfg.model.items() if k != "type"}
        cfg = IncompleteBeliefAgentConfig.from_dict(values)
        policy = IncompleteBeliefDreamer(
            env.action_nvec, cfg, device=str(self.device)
        )
        self.grad_clip = cfg.grad_clip
        source = policy.source_checkpoint
        self.console(
            f"[incomplete-belief-rl] world={cfg.belief_checkpoint} "
            f"step={source.get('step')} flow_steps={cfg.flow_steps} "
            f"intent_sampling={cfg.sample_intent}"
        )
        return policy

    def encode_batch(self, policy, batch):
        # Phase-2 heads use the frozen teacher latent.  Imagination itself is
        # seeded from fog history in ``imagine_rollout`` and never sees this z.
        with torch.no_grad(), self._amp_ctx():
            return policy.tokenizer.encode(
                batch["full_state"].to(self.device),
                batch["full_globals"].to(self.device),
            )

    def imagine_rollout(self, policy, z, batch):
        if batch.shape[0] > self.imagine_seeds:
            index = torch.randperm(batch.shape[0], device=batch.device)[:self.imagine_seeds]
            batch = batch[index]
        with self._amp_ctx():
            return policy.imagine_from_batch(batch, self.imagine_horizon)

    @torch.no_grad()
    def evaluate(self, cfg):
        if self._eval_env is None:
            bots = list(cfg.get("bots", self.bots))
            per = int(cfg.get("envs_per_bot", 2))
            n = per * len(bots)
            self._eval_env = DreamEnv(MicroRTSVecEnv(EnvConfig(
                num_envs=n,
                mode="bot",
                bots=tuple(bots),
                map_path=self.map_path,
                max_steps=int(cfg.get("max_steps", self.max_steps)),
                reward_weight=reward_weight("win_loss"),
                gridnet=True,
                full_state=False,
            )))
            self._eval_bots = [bots[i % len(bots)] for i in range(n)]
        # Evaluation starts a fresh policy history; otherwise recurrent state
        # from collection would leak across independent JVM environments.
        self.policy._online_obs = None
        self.policy._online_action = None
        self.policy._online_first = None
        stats = run_match_eval(
            self.policy,
            self._eval_env,
            self._eval_bots,
            games=int(cfg.get("games", 3)),
            max_steps=int(cfg.get("max_steps", self.max_steps)),
            device=self.device,
            deterministic=bool(cfg.get("greedy", True)),
        )
        overall = build_report(stats)["overall"]
        # Collection uses the same policy object, so force a clean context when
        # control returns to the collector.
        self.policy._online_obs = None
        self.policy._online_action = None
        self.policy._online_first = None
        return {"eval/win_rate": overall["win_rate"], "eval/elo": overall["elo"]}

