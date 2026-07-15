"""``OfflineCollector`` — drive one env with a policy and stream to the writer.

Timing matches :class:`~collectors.dream_collector.DreamCollector` (the world
model's data contract): at step ``t`` we observe ``obs_t``/``mask_t``/
``is_first_t``, the policy picks ``action_t``, we step the env to get
``reward_t``/``done_t``, and store ``(obs_t, action_t, mask_t, reward_t,
raw_rewards_t, done_t, is_first_t)``. So the stored action is the one that
*produced* the next frame — exactly how the dynamics model's shifted action
alignment and the future actor distillation both read it.

The env is a :class:`~environments.dream_env.DreamEnv` wrapping
:class:`~environments.microrts_env.MicroRTSVecEnv` (bot mode). gym_microrts
auto-resets internally, so a single ``collect`` call streams continuous
experience across episode boundaries; ``is_first`` (from ``DreamEnv``) marks the
resets so the writer's trajectory windows stay episode-aware without ever
blocking the JVM hot loop — every batch is handed off to the writer thread.

**Opponent action (format v3):** every stored step carries BOTH players'
gridnet actions. In bot mode it comes from the patched jar via the env's
``opponent_action`` (same tick as the submitted action). In self-play mode
(``selfplay_pairs=True``) gym_microrts exposes the two players of one game as
consecutive lanes — every lane is recorded as a player-1 trajectory whose
opponent action is simply its partner lane's own action (2N trajectories from
N games, both channels non-scripted).
"""

from __future__ import annotations

import torch


class OfflineCollector:
    def __init__(
        self,
        env,
        policy,
        writer,
        *,
        device="cpu",
        steps_per_segment=512,
        selfplay_pairs=False,
        counterfactual_frac=0.0,
    ):
        self.env, self.policy, self.writer = env, policy, writer
        self.device = device
        self.steps_per_segment = int(steps_per_segment)
        self.selfplay_pairs = bool(selfplay_pairs)
        self.counterfactual_frac = float(counterfactual_frac)
        if not 0.0 <= self.counterfactual_frac <= 1.0:
            raise ValueError("counterfactual_frac must be in [0,1]")
        self._cf_policy = None
        if self.counterfactual_frac > 0:
            from .policies import MaskedRandomPolicy

            self._cf_policy = MaskedRandomPolicy(env.action_nvec, device=device)
        if self.selfplay_pairs:
            assert env.num_envs % 2 == 0, "selfplay_pairs needs paired lanes"
        self._trans = env.reset()
        if not self.selfplay_pairs and "opponent_action" not in self._trans.keys():
            raise RuntimeError(
                "env does not surface opponent_action — format v3 collection needs "
                "the patched jar (infra/microrts-jar-patch/apply_patch.sh) and "
                "EnvConfig(opponent_action=True)"
            )

    @torch.no_grad()
    def collect(
        self, steps: int, *, map_id: int, opponent_id, policy_id=0, action_noise=0.0
    ) -> int:
        """Collect ``steps`` timesteps into the writer, tagged with map/opponent
        and the player-1 controller's provenance (``policy_id``/``action_noise``).

        ``opponent_id`` is a scalar or a per-lane array of length ``num_envs``
        (when lanes play different bots in one env). Segments the stream into
        ``steps_per_segment`` blocks so the writer's RAM buffer (it accumulates a
        segment before the lane-major flush) stays bounded. Returns the number of
        timesteps collected.
        """
        n = self.env.num_envs
        done_in_seg = 0
        for _ in range(steps):
            trans = self._trans
            obs = trans["obs"].to(self.device)
            mask = trans["mask"].to(self.device)
            is_first = trans.get("is_first", torch.zeros(n, dtype=torch.bool))
            action = self.policy.step(obs, mask)["action"]
            nxt = self.env.step(action)
            if self.selfplay_pairs:
                # Partner lane's action IS the opponent action (pairs (0,1),(2,3),..).
                opp_action = (
                    action.view(n // 2, 2, *action.shape[1:]).flip(1).reshape_as(action)
                )
            else:
                opp_action = nxt["opponent_action"]
            batch = {
                "obs": obs,
                "action": action,
                "opponent_action": opp_action,
                "mask": mask,
                "reward": nxt["reward"],
                "raw_rewards": nxt["raw_rewards"],
                "done": nxt["done"],
                "is_first": is_first,
            }
            if "full_state" in trans.keys():
                done = nxt["done"].bool()
                dstate = done.view(n, 1, 1)
                dglob = done.view(n, 1)
                # The patched Java client auto-resets done lanes. Use its sparse
                # pre-reset terminal snapshot as the true transition arrival.
                next_state = torch.where(
                    dstate, nxt["terminal_full_state"], nxt["full_state"]
                )
                next_globals = torch.where(
                    dglob, nxt["terminal_full_globals"], nxt["full_globals"]
                )
                batch.update(
                    {
                        "state": trans["full_state"],
                        "globals": trans["full_globals"],
                        "next_state": next_state,
                        "next_globals": next_globals,
                    }
                )
                if self._cf_policy is not None:
                    alt = self._cf_policy.step(obs, mask)["action"]
                    valid = torch.rand(n) < self.counterfactual_frac
                    cf_state, cf_globals = self.env.counterfactual(
                        alt, opp_action, valid
                    )
                    batch.update(
                        {
                            "counterfactual_action": alt,
                            "counterfactual_opponent_action": opp_action,
                            "counterfactual_next_state": cf_state,
                            "counterfactual_next_globals": cf_globals,
                            "counterfactual_valid": valid,
                        }
                    )
            # Envs that surface the true terminal arrival frame (autoreset
            # otherwise swallows it) get it stored sparsely at done rows —
            # requires the writer's ``store_terminal_obs=True``.
            if "terminal_obs" in nxt.keys():
                batch["terminal_obs"] = nxt["terminal_obs"]
            self.writer.add_batch(batch)
            self._trans = nxt
            done_in_seg += 1
            if done_in_seg >= self.steps_per_segment:
                self.writer.end_segment(
                    map_id=map_id,
                    opponent_id=opponent_id,
                    policy_id=policy_id,
                    action_noise=action_noise,
                )
                done_in_seg = 0
        if done_in_seg > 0:
            self.writer.end_segment(
                map_id=map_id,
                opponent_id=opponent_id,
                policy_id=policy_id,
                action_noise=action_noise,
            )
        return steps
