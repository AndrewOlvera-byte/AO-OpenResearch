"""``SelfPlayCollector`` — fills a buffer from a self-play env using a frozen opponent.

In ``selfplay`` mode ``MicroRTSVecEnv`` returns the two players of each game as
consecutive rows: game ``i`` occupies rows ``2i`` (player 0, the *learner*) and
``2i+1`` (player 1, the *opponent*). We act the learner rows with the live policy
and the opponent rows with a frozen snapshot sampled from the :class:`OpponentPool`,
and only the learner rows are written to the buffer (so we optimize player 0 against
a fixed past self). The buffer therefore has ``num_envs // 2`` lanes.

Timing matches ``Collector``: observe ``obs_t``, choose ``action_t`` (store
value/logprob), step, store ``reward_t``/``done_t``.
"""

from __future__ import annotations

import torch

from .buffer import RolloutBuffer


class SelfPlayCollector:
    def __init__(self, env, learner, opponent, horizon, device="cpu"):
        assert env.num_envs % 2 == 0, "self-play needs paired envs"
        self.env, self.learner, self.opponent = env, learner, opponent
        self.horizon, self.device = horizon, device
        self.n_learn = env.num_envs // 2
        self._trans = env.reset()
        mask = self._trans.get("mask", None)
        mask_shape = tuple(mask[0::2].shape[1:]) if mask is not None else None
        with torch.no_grad():
            probe = learner.step(self._trans["obs"].to(device)[0::2],
                                 mask[0::2].to(device) if mask is not None else None)
        action_shape = tuple(probe["action"].shape[1:])
        self.buffer = RolloutBuffer(
            horizon, self.n_learn, env.obs_shape, action_shape, device, mask_shape=mask_shape,
        )

    @torch.no_grad()
    def collect(self, buffer: RolloutBuffer | None = None) -> RolloutBuffer:
        buf = self.buffer if buffer is None else buffer
        for t in range(self.horizon):
            obs = self._trans["obs"].to(self.device)
            learn_obs, opp_obs = obs[0::2], obs[1::2]
            mask = self._trans.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device)
            learn_mask = mask[0::2] if mask is not None else None
            opp_mask = mask[1::2] if mask is not None else None

            learn_out = self.learner.step(learn_obs, learn_mask)
            opp_out = self.opponent.step(opp_obs, opp_mask)

            # Trailing dims come from the head: (7,) single-unit or (H*W, 7) gridnet.
            actions = torch.empty(
                (self.env.num_envs, *learn_out["action"].shape[1:]),
                dtype=learn_out["action"].dtype, device=learn_out["action"].device,
            )
            actions[0::2] = learn_out["action"]
            actions[1::2] = opp_out["action"]

            self.env.send(actions)
            nxt = self.env.recv()
            buf.add(
                t, obs=learn_obs, action=learn_out["action"], logprob=learn_out["logprob"],
                value=learn_out["value"], reward=nxt["reward"][0::2], done=nxt["done"][0::2],
                mask=learn_mask,
            )
            self._trans = nxt

        last = self._trans["obs"].to(self.device)[0::2]
        last_mask = self._trans.get("mask", None)
        if last_mask is not None:
            last_mask = last_mask.to(self.device)[0::2]
        last_value = self.learner.step(last, last_mask)["value"]
        buf.compute_gae(last_value)
        return buf
