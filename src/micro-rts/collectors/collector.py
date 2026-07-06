"""``Collector`` — fills a ``RolloutBuffer`` by stepping a ``VecEnv`` with a ``Policy``.

``collect`` fills either its own buffer (no-arg, the simple path) or an explicit
buffer passed in — the latter lets a double-buffered runner ping-pong two buffers so
the trainer reads one while the collector (on a background thread) fills the other.

The collector keeps the env's running state (``_trans``) across calls, so successive
``collect`` calls continue the same episodes. Only one ``collect`` may run at a time
(the env is single-threaded); overlap happens between ``collect`` and the *trainer*,
not between two collects.

Timing convention: at step t we observe ``obs_t``, choose ``action_t`` (storing its
value/logprob), then step the env to obtain ``reward_t``/``done_t`` and ``obs_{t+1}``.
"""

from __future__ import annotations

import torch

from .buffer import RolloutBuffer


class Collector:
    def __init__(self, env, policy, horizon, device="cpu"):
        self.env, self.policy, self.horizon, self.device = env, policy, horizon, device
        self._trans = env.reset()
        mask = self._trans.get("mask", None)
        mask_shape = tuple(mask.shape[1:]) if mask is not None else None
        # Probe the per-env action shape from the policy — (8,) single-unit, (H*W, 7)
        # gridnet — so the buffer allocates correctly for either head.
        with torch.no_grad():
            probe = policy.step(self._trans["obs"].to(device),
                                mask.to(device) if mask is not None else None)
        action_shape = tuple(probe["action"].shape[1:])
        self.buffer = RolloutBuffer(
            horizon, env.num_envs, env.obs_shape, action_shape, device, mask_shape=mask_shape,
        )

    @torch.no_grad()
    def collect(self, buffer: RolloutBuffer | None = None) -> RolloutBuffer:
        buf = self.buffer if buffer is None else buffer
        for t in range(self.horizon):
            obs = self._trans["obs"].to(self.device)
            mask = self._trans.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device)
            out = self.policy.step(obs, mask)
            self.env.send(out["action"])
            nxt = self.env.recv()
            buf.add(
                t, obs=obs, action=out["action"], logprob=out["logprob"],
                value=out["value"], reward=nxt["reward"], done=nxt["done"], mask=mask,
            )
            self._trans = nxt
        last_obs = self._trans["obs"].to(self.device)
        last_mask = self._trans.get("mask", None)
        if last_mask is not None:
            last_mask = last_mask.to(self.device)
        last_value = self.policy.step(last_obs, last_mask)["value"]
        buf.compute_gae(last_value)
        return buf
