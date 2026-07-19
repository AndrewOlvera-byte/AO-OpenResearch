"""``SequenceReplayBuffer`` — ring buffer of ``(T_cap, N)`` transitions for Dreamer.

The PPO ``RolloutBuffer`` stores a single on-policy ``[T, N]`` window and throws it
away after one update. Dreamer trains its world model on **sampled sub-sequences**
of past experience, so this buffer keeps a rolling history and samples contiguous
``(B, T)`` chunks (per env) from it.

Collation differences vs the PPO buffer:

- stores the **engine action mask** (for reconstruction/BCE + the action head),
- stores **is_first** (episode-start flags) and **cont** (``1 - done`` continue
  targets) instead of value/logprob/advantage — Dreamer's values come from
  imagination, not from stored rollouts,
- indexed as a ring by an absolute write counter, so sampling always draws from the
  most recent ``T_cap`` steps.

Memory: observations are one-hot planes, so they are stored as ``uint8`` and cast
back to float on sample; ``storage_device`` keeps the ring itself off the GPU
(default ``"cpu"`` — at RL-scale capacities the obs+mask rings are gigabytes, far
too big to pin in VRAM), while ``device`` is where sampled batches land.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


class SequenceReplayBuffer:
    def __init__(self, capacity, num_envs, obs_shape, action_shape, mask_shape,
                 device="cpu", storage_device="cpu", state_shape=None,
                 globals_shape=None):
        self.T_cap, self.N = int(capacity), int(num_envs)
        self.device = device                    # where sampled batches go
        self.storage_device = storage_device    # where the ring lives
        self.count = 0
        fields = {
                # one-hot planes: uint8 in the ring, float32 out of sample()
                "obs": torch.zeros(self.T_cap, self.N, *obs_shape, dtype=torch.uint8,
                                   device=storage_device),
                "action": torch.zeros(self.T_cap, self.N, *action_shape, dtype=torch.long,
                                      device=storage_device),
                # Joint-action world models need the action that the other player
                # executed on the same frame.  ``opponent_valid`` distinguishes a
                # real all-NOOP action from collectors/environments that cannot
                # surface the opponent stream.
                "opponent_action": torch.zeros(self.T_cap, self.N, *action_shape,
                                               dtype=torch.long, device=storage_device),
                "opponent_valid": torch.zeros(self.T_cap, self.N, dtype=torch.bool,
                                              device=storage_device),
                "mask": torch.zeros(self.T_cap, self.N, *mask_shape, dtype=torch.bool,
                                    device=storage_device),
                "reward": torch.zeros(self.T_cap, self.N, device=storage_device),
                "cont": torch.zeros(self.T_cap, self.N, device=storage_device),
                "is_first": torch.zeros(self.T_cap, self.N, dtype=torch.bool,
                                        device=storage_device),
            }
        if state_shape is not None and globals_shape is not None:
            fields.update({
                "full_state": torch.zeros(
                    self.T_cap, self.N, *state_shape, dtype=torch.long,
                    device=storage_device),
                "full_globals": torch.zeros(
                    self.T_cap, self.N, *globals_shape, dtype=torch.long,
                    device=storage_device),
            })
        self.data = TensorDict(
            fields,
            batch_size=[self.T_cap, self.N],
            device=storage_device,
        )

    @property
    def size(self) -> int:
        """Number of filled timestep-rows (<= capacity)."""
        return min(self.count, self.T_cap)

    @property
    def num_transitions(self) -> int:
        return self.size * self.N

    def add(self, *, obs, action, mask, reward, cont, is_first,
            opponent_action=None, opponent_valid=None, full_state=None,
            full_globals=None) -> None:
        phys = self.count % self.T_cap
        if opponent_action is None:
            opponent_action = torch.zeros_like(action)
        if opponent_valid is None:
            opponent_valid = torch.zeros(action.shape[0], dtype=torch.bool,
                                         device=action.device)
        fields = {
                "obs": obs.to(torch.uint8), "action": action, "mask": mask.bool(),
                "opponent_action": opponent_action,
                "opponent_valid": opponent_valid.bool(),
                "reward": reward, "cont": cont, "is_first": is_first.bool(),
            }
        if "full_state" in self.data.keys():
            if full_state is None or full_globals is None:
                raise ValueError("structured replay row is missing full state")
            fields.update({"full_state": full_state.long(),
                           "full_globals": full_globals.long()})
        row = TensorDict(
            fields,
            batch_size=[self.N],
        ).to(self.storage_device)
        self.data[phys] = row
        self.count += 1

    def can_sample(self, seq_len: int) -> bool:
        return self.size >= seq_len

    def sample(self, batch: int, seq_len: int) -> TensorDict:
        """Draw ``batch`` sequences of length ``seq_len`` -> TensorDict[B, T, ...]
        on ``self.device``, with ``obs`` cast back to float32."""
        assert self.can_sample(seq_len), "not enough transitions to sample a sequence"
        lo = max(0, self.count - self.T_cap)
        hi = self.count - seq_len                          # inclusive last valid start
        starts = torch.randint(lo, hi + 1, (batch,), device=self.storage_device)
        envs = torch.randint(0, self.N, (batch,), device=self.storage_device)
        offs = torch.arange(seq_len, device=self.storage_device)
        phys = (starts[:, None] + offs[None, :]) % self.T_cap   # (B, T)
        out = self.data[phys, envs[:, None]]              # (B, T, ...)
        if str(self.device) != str(self.storage_device):
            out = out.to(self.device)
        out.set("obs", out.get("obs").float())
        return out
