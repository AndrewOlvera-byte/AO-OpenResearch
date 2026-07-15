"""``AtariDreamerV4`` — top-level Atari assembly: tokenizer + world model + actor/critic.

Same model contract the shared ``AbstractDreamerTrainer`` expects (``tokenizer`` /
``world_model`` / ``action_expert`` submodules, ``build_optimizers`` -> world/actor/
critic, ``imagine``, and the ``Policy.step`` collection API), so the exact same
training loop drives both MicroRTS and Atari; only the spaces differ.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from core.registry import register

from .config import AtariDreamerConfig
from .tokenizer import AtariTokenizer
from .world_model import AtariWorldModel
from .action_expert import AtariActionExpert


class AtariDreamerV4(nn.Module):
    def __init__(self, obs_shape, num_actions: int, cfg: AtariDreamerConfig | None = None,
                 device: str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg or AtariDreamerConfig()
        self.num_actions = int(num_actions)

        self.tokenizer = AtariTokenizer(obs_shape, self.cfg.tokenizer)
        n_spatial, d_latent = self.tokenizer.n_spatial, self.tokenizer.d_latent
        self.world_model = AtariWorldModel(n_spatial, d_latent, num_actions, self.cfg.dynamics)
        self.action_expert = AtariActionExpert(n_spatial, d_latent, num_actions, self.cfg.actor_critic)

        self.device = device
        self.to(device)

    @torch.no_grad()
    def step(self, obs: torch.Tensor, mask=None, deterministic: bool = False) -> TensorDict:
        obs = obs.to(self.device)
        z = self.tokenizer.encode(obs)
        action, logprob, _, value = self.action_expert.act(z, deterministic=deterministic)
        return TensorDict({"action": action, "logprob": logprob, "value": value},
                          batch_size=[obs.shape[0]])

    @torch.no_grad()
    def imagine(self, z0: torch.Tensor, horizon: int | None = None) -> dict:
        """Autoregressive latent rollout. Returns detached z (B,H+1,...), action (B,H),
        reward (B,H), cont (B,H)."""
        horizon = horizon or self.cfg.actor_critic.imagine_horizon
        flow_steps = self.cfg.actor_critic.imagine_flow_steps
        z_list = [z0]
        a_list, r_list, c_list = [], [], []
        for _ in range(horizon):
            z_t = z_list[-1]
            action, _, _, _ = self.action_expert.act(z_t)
            a_list.append(action)
            z_seq = torch.stack(z_list, dim=1)
            a_seq = torch.stack(a_list, dim=1)
            ctx = self.world_model.contextualize(z_seq, a_seq)
            h_last = ctx["h"][:, -1:]                          # (B,1,n_spatial,d)
            # Generative dynamics: sample the next latent from the shortcut-forcing
            # flow (predicts motion) instead of the identity-prone deterministic head.
            if flow_steps > 0:
                z_next = self.world_model.flow_sample(h_last, flow_steps)[:, 0]
            else:
                z_next = self.world_model.next_latent(h_last, z_t[:, None])[:, 0]
            if self.cfg.tokenizer.tanh_bottleneck:
                z_next = torch.tanh(z_next)
            z_list.append(z_next)
            r_list.append(ctx["reward"][:, -1])
            c_list.append(torch.sigmoid(ctx["continue_logit"][:, -1]))
        return {
            "z": torch.stack(z_list, dim=1),
            "action": torch.stack(a_list, dim=1),
            "reward": torch.stack(r_list, dim=1),
            "cont": torch.stack(c_list, dim=1),
        }

    def build_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        world_params = list(self.tokenizer.parameters()) + list(self.world_model.parameters())
        c = self.cfg
        return {
            "world": torch.optim.Adam(world_params, lr=c.world_lr, eps=1e-5),
            "actor": torch.optim.Adam(self.action_expert.actor.parameters(), lr=c.actor_lr, eps=1e-5),
            "critic": torch.optim.Adam(self.action_expert.critic.parameters(), lr=c.critic_lr, eps=1e-5),
        }


@register("model", "atari_dreamerv4")
def build_atari_dreamerv4(obs_shape, action_nvec=None, num_actions=None, device="cpu", **kwargs):
    if num_actions is None:
        num_actions = int(action_nvec[0]) if action_nvec is not None else kwargs.pop("num_actions")
    cfg = AtariDreamerConfig.from_dict(kwargs)
    return AtariDreamerV4(obs_shape, num_actions, cfg=cfg, device=device)
