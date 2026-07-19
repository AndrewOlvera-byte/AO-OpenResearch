"""Dreamer agent heads and imagination loop for the structured world model.

The structured transition model is pretrained independently.  Online RL adds
small policy, behavior-prior, opponent, reward, continuation, and value heads
without changing the tokenizer/dynamics checkpoint key layout.  Actor/value
updates consume detached imagined trajectories; dynamics are updated separately
from real and anchored replay only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from core.registry import register
from models.shared.GridNetHead import GridNetActionHead
from shared.dreamerv4 import TwoHot

from .config import StructuredDynamicsConfig, StructuredTokenizerConfig
from .dynamics import StructuredWorldModelV2


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


@dataclass
class StructuredActorCriticConfig:
    dec_channels: int = 128
    summary_hidden: int = 512
    critic_hidden: tuple[int, ...] = (512, 512)
    reward_bins: int = 255
    critic_bins: int = 255
    imagine_horizon: int = 8
    imagine_flow_steps: int = 1
    gamma: float = 0.997
    lam: float = 0.95
    critic_ema: float = 0.98
    pmpo_alpha: float = 0.5
    prior_kl_coef: float = 0.3
    entropy_coef: float = 0.0


@dataclass
class StructuredAgentFreezeConfig:
    tokenizer: bool = True
    action_encoder: bool = True
    action_router: bool = True
    dynamics: bool = False
    actor: bool = False
    critic: bool = False
    agent_heads: bool = False


@dataclass
class StructuredDreamerConfig:
    tokenizer: StructuredTokenizerConfig = field(default_factory=StructuredTokenizerConfig)
    dynamics: StructuredDynamicsConfig = field(default_factory=StructuredDynamicsConfig)
    actor_critic: StructuredActorCriticConfig = field(default_factory=StructuredActorCriticConfig)
    freeze: StructuredAgentFreezeConfig = field(default_factory=StructuredAgentFreezeConfig)
    world_lr: float = 2.0e-6
    agent_lr: float = 1.0e-4
    actor_lr: float = 3.0e-5
    critic_lr: float = 1.0e-4
    grad_clip: float = 10.0

    @classmethod
    def from_dict(cls, values):
        values = dict(values or {})
        subtypes = {
            "tokenizer": StructuredTokenizerConfig,
            "dynamics": StructuredDynamicsConfig,
            "actor_critic": StructuredActorCriticConfig,
            "freeze": StructuredAgentFreezeConfig,
        }
        kwargs = {}
        for name, typ in subtypes.items():
            if isinstance(values.get(name), dict):
                valid = {f.name for f in fields(typ)}
                item = {k: v for k, v in values[name].items() if k in valid}
                if name == "actor_critic" and isinstance(item.get("critic_hidden"), list):
                    item["critic_hidden"] = tuple(item["critic_hidden"])
                kwargs[name] = typ(**item)
        top = {f.name for f in fields(cls)} - set(subtypes)
        kwargs.update({k: v for k, v in values.items() if k in top})
        return cls(**kwargs)

    def to_dict(self):
        return asdict(self)


class StructuredLatentSummary(nn.Module):
    """Small agent-only readout; world tokens never attend back to these heads."""

    def __init__(self, tokenizer, hidden: int):
        super().__init__()
        d = tokenizer.d_latent
        self.ns, self.ne = tokenizer.n_spatial, tokenizer.n_entity
        self.norm = nn.LayerNorm(d)
        self.score = nn.Linear(d, 1)
        self.out = nn.Sequential(
            nn.Linear(4 * d, hidden), nn.SiLU(), nn.LayerNorm(hidden)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        lead = z.shape[:-2]
        flat = self.norm(z.reshape(-1, *z.shape[-2:]))
        weights = torch.softmax(self.score(flat).squeeze(-1), dim=-1)
        pooled = (weights[..., None] * flat).sum(1)
        globals_ = flat[:, self.ns + self.ne :].reshape(flat.shape[0], -1)
        return self.out(torch.cat([pooled, globals_], -1)).reshape(*lead, -1)


class StructuredGridPolicy(nn.Module):
    def __init__(self, tokenizer, cell_nvec, channels: int, summary_hidden: int):
        super().__init__()
        self.h, self.w = tokenizer.h, tokenizer.w
        self.ds = tokenizer.cfg.downsample
        self.ns, self.d = tokenizer.n_spatial, tokenizer.d_latent
        self.summary = StructuredLatentSummary(tokenizer, summary_hidden)
        layers: list[nn.Module] = [
            nn.Conv2d(self.d, channels, 1), _group_norm(channels), nn.SiLU()
        ]
        if self.ds == 2:
            layers += [
                nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
                _group_norm(channels), nn.SiLU(),
            ]
        self.decoder = nn.Sequential(*layers)
        self.context = nn.Linear(summary_hidden, channels)
        self.out = nn.Conv2d(channels, int(cell_nvec.sum()), 1)
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)
        self.dist = GridNetActionHead(cell_nvec)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        n = z.shape[0]
        spatial = z[:, : self.ns].reshape(
            n, self.h // self.ds, self.w // self.ds, self.d
        ).permute(0, 3, 1, 2)
        feat = self.decoder(spatial) + self.context(self.summary(z))[:, :, None, None]
        return self.out(F.silu(feat)).permute(0, 2, 3, 1).reshape(n, self.h * self.w, -1)

    def sample(self, z, mask, deterministic=False):
        logits = self.logits(z)
        action, _ = self.dist.sample(logits, mask, deterministic=deterministic)
        logprob, entropy = self.dist.log_prob_entropy(logits, action, mask)
        return action, logprob, entropy

    def evaluate(self, z, action, mask):
        logits = self.logits(z)
        logprob, entropy = self.dist.log_prob_entropy(logits, action, mask)
        return logprob, entropy, logits

    def reverse_kl(self, z, prior: "StructuredGridPolicy", mask):
        """Exact reverse KL(pi || prior) over legal per-cell components."""
        logits, prior_logits = self.logits(z), prior.logits(z).detach()
        comp_mask = mask[..., 1:].bool()
        total = logits.new_zeros(logits.shape[0])
        offset = 0
        for size in self.dist.comp_splits:
            valid = comp_mask[..., offset : offset + size]
            any_valid = valid.any(-1)
            lp = logits[..., offset : offset + size].masked_fill(~valid, -1e8)
            lq = prior_logits[..., offset : offset + size].masked_fill(~valid, -1e8)
            lp = torch.where(any_valid[..., None], lp, torch.zeros_like(lp))
            lq = torch.where(any_valid[..., None], lq, torch.zeros_like(lq))
            logp, logq = F.log_softmax(lp, -1), F.log_softmax(lq, -1)
            kl = (logp.exp() * (logp - logq)).sum(-1) * any_valid
            total = total + kl.sum(-1)
            offset += size
        return total


class StructuredActionExpert(nn.Module):
    def __init__(self, tokenizer, action_nvec, cfg: StructuredActorCriticConfig):
        super().__init__()
        cell_nvec = torch.as_tensor(action_nvec[1:], dtype=torch.long)
        args = (tokenizer, cell_nvec, cfg.dec_channels, cfg.summary_hidden)
        self.actor_policy = StructuredGridPolicy(*args)
        self.behavior_prior = StructuredGridPolicy(*args)
        self.opponent_policy = StructuredGridPolicy(*args)
        self.value_summary = StructuredLatentSummary(tokenizer, cfg.summary_hidden)
        dims = [cfg.summary_hidden, *cfg.critic_hidden, cfg.critic_bins]
        layers = []
        for din, dout in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(din, dout), nn.SiLU(), nn.LayerNorm(dout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.critic = nn.Sequential(*layers)
        import copy
        self.target_critic = copy.deepcopy(self.critic)
        self.target_value_summary = copy.deepcopy(self.value_summary)
        self.target_critic.requires_grad_(False)
        self.target_value_summary.requires_grad_(False)
        self.coder = TwoHot(cfg.critic_bins)
        self.cfg = cfg

    def value_logits(self, z):
        return self.critic(self.value_summary(z))

    def value(self, z):
        return self.coder.mean(self.value_logits(z))

    def target_value(self, z):
        return self.coder.mean(self.target_critic(self.target_value_summary(z)))

    @torch.no_grad()
    def update_target(self, decay=None):
        decay = self.cfg.critic_ema if decay is None else float(decay)
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.mul_(decay).add_(source, alpha=1.0 - decay)
        for target, source in zip(
            self.target_value_summary.parameters(), self.value_summary.parameters()
        ):
            target.mul_(decay).add_(source, alpha=1.0 - decay)

    @torch.no_grad()
    def sync_actor_from_prior(self):
        self.actor_policy.load_state_dict(self.behavior_prior.state_dict())


class StructuredScalarHeads(nn.Module):
    def __init__(self, tokenizer, cfg: StructuredActorCriticConfig):
        super().__init__()
        self.summary = StructuredLatentSummary(tokenizer, cfg.summary_hidden)
        self.reward = nn.Linear(cfg.summary_hidden, cfg.reward_bins)
        self.continue_logit = nn.Linear(cfg.summary_hidden, 1)
        nn.init.zeros_(self.reward.weight)
        nn.init.zeros_(self.reward.bias)
        self.reward_coder = TwoHot(cfg.reward_bins)

    def forward(self, z):
        h = self.summary(z)
        return self.reward(h), self.continue_logit(h).squeeze(-1)


class StructuredDreamer(StructuredWorldModelV2):
    """Structured pretrained dynamics plus isolated Dreamer agent heads."""

    uses_structured_state = True

    def __init__(self, grid_hw, action_nvec, cfg: StructuredDreamerConfig, device="cpu"):
        super().__init__(grid_hw, cfg.tokenizer, cfg.dynamics)
        self.cfg = cfg
        self.action_nvec = torch.as_tensor(action_nvec, dtype=torch.long)
        self.action_expert = StructuredActionExpert(self.tokenizer, self.action_nvec, cfg.actor_critic)
        self.scalar_heads = StructuredScalarHeads(self.tokenizer, cfg.actor_critic)
        self.collect_with_prior = True
        self._apply_agent_freeze()
        self.device = torch.device(device)
        self.to(self.device)

    def _apply_agent_freeze(self):
        fz = self.cfg.freeze
        self.tokenizer.requires_grad_(not fz.tokenizer)
        self.dynamics.requires_grad_(not fz.dynamics)
        if fz.action_encoder:
            self.dynamics.action_encoder.requires_grad_(False)
            self.dynamics.action_position.requires_grad_(False)
        if fz.action_router:
            for name in (
                "action_router_state", "action_router_attn", "action_router_norm",
                "action_router_head",
            ):
                module = getattr(self.dynamics, name, None)
                if module is not None:
                    module.requires_grad_(False)
        self.action_expert.actor_policy.requires_grad_(not fz.actor)
        self.action_expert.critic.requires_grad_(not fz.critic)
        self.action_expert.value_summary.requires_grad_(not fz.critic)
        self.scalar_heads.requires_grad_(not fz.agent_heads)
        self.action_expert.behavior_prior.requires_grad_(not fz.agent_heads)
        self.action_expert.opponent_policy.requires_grad_(not fz.agent_heads)

    def predicted_mask(self, z):
        return (torch.sigmoid(self.tokenizer.decode_mask(z)) > 0.5).float()

    @staticmethod
    def role_mask(state, role: int, mask_width: int):
        source = state[..., 1].bool() & (state[..., 3] == role) & ~state[..., 7].bool()
        mask = source[..., None].expand(*source.shape, mask_width).clone()
        mask[..., 0] = source
        return mask.float()

    def role_bc_inputs(self, state, action, role: int):
        """Canonical labels/mask for conditional GridNet behavior cloning.

        The Java channel marks an explicitly issued TYPE_NONE with 255 in the
        otherwise inactive attack field. Conditional fields are engine-ignored,
        so zero them before Categorical validates labels and score only fields
        selected by the executed action type.
        """
        source = state[..., 1].bool() & (state[..., 3] == role) & ~state[..., 7].bool()
        safe = torch.zeros_like(action)
        safe[..., 0] = torch.where(source, action[..., 0], 0)
        mask_shape = (
            *action.shape[:-1],
            1 + sum(self.action_expert.opponent_policy.dist.comp_splits),
        )
        mask = action.new_zeros(mask_shape, dtype=torch.bool)
        mask[..., 0] = source
        active_types = (None, 1, 2, 3, 4, 4, 5)
        offset = 1
        for component, (size, active_type) in enumerate(zip(
            self.action_expert.opponent_policy.dist.comp_splits, active_types
        )):
            active = source if active_type is None else source & (action[..., 0] == active_type)
            mask[..., offset:offset + size] = active[..., None]
            if component:
                safe[..., component] = torch.where(active, action[..., component], 0)
            offset += size
        return safe, mask.float()

    @torch.no_grad()
    def step(self, obs, mask=None, deterministic=False, *, state=None, globals_=None):
        if state is None or globals_ is None:
            raise ValueError("structured_dreamer policy requires full_state/full_globals")
        z = self.tokenizer.encode(state.to(self.device), globals_.to(self.device))
        use_mask = mask.to(self.device) if mask is not None else self.predicted_mask(z)
        policy = (
            self.action_expert.behavior_prior
            if self.collect_with_prior
            else self.action_expert.actor_policy
        )
        action, logprob, entropy = policy.sample(z, use_mask, deterministic)
        return TensorDict(
            {"action": action, "logprob": logprob, "entropy": entropy,
             "value": self.action_expert.value(z), "z": z},
            batch_size=[z.shape[0]],
        )

    @torch.no_grad()
    def imagine(self, z0, horizon=None, flow_steps=None):
        horizon = int(horizon or self.cfg.actor_critic.imagine_horizon)
        flow_steps = int(flow_steps or self.cfg.actor_critic.imagine_flow_steps)
        z = z0
        z_seq, actions, masks, rewards, conts = [z], [], [], [], []
        for _ in range(horizon):
            decoded = self.tokenizer.decode(z)
            state, _globals = self.tokenizer.discretize(decoded)
            mask = (torch.sigmoid(decoded["mask"]) > 0.5).float()
            action, _, _ = self.action_expert.actor_policy.sample(z, mask)
            opp_mask = self.role_mask(state, 2, mask.shape[-1])
            opponent, _, _ = self.action_expert.opponent_policy.sample(z, opp_mask)
            events, valid, _ = self.action_events(state, action, opponent)
            z = self.dynamics.sample_next(
                z, events, valid, flow_steps,
                state_token_valid=self.state_token_valid(state),
            )
            reward_logits, continue_logit = self.scalar_heads(z)
            z_seq.append(z)
            actions.append(action)
            masks.append(mask)
            rewards.append(self.scalar_heads.reward_coder.mean(reward_logits))
            conts.append(torch.sigmoid(continue_logit))
        return {
            "z": torch.stack(z_seq, 1),
            "action": torch.stack(actions, 1),
            "mask": torch.stack(masks, 1),
            "reward": torch.stack(rewards, 1),
            "cont": torch.stack(conts, 1),
        }

    def build_optimizers(self):
        self._apply_agent_freeze()
        c = self.cfg
        groups = {
            "world": ([p for p in self.dynamics.parameters() if p.requires_grad], c.world_lr),
            "agent": ([p for m in (
                self.scalar_heads, self.action_expert.behavior_prior,
                self.action_expert.opponent_policy,
            ) for p in m.parameters() if p.requires_grad], c.agent_lr),
            "actor": ([p for p in self.action_expert.actor_policy.parameters() if p.requires_grad], c.actor_lr),
            "critic": ([p for m in (
                self.action_expert.value_summary, self.action_expert.critic,
            ) for p in m.parameters() if p.requires_grad], c.critic_lr),
        }
        return {
            name: torch.optim.AdamW(params, lr=lr, eps=1e-5, weight_decay=1e-4)
            for name, (params, lr) in groups.items() if params
        }


@register("model", "structured_dreamer")
def build_structured_dreamer(obs_shape, action_nvec, device="cpu", **kwargs):
    cfg = StructuredDreamerConfig.from_dict(kwargs)
    grid_hw = tuple(obs_shape[-2:])
    return StructuredDreamer(grid_hw, action_nvec, cfg, device=device)
