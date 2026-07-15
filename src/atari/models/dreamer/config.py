"""``AtariDreamerConfig`` — spec for the Atari DreamerV4 agent.

Same three-optimizer structure as the MicroRTS build (tokenizer + dynamics =
``world``; ``actor``; ``critic``), sized for 64x64 Atari frames and a single
``Discrete(num_actions)`` action space. Defaults are a *small* architecture tuned to
fit comfortably on a 16 GB GPU while following the DreamerV3/4 Atari recipe
(imagination horizon 15, gamma 0.997, lambda 0.95, slow critic EMA).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any, Tuple


@dataclass
class TokenizerConfig:
    base_channels: int = 32     # encoder stem width (stages grow from here)
    d_latent: int = 32          # per-cell latent token width
    downsample: int = 16        # 64 -> 4 spatial bottleneck (four stride-2 convs)
    tanh_bottleneck: bool = True


@dataclass
class DynamicsConfig:
    d_model: int = 256
    depth: int = 4
    n_heads: int = 8
    n_register: int = 4
    mlp_ratio: float = 4.0
    time_every: int = 2
    dropout: float = 0.0
    space_mode: str = "wm_agent"
    scale_pos_embeds: bool = True
    k_max: int = 16
    reward_bins: int = 255      # symlog two-hot reward head (DreamerV3 stabilizer)


@dataclass
class ActorCriticConfig:
    hidden: Tuple[int, ...] = (512,)
    imagine_horizon: int = 15
    gamma: float = 0.997        # DreamerV3 Atari default
    lam: float = 0.95
    entropy_coef: float = 3e-4
    critic_ema: float = 0.98
    unimix: float = 0.01         # Dreamer-style categorical smoothing for exploration
    actor_outscale: float = 0.01 # small initial logits -> near-uniform policy
    critic_bins: int = 255      # symlog two-hot critic (matches reward head)
    imagine_flow_steps: int = 4  # shortcut-flow ODE steps for imagination (0 = deterministic head)
    normalize_adv: bool = False
    return_norm: str = "percentile"
    return_norm_rate: float = 0.01
    return_norm_limit: float = 1.0
    return_norm_low: float = 5.0
    return_norm_high: float = 95.0


@dataclass
class AtariDreamerConfig:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)

    world_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4
    grad_clip: float = 1000.0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "AtariDreamerConfig":
        d = dict(d or {})
        sub = {
            "tokenizer": TokenizerConfig,
            "dynamics": DynamicsConfig,
            "actor_critic": ActorCriticConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sub.items():
            if name in d and isinstance(d[name], dict):
                valid = {f.name for f in fields(klass)}
                kwargs[name] = klass(**{k: v for k, v in d[name].items() if k in valid})
        top = {f.name for f in fields(cls)} - set(sub)
        for k, v in d.items():
            if k in top:
                kwargs[k] = v
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
