from __future__ import annotations

from dataclasses import dataclass, fields


def _from_dict(cls, values):
    valid = {item.name for item in fields(cls)}
    return cls(**{k: v for k, v in dict(values or {}).items() if k in valid})


@dataclass
class EgoTokenizerConfig:
    obs_channels: int = 27
    d_model: int = 320
    d_latent: int = 320
    depth: int = 4
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    downsample: int = 2
    n_registers: int = 8
    mask_fraction: float = 0.35
    ema_decay: float = 0.996

    @classmethod
    def from_dict(cls, values):
        return _from_dict(cls, values)


@dataclass
class SelfActionTokenizerConfig:
    d_model: int = 512
    d_latent: int = 320
    field_dim: int = 64
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_action_events: int = 48
    max_unit_types: int = 16

    @classmethod
    def from_dict(cls, values):
        return _from_dict(cls, values)


@dataclass
class OpponentPlanTokenizerConfig:
    d_model: int = 512
    d_latent: int = 320
    field_dim: int = 64
    n_heads: int = 8
    depth: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_action_events: int = 48
    max_unit_types: int = 16
    n_plan_tokens: int = 8
    horizons: tuple[int, ...] = (0, 1, 2, 4, 8, 16)

    @classmethod
    def from_dict(cls, values):
        out = _from_dict(cls, values)
        out.horizons = tuple(int(x) for x in out.horizons)
        return out


@dataclass
class HistoryConfig:
    d_model: int = 512
    depth: int = 4
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    n_registers: int = 8
    context_length: int = 64
    time_every: int = 1

    @classmethod
    def from_dict(cls, values):
        return _from_dict(cls, values)


@dataclass
class JointFlowConfig:
    d_model: int = 512
    depth: int = 2
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    sample_steps: int = 4

    @classmethod
    def from_dict(cls, values):
        return _from_dict(cls, values)


@dataclass
class IntentPriorConfig:
    d_model: int = 512
    depth: int = 3
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    n_modes: int = 8
    contrast_dim: int = 128

    @classmethod
    def from_dict(cls, values):
        return _from_dict(cls, values)


@dataclass
class BeliefDynamicsConfig:
    d_model: int = 512
    depth: int = 4
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    sample_steps: int = 4
    explicit_action_residual: bool = False
    # V3.5 keeps the deployable next-belief flow interface but solves the
    # easier internal problem: history -> current anchor, flow -> transition
    # residual, returned sample -> anchor + residual.
    current_belief_anchor: bool = False
    axial_state_position: bool = False
    direct_action_attention: bool = False

    @classmethod
    def from_dict(cls, values):
        return _from_dict(cls, values)
