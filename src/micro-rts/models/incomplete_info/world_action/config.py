from __future__ import annotations

from dataclasses import dataclass, fields


def _from_dict(cls, values):
    valid = {item.name for item in fields(cls)}
    return cls(**{k: v for k, v in dict(values or {}).items() if k in valid})


@dataclass
class PredictiveBeliefConfig:
    """Factorized predictive-state encoder used by the world-action model."""

    d_model: int = 512
    d_latent: int = 320
    n_heads: int = 8
    frame_depth: int = 2
    temporal_depth: int = 4
    mixer_depth: int = 2
    predictor_depth: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    n_self_tokens: int = 8
    n_opponent_tokens: int = 8
    n_static_tokens: int = 4
    n_interaction_tokens: int = 8
    context_length: int = 64
    horizons: tuple[int, ...] = (1, 2, 4, 8, 16)
    ema_decay: float = 0.996
    discount: float = 0.997
    event_dim: int = 8

    @classmethod
    def from_dict(cls, values):
        out = _from_dict(cls, values)
        out.horizons = tuple(int(value) for value in out.horizons)
        return out

    @property
    def branch_sizes(self):
        return (
            self.n_self_tokens,
            self.n_opponent_tokens,
            self.n_static_tokens,
            self.n_interaction_tokens,
        )

    @property
    def n_tokens(self):
        return sum(self.branch_sizes)


@dataclass
class FactorizedDynamicsConfig:
    """Flow transition over intrinsic, extrinsic, and interaction branches."""

    d_model: int = 512
    d_latent: int = 320
    n_heads: int = 8
    depth: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    sample_steps: int = 4
    n_self_tokens: int = 8
    n_opponent_tokens: int = 8
    n_static_tokens: int = 4
    n_interaction_tokens: int = 8
    n_opponent_plan_tokens: int = 8

    @classmethod
    def from_dict(cls, values):
        return _from_dict(cls, values)

    @property
    def branch_sizes(self):
        return (
            self.n_self_tokens,
            self.n_opponent_tokens,
            self.n_static_tokens,
            self.n_interaction_tokens,
        )
