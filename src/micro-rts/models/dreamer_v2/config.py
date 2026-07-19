from __future__ import annotations

from dataclasses import dataclass, fields


def _from_dict(cls, values):
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in dict(values or {}).items() if k in valid})


@dataclass
class StructuredTokenizerConfig:
    d_cell: int = 128
    d_latent: int = 128
    downsample: int = 2
    depth: int = 2
    n_heads: int = 4
    dropout: float = 0.0
    max_unit_types: int = 16
    max_entities: int = 128
    mask_width: int = 79
    legacy_obs_channels: int = 27
    # Optional exact-field interface. The latent remains continuous, but every
    # finite engine integer is embedded and decoded categorically rather than
    # optimized only through scale-normalized regression and rounding.
    exact_categorical: bool = False
    max_tick: int = 4096
    max_hp: int = 64
    max_carried: int = 64
    max_eta: int = 512
    max_remaining: int = 512
    max_resources: int = 256
    max_reserved_positions: int = 256
    exact_consistency_coef: float = 0.1

    @classmethod
    def from_dict(cls, values):
        return _from_dict(cls, values)


@dataclass
class TemporalJEPAConfig:
    """Disposable action-conditioned predictor used during tokenizer SSL."""

    d_model: int = 512
    field_dim: int = 64
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_action_events: int = 32
    max_unit_types: int = 16
    ema_decay: float = 0.996
    ema_end_decay: float = 1.0
    # Predict a raw latent delta so the resulting latent can be decoded through
    # the exact tokenizer heads. Legacy JEPA checkpoints predict in normalized
    # space and retain the default false value.
    raw_prediction: bool = False

    @classmethod
    def from_dict(cls, values):
        return _from_dict(cls, values)


@dataclass
class StructuredDynamicsConfig:
    d_model: int = 512
    depth: int = 8
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_action_events: int = 32
    k_max: int = 16
    prior_fraction: float = 0.25
    skip_fraction: float = 0.25
    # Deterministic mechanics can start the one-step sampler from zero instead
    # of forcing the transformer to cancel an irrelevant Gaussian target token.
    initial_noise: str = "gaussian"
    # ``legacy`` preserves checkpoint compatibility with the original summed
    # field embeddings. ``factorized`` is the separately pretrainable encoder.
    action_encoder_type: str = "legacy"
    action_field_dim: int = 64
    # Predict normalized next-state latents as ``departure + delta``. This gives
    # deterministic mechanics an exact identity path for unchanged state.
    residual_prediction: bool = False
    # Instantiate the action-tokenizer's pretrained state-query/cross-attention
    # delta router. The entrypoint loads its weights with the event encoder.
    pretrained_action_router: bool = False
    # Scatter source/destination event features directly into their spatial
    # target queries instead of requiring the transformer to infer coordinate
    # binding through dense self-attention alone.
    explicit_spatial_action_routing: bool = False
    # Exclude unused packed entity inputs from transformer key/value attention.
    mask_empty_entity_tokens: bool = False

    @classmethod
    def from_dict(cls, values):
        return _from_dict(cls, values)


@dataclass
class ActionTokenizerConfig:
    d_model: int = 512
    field_dim: int = 64
    n_heads: int = 8
    inverse_depth: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_action_events: int = 32
    max_unit_types: int = 16

    @classmethod
    def from_dict(cls, values):
        return _from_dict(cls, values)


@dataclass
class DiscreteTokenizerConfig:
    """Hard categorical tokenizer for canonical MicroRTS state.

    ``spatial_downsample`` and ``codebook_depth`` deliberately remain exposed:
    representation capacity is an experiment axis, not a hidden architectural
    assumption.  Every decoder path starts from hard product-code IDs.
    """

    d_model: int = 256
    depth: int = 2
    n_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    spatial_downsample: int = 2
    codebook_size: int = 512
    codebook_depth: int = 4
    n_global_tokens: int = 4
    max_unit_types: int = 16
    max_tick: int = 4096
    max_hp: int = 64
    max_carried: int = 64
    max_eta: int = 512
    max_remaining: int = 512
    max_resources: int = 256
    max_reserved_positions: int = 256
    mask_width: int = 79
    legacy_obs_channels: int = 27

    @classmethod
    def from_dict(cls, values):
        return _from_dict(cls, values)


@dataclass
class DiscreteActionTokenizerConfig:
    d_model: int = 512
    field_dim: int = 64
    n_heads: int = 8
    inverse_depth: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # The observed v4 corpus reaches 42 simultaneous events.  Forty-eight keeps
    # that corpus exact while leaving 32/64 available as explicit ablations.
    max_action_events: int = 48
    max_unit_types: int = 16

    @classmethod
    def from_dict(cls, values):
        return _from_dict(cls, values)


@dataclass
class DiscreteDynamicsConfig:
    d_model: int = 512
    depth: int = 8
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_action_events: int = 48
    max_unit_types: int = 16
    action_field_dim: int = 64
    pretrained_action_router: bool = True
    zero_init_correction: bool = True

    @classmethod
    def from_dict(cls, values):
        return _from_dict(cls, values)
