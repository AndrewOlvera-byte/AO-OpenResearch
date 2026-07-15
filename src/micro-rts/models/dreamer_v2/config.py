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
