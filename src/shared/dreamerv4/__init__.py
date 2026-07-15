"""Vendored Dreamer 4 primitive architectures.

Pulled from https://github.com/nicklashansen/dreamer4 (commit bdeddfe, MIT).
Only the primitive nn.Module building blocks are included -- forward passes are
defined explicitly and backward is provided by torch autograd. Training loops,
datasets, and the interactive web UI from the upstream repo are intentionally
omitted.
"""

from .model import (
    # helpers / layout
    Modality,
    TokenLayout,
    temporal_patchify,
    temporal_unpatchify,
    sinusoid_table,
    add_sinusoidal_positions,
    pack_bottleneck_to_spatial,
    unpack_spatial_to_bottleneck,
    # primitives
    RMSNorm,
    MLP,
    MultiheadSelfAttention,
    SpaceSelfAttentionModality,
    TimeSelfAttention,
    BlockCausalLayer,
    BlockCausalTransformer,
    MAEReplacer,
    # composed architectures
    Encoder,
    Decoder,
    Tokenizer,
    ActionEncoder,
    TaskEmbedder,
    Dynamics,
    # loss helpers
    recon_loss_from_mae,
    lpips_on_mae_recon,
)
from .twohot import symlog, symexp, two_hot, TwoHot

__all__ = [
    "Modality",
    "TokenLayout",
    "temporal_patchify",
    "temporal_unpatchify",
    "sinusoid_table",
    "add_sinusoidal_positions",
    "pack_bottleneck_to_spatial",
    "unpack_spatial_to_bottleneck",
    "RMSNorm",
    "MLP",
    "MultiheadSelfAttention",
    "SpaceSelfAttentionModality",
    "TimeSelfAttention",
    "BlockCausalLayer",
    "BlockCausalTransformer",
    "MAEReplacer",
    "Encoder",
    "Decoder",
    "Tokenizer",
    "ActionEncoder",
    "TaskEmbedder",
    "Dynamics",
    "recon_loss_from_mae",
    "lpips_on_mae_recon",
    "symlog",
    "symexp",
    "two_hot",
    "TwoHot",
]
