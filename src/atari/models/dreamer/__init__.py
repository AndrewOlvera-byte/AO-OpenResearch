"""Atari DreamerV4 — tokenizer + block-causal world model + categorical action expert.

Importing this package registers the ``atari_dreamerv4`` model type.
"""

from .config import (
    AtariDreamerConfig,
    TokenizerConfig,
    DynamicsConfig,
    ActorCriticConfig,
)
from .tokenizer import AtariTokenizer
from .world_model import AtariWorldModel
from .action_expert import AtariActionExpert
from .dreamer import AtariDreamerV4, build_atari_dreamerv4

__all__ = [
    "AtariDreamerConfig",
    "TokenizerConfig",
    "DynamicsConfig",
    "ActorCriticConfig",
    "AtariTokenizer",
    "AtariWorldModel",
    "AtariActionExpert",
    "AtariDreamerV4",
    "build_atari_dreamerv4",
]
