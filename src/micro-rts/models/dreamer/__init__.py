"""DreamerV4 for MicroRTS — tokenizer + block-causal world model + action expert.

Importing this package registers the ``dreamerv4`` model type. The public classes
are re-exported for direct construction and testing.
"""

from .config import (
    DreamerV4Config,
    TokenizerConfig,
    DynamicsConfig,
    ActorCriticConfig,
    FreezeConfig,
)
from .tokenizer import GridTokenizer
from .world_model import WorldModel, GridActionEncoder
from .action_expert import ActionExpert
from .memory import WorldModelMemory
from .dreamer import DreamerV4, build_dreamerv4

__all__ = [
    "DreamerV4Config",
    "TokenizerConfig",
    "DynamicsConfig",
    "ActorCriticConfig",
    "FreezeConfig",
    "GridTokenizer",
    "WorldModel",
    "GridActionEncoder",
    "ActionExpert",
    "WorldModelMemory",
    "DreamerV4",
    "build_dreamerv4",
]
