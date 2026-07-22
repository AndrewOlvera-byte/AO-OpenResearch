"""Trainer registration hub.

Importing this module imports every trainer module so their
``@register("trainer", ...)`` decorators fire and ``build("trainer", ...)``
can resolve a trainer class from a config ``model.type``.

Import-only and idempotent.  Keep this package-local: it must never import
``atari`` (atari and micro-rts both expose a top-level ``models``/``trainers``
and share only ``core``/``shared``).
"""

from __future__ import annotations

# RL trainers (registered by their model.type keys).
from . import DreamerRLTrainer  # noqa: F401
from . import IncompleteBeliefRLTrainer  # noqa: F401
from . import PPOTrainer  # noqa: F401
from . import StructuredDreamerRLTrainer  # noqa: F401

# Pretraining trainers (PretrainTrainer subclasses, one per stage) are appended
# here as each stage is converted to the registry pattern, so that a single
# ``import registry_imports`` fires every decorator.
from . import incomplete_dynamics_trainers  # noqa: F401
from . import incomplete_tokenizer_trainers  # noqa: F401
from . import WorldActionDynamicsTrainer  # noqa: F401
from . import WorldActionEncoderTrainer  # noqa: F401
