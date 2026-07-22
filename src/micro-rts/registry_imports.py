"""Central registry-import hub for the micro-rts package.

Importing this module imports every model / loss / trainer module so that their
``@register(kind, name)`` decorators fire.  After ``import registry_imports``,
``build("model"|"loss"|"trainer", type=..., **kwargs)`` resolves any component
declared by a config ``type`` field.

Registration is import-side-effect driven, so entrypoints do a single
``import registry_imports  # noqa`` after their ``sys.path`` bootstrap instead of
scattering per-module side-effect imports.

CRITICAL INVARIANT: this hub is package-local and must never import ``atari``.
``atari`` and ``micro-rts`` both expose a top-level ``models`` / ``environments``
/ ``loss`` / ``collectors`` and share only ``core`` / ``shared``; importing atari
here would collide those top-level names.  ``tests/test_registry_wiring.py``
asserts ``atari`` is absent from ``sys.modules`` after this import.
"""

from __future__ import annotations

# Models + their co-located losses (decorators live on the module functions).
from models import incomplete_info  # noqa: F401
from models.incomplete_info import world_action  # noqa: F401
from models import dreamer_v2  # noqa: F401
from models import dreamer as _dreamer  # noqa: F401  (dreamerv4 RL model)

# Model-free policies used by PPO (registered by module import).
from models import cnn_mlp_policy  # noqa: F401
from models import gridnet_policy  # noqa: F401
from models import masked_policy  # noqa: F401

# RL / structured losses.
import loss  # noqa: F401

# Trainers (RL + pretraining stages).
from trainers import registry as _trainer_registry  # noqa: F401
