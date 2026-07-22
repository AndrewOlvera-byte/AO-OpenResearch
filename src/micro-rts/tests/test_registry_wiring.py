"""Regression guard for the registry-driven model/loss/trainer wiring.

Asserts that importing the package-local hub fires every decorator, that the
canonical ``causal_world_action_dynamics`` stage resolves as model + loss +
trainer, that the loss-composition coefficient contract holds, and that the hub
does not leak the sibling ``atari`` package (which shares top-level module names).
"""

from __future__ import annotations

import inspect
import sys

import pytest


def test_hub_registers_components_without_atari_leak():
    import registry_imports  # noqa: F401
    from core.registry import _REGISTRY

    assert "atari" not in sys.modules, "micro-rts hub must not import atari"

    for name in ("causal_world_action_dynamics", "cnn_mlp", "structured_dreamer"):
        assert name in _REGISTRY["model"], name
    for name in ("predictive_belief", "causal_world_action_dynamics", "belief_dynamics"):
        assert name in _REGISTRY["loss"], name
    for name in (
        "causal_world_action_dynamics",
        "causal_world_action_encoder",
        "incomplete_ego_tokenizer",
        "opponent_intent_prior",
        "belief_dynamics",
        "joint_flow_dynamics",
    ):
        assert name in _REGISTRY["trainer"], name


def test_canonical_model_builds_from_registry():
    import registry_imports  # noqa: F401
    from core.registry import build

    model = build(
        "model",
        type="causal_world_action_dynamics",
        factorized_dynamics={"d_model": 64, "d_latent": 32, "depth": 2, "n_heads": 4},
    )
    assert sum(p.numel() for p in model.parameters()) > 0


def test_dynamics_loss_coefficient_contract_matches_config_keys():
    """Every canonical-config loss key maps to a ``<key>_coef`` loss kwarg."""
    import registry_imports  # noqa: F401
    from core.config import Config
    from core.registry import _REGISTRY

    cfg = Config.from_experiment(
        "micro-rts/paper/incomplete_info/causal_world_action_v1/"
        "pretrain_factorized_world_action_dynamics"
    )
    params = inspect.signature(
        _REGISTRY["loss"]["causal_world_action_dynamics"]
    ).parameters
    for key in (cfg.training.get("loss") or {}):
        assert f"{key}_coef" in params, f"loss missing coefficient for '{key}'"


@pytest.mark.parametrize(
    "exp,expected",
    [
        ("probe_ego_tokenizer", "incomplete_ego_tokenizer"),
        ("probe_self_action_tokenizer", "incomplete_self_action_tokenizer"),
        ("probe_opponent_plan_tokenizer", "incomplete_opponent_plan_tokenizer"),
        ("pretrain_belief_dynamics_medium_v1", "belief_dynamics"),
        ("probe_joint_flow_dynamics", "joint_flow_dynamics"),
    ],
)
def test_config_resolves_to_trainer(exp, expected):
    import registry_imports  # noqa: F401
    from core.config import Config
    from entrypoints.pretrain import resolve_trainer_type

    cfg = Config.from_experiment(f"micro-rts/paper/incomplete_info/{exp}")
    assert resolve_trainer_type(cfg) == expected
