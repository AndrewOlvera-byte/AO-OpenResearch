"""Regression guard for the registry-driven model/loss/trainer wiring.

Asserts that importing the package-local hub fires every decorator, that the
canonical ``causal_world_action_dynamics`` stage resolves as model + loss +
trainer, that the loss-composition coefficient contract holds, and that the hub
does not leak the sibling ``atari`` package (which shares top-level module names).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import yaml


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


def test_every_microrts_experiment_selects_a_registered_trainer():
    import registry_imports  # noqa: F401
    from core.registry import registered

    configs = Path(__file__).parents[2] / "configs" / "exp" / "micro-rts"
    known = set(registered("trainer"))
    for path in configs.rglob("*.yaml"):
        payload = yaml.safe_load(path.read_text())
        assert payload, f"empty experiment config: {path}"
        trainer_type = (payload.get("trainer") or {}).get("type")
        assert trainer_type, f"missing trainer.type: {path}"
        assert trainer_type in known, f"unknown trainer {trainer_type!r}: {path}"


def test_registry_reports_missing_and_duplicate_components():
    from core.registry import build, register

    with pytest.raises(KeyError, match="registered types"):
        build("test_missing_kind", type="absent")

    @register("test_duplicate_kind", "thing")
    def first():
        return 1

    with pytest.raises(ValueError, match="duplicate registry entry"):

        @register("test_duplicate_kind", "thing")
        def second():
            return 2


def test_component_entrypoints_are_thin_dispatchers():
    entrypoints = Path(__file__).parents[1] / "entrypoints"
    for name in (
        "train_dreamer_tokenizer.py",
        "train_action_tokenizer.py",
        "train_dreamer_dynamics.py",
        "train_discrete_tokenizer.py",
        "train_discrete_action_tokenizer.py",
        "train_discrete_dynamics.py",
        "train_encoder.py",
        "train_opponent_latent.py",
    ):
        source = (entrypoints / name).read_text()
        assert len(source.splitlines()) <= 30, name
        assert "optimizer.step" not in source, name


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
    loss_cfg = cfg.training.get("loss") or {}
    assert loss_cfg["type"] == "causal_world_action_dynamics"
    for key in loss_cfg["weights"]:
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
