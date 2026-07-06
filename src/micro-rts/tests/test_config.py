"""Config loading: deep-merge, interpolation, nested training, CWD-independence."""

import os

from core.config import Config


def test_from_experiment_merges_and_interpolates():
    cfg = Config.from_experiment("micro-rts/base_rlFS_expert")
    assert cfg.model["type"] == "cnn_mlp"
    assert cfg.wandb["project"] == "micro-rts"
    # ${run.name} resolved against the merged (overridden) run name.
    assert cfg.wandb["run_name"] == "base_rlFS_expert"
    # nested training sections present after merge.
    assert "ppo" in cfg.training and "curriculum" in cfg.training and "eval" in cfg.training
    assert cfg.training["curriculum"]["type"] == "ppo"
    assert isinstance(cfg.training["ppo"]["lr"], float)


def test_apply_overrides_types_and_paths():
    cfg = Config.from_experiment("micro-rts/base_rlFS_expert")
    cfg.apply_overrides([
        "training.ppo.lr=1e-3",
        "training.iters=42",
        "run.seed=7",
        "training.collector.normalize_reward=false",
    ])
    assert cfg.training["ppo"]["lr"] == 1e-3 and isinstance(cfg.training["ppo"]["lr"], float)
    assert cfg.training["iters"] == 42
    assert cfg.run["seed"] == 7
    assert cfg.training["collector"]["normalize_reward"] is False


def test_apply_overrides_rejects_bad_input():
    cfg = Config.from_experiment("micro-rts/base_rlFS_expert")
    try:
        cfg.apply_overrides(["training.ppo.lr"])  # no '='
        assert False
    except ValueError:
        pass


def test_from_experiment_is_cwd_independent(tmp_path):
    here = os.getcwd()
    try:
        os.chdir(tmp_path)
        cfg = Config.from_experiment("micro-rts/base_rlFS_expert")
        assert cfg.run["name"] == "base_rlFS_expert"
    finally:
        os.chdir(here)
