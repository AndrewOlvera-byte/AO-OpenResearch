"""Config loading: deep-merge, interpolation, nested training, CWD-independence."""

import os

from core.config import Config


def test_from_experiment_merges_and_interpolates():
    cfg = Config.from_experiment("micro-rts/rl/ppo/base_rlFS_expert")
    assert cfg.model["type"] == "cnn_mlp"
    assert cfg.wandb["project"] == "micro-rts"
    # ${run.name} resolved against the merged (overridden) run name.
    assert cfg.wandb["run_name"] == "base_rlFS_expert"
    # nested training sections present after merge.
    assert "ppo" in cfg.training and "curriculum" in cfg.training and "eval" in cfg.training
    assert cfg.training["curriculum"]["type"] == "ppo"
    assert isinstance(cfg.training["ppo"]["lr"], float)


def test_apply_overrides_types_and_paths():
    cfg = Config.from_experiment("micro-rts/rl/ppo/base_rlFS_expert")
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
    cfg = Config.from_experiment("micro-rts/rl/ppo/base_rlFS_expert")
    try:
        cfg.apply_overrides(["training.ppo.lr"])  # no '='
        assert False
    except ValueError:
        pass


def test_from_experiment_is_cwd_independent(tmp_path):
    here = os.getcwd()
    try:
        os.chdir(tmp_path)
        cfg = Config.from_experiment("micro-rts/rl/ppo/base_rlFS_expert")
        assert cfg.run["name"] == "base_rlFS_expert"
    finally:
        os.chdir(here)


def test_discrete_tokenizer_matrix_is_matched_and_isolated():
    experiments = [
        "micro-rts/tokenizer/discrete_v3/pretrain_discrete_tokenizer_v3",
        "micro-rts/tokenizer/discrete_v3/pretrain_discrete_tokenizer_v3_compact_shallow",
    ]
    configs = [Config.from_experiment(name) for name in experiments]
    expected = [
        (2, 4, 512, 272),
        (2, 2, 1024, 136),
    ]
    corpus = "/data/micro-rts/wm_v2_pretrain__20260713-030227__5a273944.h5"
    for cfg, (downsample, depth, size, base_codes) in zip(configs, expected):
        tokenizer = cfg.model["tokenizer"]
        semantic = (16 // downsample) ** 2 + tokenizer["n_global_tokens"]
        assert tokenizer["spatial_downsample"] == downsample
        assert tokenizer["codebook_depth"] == depth
        assert tokenizer["codebook_size"] == size
        assert semantic * depth == base_codes
        assert cfg.data["path"] == corpus
        assert cfg.run["seed"] == 0
        assert cfg.training["batch"] == 32
        assert cfg.training["fixed_val_seed"] == 0
    assert len({cfg.run["name"] for cfg in configs}) == len(configs)
    assert len({cfg.training["out"] for cfg in configs}) == len(configs)


def test_medium_temporal_jepa_matrix_is_architecture_and_data_matched():
    recon = Config.from_experiment(
        "micro-rts/paper/tokenizer/pretrain_medium_recon"
    )
    jepa = Config.from_experiment(
        "micro-rts/paper/tokenizer/pretrain_medium_jepa"
    )
    assert recon.model["tokenizer"] == jepa.model["tokenizer"]
    assert recon.data == jepa.data
    for key in (
        "steps",
        "batch",
        "lr",
        "weight_decay",
        "paired_batch_fraction",
        "val_frac",
        "fixed_val_seed",
    ):
        assert recon.training[key] == jepa.training[key]
    assert recon.training["objective"] == "reconstruction"
    assert jepa.training["objective"] == "temporal_jepa"
    assert recon.model["tokenizer"]["d_latent"] == 320
    assert recon.model["tokenizer"]["depth"] == 4

    action_recon = Config.from_experiment(
        "micro-rts/paper/tokenizer/pretrain_action_encoder_medium_recon"
    )
    action_jepa = Config.from_experiment(
        "micro-rts/paper/tokenizer/pretrain_action_encoder_medium_jepa"
    )
    assert action_recon.model == action_jepa.model
    assert action_recon.data == action_jepa.data
    ignored = {"tokenizer_ckpt", "out"}
    assert {
        key: value
        for key, value in action_recon.training.items()
        if key not in ignored
    } == {
        key: value
        for key, value in action_jepa.training.items()
        if key not in ignored
    }
    assert action_recon.training["tokenizer_ckpt"].endswith("medium_recon.pt")
    assert action_jepa.training["tokenizer_ckpt"].endswith("medium_jepa.pt")
