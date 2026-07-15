"""AtariDreamTrainer smoke test — full model-based loop on a tiny Pong env."""

import types

from entrypoints.AtariDreamTrainer import AtariDreamTrainer


def _cfg():
    return types.SimpleNamespace(
        run={"name": "atari-smoke", "device": "cpu", "verbose": False, "seed": 0,
             "ckpt_dir": "/tmp/atari_ck"},
        model={"type": "atari_dreamerv4",
               "tokenizer": {"base_channels": 16, "d_latent": 16},
               "dynamics": {"d_model": 32, "depth": 2, "n_heads": 4, "n_register": 2},
               "actor_critic": {"hidden": [64], "imagine_horizon": 4}},
        data={},
        training={"dreamer": {}, "env": {"game": "pong", "num_envs": 4}},
        wandb={},
    )


def test_atari_trainer_smoke():
    trainer = AtariDreamTrainer(_cfg())
    trainer.use_wandb = False
    metrics = trainer.smoke_test()
    for key in ("wm/total", "wm/dynamics", "flow/matching", "ac/actor_loss", "ac/critic_loss"):
        assert key in metrics and metrics[key] == metrics[key], f"bad {key}"
