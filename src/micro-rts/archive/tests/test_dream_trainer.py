"""DreamerRLTrainer smoke tests — collect + WM update + actor-critic + analysis.

Needs the MicroRTS JVM, so these run under ``--forked`` like the other env-backed
tests. Verifies the whole DreamerV4 model-based loop wires together end to end,
produces finite losses across all three optimizers, and that config-driven module
freezing degrades gracefully (frozen groups simply skip their updates).
"""

import types

import pytest

from trainers.DreamerRLTrainer import DreamerRLTrainer


def _cfg(freeze=None):
    model = {"type": "dreamerv4",
             "tokenizer": {"d_latent": 8, "enc_channels": 16},
             "dynamics": {"d_model": 32, "depth": 2, "n_heads": 4, "n_register": 2,
                          "k_max": 4, "action_emb": 8, "action_channels": 16},
             "actor_critic": {"dec_channels": 16, "imagine_horizon": 4,
                              "imagine_flow_steps": 2, "imagine_context": 2}}
    if freeze:
        model["freeze"] = freeze
    return types.SimpleNamespace(
        run={"name": "dream-smoke", "device": "cpu", "verbose": False, "ckpt_dir": "/tmp/ck"},
        model=model,
        data={},
        training={"dreamer": {}},
        wandb={},
    )


def test_dream_trainer_smoke():
    trainer = DreamerRLTrainer(_cfg())
    trainer.use_wandb = False
    metrics = trainer.smoke_test()
    for key in ("wm/total", "wm/reward", "flow/matching", "ac/actor_loss",
                "ac/critic_loss", "dyn/openloop_mse", "dyn/z_motion"):
        assert key in metrics, f"missing {key}"
        assert metrics[key] == metrics[key], f"non-finite {key}"


def test_dream_trainer_smoke_frozen_world():
    """RL-on-frozen-world-model staging: no world optimizer, loop still runs."""
    trainer = DreamerRLTrainer(_cfg(freeze={"tokenizer": True, "world_model": True}))
    trainer.use_wandb = False
    metrics = trainer.smoke_test()
    assert "wm/total" not in metrics            # world update skipped
    for key in ("ac/actor_loss", "ac/critic_loss"):
        assert key in metrics and metrics[key] == metrics[key]


def test_dream_trainer_smoke_online_mode():
    """mode=online: actor/critic learn from REAL replay sequences (no imagination,
    no world update); the loss reports which rows carried a legal action."""
    cfg = _cfg(freeze={"tokenizer": True, "world_model": True})
    cfg.training["dreamer"]["mode"] = "online"
    trainer = DreamerRLTrainer(cfg)
    trainer.use_wandb = False
    metrics = trainer.smoke_test()
    assert "wm/total" not in metrics
    for key in ("ac/actor_loss", "ac/critic_loss", "ac/valid_frac"):
        assert key in metrics and metrics[key] == metrics[key], f"bad {key}"
    assert 0.0 < metrics["ac/valid_frac"] <= 1.0


def test_dream_trainer_imagination_mode_forces_freeze():
    """mode=imagination with a trainable-WM yaml must not silently become hybrid:
    the trainer freezes tokenizer+world_model itself."""
    cfg = _cfg()                                # nothing frozen in the yaml
    cfg.training["dreamer"]["mode"] = "imagination"
    trainer = DreamerRLTrainer(cfg)
    trainer.use_wandb = False
    metrics = trainer.smoke_test()
    assert "wm/total" not in metrics            # world update was forced off
    for key in ("ac/actor_loss", "ac/critic_loss"):
        assert key in metrics and metrics[key] == metrics[key]


def test_dream_trainer_rejects_unknown_mode():
    cfg = _cfg()
    cfg.training["dreamer"]["mode"] = "dreamy"
    with pytest.raises(ValueError, match="mode"):
        DreamerRLTrainer(cfg)


def test_real_actor_critic_losses_shapes_and_grads():
    """Online-RL loss unit test (no JVM): finite losses, gradient flow into the
    actor decoder and critic only, and zero-mask rows excluded from the actor."""
    import torch

    from core.registry import build
    import models.dreamer  # noqa: F401
    from loss.dreamer import real_actor_critic_losses

    nvec = [64, 6, 4, 4, 4, 4, 7, 49]
    model = build("model", type="dreamerv4", obs_shape=(6, 8, 8), action_nvec=nvec,
                  device="cpu",
                  tokenizer={"d_latent": 8, "enc_channels": 16},
                  dynamics={"d_model": 32, "depth": 2, "n_heads": 4, "n_register": 2,
                            "k_max": 4, "action_emb": 8, "action_channels": 16},
                  actor_critic={"dec_channels": 16})
    B, T, cells, mask_w = 2, 5, 64, 1 + sum(nvec[1:])
    z = torch.randn(B, T, model.tokenizer.n_spatial, model.tokenizer.d_latent)
    batch = {
        "action": torch.zeros(B, T, cells, 7, dtype=torch.long),
        "reward": torch.randn(B, T),
        "cont": torch.ones(B, T),
        "mask": torch.ones(B, T, cells, mask_w, dtype=torch.bool),
        "is_first": torch.zeros(B, T, dtype=torch.bool),
    }
    batch["cont"][0, 2] = 0.0                    # an episode ends mid-sequence
    batch["mask"][1, 1] = False                  # a no-legal-action row

    actor_loss, critic_loss, m = real_actor_critic_losses(model, z, batch)
    assert actor_loss.isfinite() and critic_loss.isfinite()
    assert m["ac/valid_frac"] < 1.0              # the zero-mask row was excluded

    actor_loss.backward(retain_graph=True)
    critic_loss.backward()
    ae = model.action_expert
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in ae.decoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in ae.critic.parameters())
    assert all(p.grad is None for p in model.world_model.parameters())
