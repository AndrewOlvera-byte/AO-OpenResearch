"""Atari DreamerV4 architecture tests — shapes, config, optimizer split (no ALE)."""

import torch

import models.dreamer as D

OBS = (1, 64, 64)
NA = 6


def _model():
    cfg = D.AtariDreamerConfig.from_dict({
        "tokenizer": {"base_channels": 16, "d_latent": 16},
        "dynamics": {"d_model": 32, "depth": 2, "n_heads": 4, "n_register": 2},
        "actor_critic": {"hidden": [64], "imagine_horizon": 4},
    })
    return D.AtariDreamerV4(OBS, NA, cfg)


def test_config_partial_and_ignores_unknown():
    cfg = D.AtariDreamerConfig.from_dict({"type": "atari_dreamerv4", "world_lr": 5e-4,
                                          "dynamics": {"depth": 6, "bogus": 1}})
    assert cfg.world_lr == 5e-4 and cfg.dynamics.depth == 6
    assert cfg.tokenizer.d_latent == D.TokenizerConfig().d_latent


def test_tokenizer_roundtrip_and_grad():
    m = _model()
    obs = torch.rand(3, *OBS)
    z = m.tokenizer.encode(obs)
    assert z.shape == (3, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    assert z.abs().max() <= 1.0 + 1e-5
    recon = m.tokenizer.decode(z)
    assert recon.shape == obs.shape and 0.0 <= float(recon.min()) and float(recon.max()) <= 1.0
    recon.sum().backward()
    assert m.tokenizer.to_latent.weight.grad is not None


def test_tokenizer_time_axis():
    m = _model()
    obs = torch.rand(2, 5, *OBS)
    z = m.tokenizer.encode(obs)
    assert z.shape == (2, 5, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    assert m.tokenizer.decode(z).shape == obs.shape


def test_world_model_shapes_and_backward():
    m = _model()
    B, T = 2, 4
    z = torch.rand(B, T, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    act = torch.randint(0, NA, (B, T))
    pred = m.world_model.predict(z, act)
    assert pred["next_latent"].shape == z.shape
    assert pred["reward"].shape == (B, T) and pred["continue_logit"].shape == (B, T)
    pred["next_latent"].sum().backward()
    assert m.world_model.action_embed.weight.grad is not None


def test_action_expert_act_evaluate_consistent():
    m = _model()
    z = torch.rand(3, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    action, logp, ent, val = m.action_expert.act(z)
    assert action.shape == (3,) and logp.shape == (3,) and val.shape == (3,)
    assert int(action.max()) < NA
    logp2, ent2, val2 = m.action_expert.evaluate(z, action)
    assert torch.allclose(logp, logp2, atol=1e-5) and torch.allclose(val, val2, atol=1e-5)


def test_step_and_imagine_shapes():
    m = _model()
    obs = torch.rand(4, *OBS)
    out = m.step(obs)
    assert set(out.keys()) >= {"action", "logprob", "value"} and out["action"].shape == (4,)
    z0 = torch.rand(5, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    imag = m.imagine(z0, horizon=4)
    assert imag["z"].shape == (5, 5, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    assert imag["action"].shape == (5, 4) and imag["reward"].shape == (5, 4)
    assert not imag["z"].requires_grad


def test_optimizer_partition_disjoint():
    m = _model()
    opts = m.build_optimizers()
    assert set(opts) == {"world", "actor", "critic"}
    ids = [id(p) for o in opts.values() for g in o.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    tgt = {id(p) for p in m.action_expert.target_critic.parameters()}
    assert tgt.isdisjoint(set(ids))
