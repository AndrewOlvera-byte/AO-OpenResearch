"""DreamerV4 architecture tests — shapes, config, policy protocol, optimizer split,
shortcut-forcing conditioning, temporal causality, and freezing.

Pure-tensor (no JVM), so they run without a MicroRTS env. A small 8x8 grid keeps
them fast while exercising the real code paths (n_spatial = n_action = 4,
mask_width = 79).
"""

import torch
import pytest

import models.dreamer as D
from environments.base import Policy

OBS = (6, 8, 8)
NVEC = [64, 6, 4, 4, 4, 4, 7, 49]
MASK_W = 1 + sum(NVEC[1:])  # 79


def _cfg():
    return D.DreamerV4Config.from_dict({
        "tokenizer": {"d_latent": 8, "enc_channels": 16},
        "dynamics": {"d_model": 32, "depth": 2, "n_heads": 4, "n_register": 2,
                     "k_max": 4, "action_emb": 8, "action_channels": 16},
        "actor_critic": {"dec_channels": 16, "imagine_horizon": 4,
                         "imagine_flow_steps": 2, "imagine_context": 2},
    })


def _model(**freeze):
    cfg = _cfg()
    for k, v in freeze.items():
        setattr(cfg.freeze, k, v)
    return D.DreamerV4(OBS, NVEC, cfg)


def _actions(B, T):
    return torch.zeros(B, T, NVEC[0], 7, dtype=torch.long)


def test_config_from_dict_partial_and_ignores_unknown():
    cfg = D.DreamerV4Config.from_dict({"type": "dreamerv4", "world_lr": 5e-4,
                                       "dynamics": {"depth": 7, "bogus": 1},
                                       "freeze": {"world_model": True}})
    assert cfg.world_lr == 5e-4
    assert cfg.dynamics.depth == 7
    assert cfg.freeze.world_model and not cfg.freeze.tokenizer
    assert cfg.tokenizer.d_latent == D.TokenizerConfig().d_latent  # default preserved


def test_tokenizer_roundtrip_shapes_and_grad():
    m = _model()
    obs = torch.rand(3, *OBS)
    z = m.tokenizer.encode(obs)
    assert z.shape == (3, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    assert z.abs().max() <= 1.0 + 1e-5  # tanh bottleneck
    recon = m.tokenizer.decode(z)
    assert recon.shape == obs.shape
    recon.sum().backward()
    assert m.tokenizer.to_latent.weight.grad is not None
    assert m.tokenizer.decode_mask(z.detach()).shape == (3, NVEC[0], MASK_W)


# --- action encoder: multiple CNN-GridNet embeddings ----------------------
def test_action_encoder_emits_one_token_per_bottleneck_cell():
    m = _model()
    enc = m.world_model.action_encoder
    assert enc.n_action == m.tokenizer.n_spatial  # spatially aligned token grids
    tok = enc(_actions(2, 3))
    assert tok.shape == (2, 3, enc.n_action, m.cfg.dynamics.d_model)


def test_action_encoder_no_action_flag_replaces_signal():
    m = _model()
    enc = m.world_model.action_encoder
    act = torch.randint(0, 4, (2, 3, NVEC[0], 7))
    flagged = torch.zeros(2, 3, dtype=torch.bool)
    flagged[:, 1] = True
    tok = enc(act, flagged)
    tok_all_first = enc(torch.zeros_like(act), torch.ones(2, 3, dtype=torch.bool))
    # A flagged frame's tokens must not depend on the (stale) action content.
    assert torch.allclose(tok[:, 1], tok_all_first[:, 1], atol=1e-6)
    assert not torch.allclose(tok[:, 0], tok_all_first[:, 0], atol=1e-3)


def test_shift_actions_alignment():
    m = _model()
    act = torch.arange(2 * 3).view(2, 3, 1, 1).expand(2, 3, NVEC[0], 7).contiguous()
    is_first = torch.zeros(2, 3, dtype=torch.bool)
    is_first[:, 2] = True
    prev, no_act = m.world_model.shift_actions(act, is_first)
    assert torch.all(prev[:, 1] == act[:, 0])  # slot t carries the action chosen at t-1
    assert torch.all(prev[:, 2] == act[:, 1])
    assert no_act[:, 0].all()                  # window start: no previous action
    assert no_act[:, 2].all()                  # episode start: previous action invalid
    assert not no_act[:, 1].any()


# --- world model: shortcut forcing plumbing --------------------------------
def test_contextualize_shapes():
    m = _model()
    B, T = 2, 4
    z = torch.rand(B, T, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    ctx = m.world_model.contextualize(z, _actions(B, T))
    assert ctx["h"].shape == (B, T, m.tokenizer.n_spatial, m.cfg.dynamics.d_model)
    assert ctx["reward"].shape == (B, T)
    assert ctx["continue_logit"].shape == (B, T)


def test_denoise_zero_init_predicts_zero():
    m = _model()
    B, T = 2, 4
    wm = m.world_model
    z_tilde = torch.randn(B, T, wm.n_spatial, wm.d_latent)
    sig = torch.randint(0, wm.k_max, (B, T))
    stp = torch.full((B, T), wm.emax, dtype=torch.long)
    x1 = wm.denoise(z_tilde, _actions(B, T), None, sig, stp)
    assert x1.shape == z_tilde.shape
    assert torch.all(x1 == 0)  # flow_x_head is zero-init (upstream)


def test_denoise_is_temporally_causal():
    torch.manual_seed(0)
    m = _model()
    wm = m.world_model
    torch.nn.init.normal_(wm.flow_x_head.weight, std=0.1)  # un-zero the head
    B, T = 1, 4
    z = torch.randn(B, T, wm.n_spatial, wm.d_latent)
    sig = torch.full((B, T), wm.k_max - 1, dtype=torch.long)
    stp = torch.full((B, T), wm.emax, dtype=torch.long)
    act = _actions(B, T)
    base = wm.denoise(z, act, None, sig, stp)
    z2 = z.clone()
    z2[:, -1] += 10.0                       # perturb only the last frame
    out = wm.denoise(z2, act, None, sig, stp)
    assert torch.allclose(base[:, :-1], out[:, :-1], atol=1e-5)  # past unaffected
    assert not torch.allclose(base[:, -1], out[:, -1], atol=1e-3)


def test_sample_next_shape_and_finite():
    m = _model()
    wm = m.world_model
    B, Tc = 3, 2
    z_ctx = torch.rand(B, Tc, wm.n_spatial, wm.d_latent) * 2 - 1
    z_next = wm.sample_next(z_ctx, _actions(B, Tc), None, steps=2)
    assert z_next.shape == (B, wm.n_spatial, wm.d_latent)
    assert torch.isfinite(z_next).all()
    with pytest.raises(AssertionError):
        wm.sample_next(z_ctx, _actions(B, Tc), None, steps=3)  # not a power of two


# --- action expert / policy protocol ---------------------------------------
def test_action_expert_act_and_evaluate_consistent():
    m = _model()
    z = torch.rand(3, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    mask = torch.ones(3, NVEC[0], MASK_W)
    action, logp, ent, val = m.action_expert.act(z, mask)
    assert action.shape == (3, NVEC[0], 7)
    assert logp.shape == (3,) and ent.shape == (3,) and val.shape == (3,)
    logp2, ent2, val2 = m.action_expert.evaluate(z, action, mask)
    assert torch.allclose(logp, logp2, atol=1e-5)
    assert torch.allclose(val, val2, atol=1e-5)


def test_step_satisfies_policy_protocol_and_returns_latents():
    m = _model()
    assert isinstance(m, Policy)
    obs = torch.rand(4, *OBS)
    mask = torch.ones(4, NVEC[0], MASK_W)
    out = m.step(obs, mask)
    assert set(out.keys()) >= {"action", "logprob", "value", "z"}
    assert out["action"].shape == (4, NVEC[0], 7)
    assert out["z"].shape == (4, m.tokenizer.n_spatial, m.tokenizer.d_latent)


# --- imagination ------------------------------------------------------------
def test_imagine_single_frame_shapes_detached():
    m = _model()
    z0 = torch.rand(5, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    imag = m.imagine(z0, horizon=4)
    assert imag["z"].shape == (5, 5, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    assert imag["action"].shape == (5, 4, NVEC[0], 7)
    assert imag["mask"].shape == (5, 4, NVEC[0], MASK_W)
    assert imag["reward"].shape == (5, 4)
    assert imag["cont"].shape == (5, 4)
    assert not imag["z"].requires_grad  # imagination is under no_grad
    assert imag["z"].abs().max() <= 1.0 + 1e-5  # clamped to the tanh range


def test_imagine_with_context_window():
    m = _model()
    B, Tc = 3, 2
    z0 = torch.rand(B, Tc, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    a0 = _actions(B, Tc)
    f0 = torch.zeros(B, Tc, dtype=torch.bool)
    imag = m.imagine(z0, horizon=3, ctx_action=a0, ctx_is_first=f0)
    assert imag["z"].shape == (B, 4, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    # First returned frame is the last context frame (the actor's start state).
    assert torch.allclose(imag["z"][:, 0], z0[:, -1])
    assert imag["reward"].shape == (B, 3)


def test_imagine_context_requires_actions():
    m = _model()
    z0 = torch.rand(2, 3, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    with pytest.raises(AssertionError):
        m.imagine(z0, horizon=2)


def test_open_loop_shapes():
    m = _model()
    B, T = 2, 5
    z = torch.rand(B, T, m.tokenizer.n_spatial, m.tokenizer.d_latent) * 2 - 1
    pred = m.open_loop(z, _actions(B, T), context=2)
    assert pred.shape == (B, T - 2, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    assert torch.isfinite(pred).all()


# --- optimizers / freezing ---------------------------------------------------
def test_build_optimizers_partition_is_disjoint_and_complete():
    m = _model()
    opts = m.build_optimizers()
    assert set(opts) == {"world", "actor", "critic"}
    ids = [id(p) for o in opts.values() for g in o.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter is owned by two optimizers"
    # Target critic must be excluded (it is an EMA copy, never optimized).
    target_ids = {id(p) for p in m.action_expert.target_critic.parameters()}
    assert target_ids.isdisjoint(set(ids))


def test_freeze_drops_optimizer_groups_and_grads():
    m = _model(tokenizer=True, world_model=True)
    opts = m.build_optimizers()
    assert set(opts) == {"actor", "critic"}
    assert all(not p.requires_grad for p in m.tokenizer.parameters())
    assert all(not p.requires_grad for p in m.world_model.parameters())

    m2 = _model(actor=True, critic=True)
    opts2 = m2.build_optimizers()
    assert set(opts2) == {"world"}
    assert all(not p.requires_grad for p in m2.action_expert.critic.parameters())
    assert all(not p.requires_grad for p in m2.action_expert.action_conv.parameters())


def test_partial_freeze_keeps_world_group_for_unfrozen_half():
    m = _model(tokenizer=True)  # world model still trains
    opts = m.build_optimizers()
    assert "world" in opts
    world_ids = {id(p) for g in opts["world"].param_groups for p in g["params"]}
    assert world_ids == {id(p) for p in m.world_model.parameters()}
