"""Opponent-policy head tests (v4.4 — Step 1 BC head + Step 2 dream opponent).

Pure-tensor (no JVM). Verifies the properties the head exists for:

- shapes: trunk spatial features -> per-cell concatenated component logits,
- ``zero_inactive_components``: sampled conditional components are zeroed
  unless the sampled TYPE selects them (engine-executed support),
- the BC loss only reads labels at opponent source cells (idle enemy unit)
  and skips terminal-splice slots (label belongs to the next episode),
- gradients flow through BOTH the head and the transformer trunk (the
  auxiliary-conditioning point of reading the trunk, not the raw latent),
- ``sample_opponent`` respects the source mask and the component support,
- ``imagine`` plays the head's opponent when present (and can be disabled).
"""

import torch

import models.dreamer as D
from models.dreamer.world_model import (
    opponent_source_cells, zero_inactive_components,
)
from loss.dreamer import dynamics_loss, opponent_bc_loss

OBS = (27, 8, 8)                       # real MicroRTS channel layout, small board
NVEC = [64, 6, 4, 4, 4, 4, 7, 49]
TOTAL_LOGITS = sum(NVEC[1:])           # 78
OWNER_ENEMY_CH, ACTION_NONE_CH = 12, 21


def _cfg(**dyn):
    d = {"d_model": 32, "depth": 2, "n_heads": 4, "n_register": 2,
         "k_max": 4, "action_emb": 8, "action_channels": 16,
         "opp_dropout": 0.0, "opp_head": True, "opp_head_channels": 16}
    d.update(dyn)
    return D.DreamerV4Config.from_dict({
        "tokenizer": {"d_latent": 8, "enc_channels": 16},
        "dynamics": d,
        "actor_critic": {"dec_channels": 16, "imagine_horizon": 3,
                         "imagine_flow_steps": 2, "imagine_context": 2},
    })


def _model(**dyn):
    torch.manual_seed(0)
    return D.DreamerV4(OBS, NVEC, _cfg(**dyn))


def _rand_actions(B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    comps = [torch.randint(0, n, (B, T, NVEC[0], 1), generator=g)
             for n in NVEC[1:]]
    return torch.cat(comps, dim=-1)


def _obs_with_sources(B, T, source_cells):
    """Zeros obs with an idle enemy unit at each (row, col) in source_cells."""
    obs = torch.zeros(B, T, *OBS)
    for r, c in source_cells:
        obs[..., OWNER_ENEMY_CH, r, c] = 1.0
        obs[..., ACTION_NONE_CH, r, c] = 1.0
    return obs


def test_head_shapes():
    m = _model()
    wm = m.world_model
    assert wm.opp_head is not None
    feats = torch.randn(2, 3, wm.n_spatial, wm.cfg.d_model)
    logits = wm.opp_head(feats)
    assert logits.shape == (2, 3, NVEC[0], TOTAL_LOGITS)
    # no-time-axis input works too
    assert wm.opp_head(feats[:, 0]).shape == (2, NVEC[0], TOTAL_LOGITS)


def test_opp_head_off_by_default():
    cfg = D.DreamerV4Config.from_dict({"dynamics": {"d_model": 32, "depth": 2,
                                                    "n_heads": 4, "k_max": 4}})
    m = D.DreamerV4(OBS, NVEC, cfg)
    assert m.world_model.opp_head is None


def test_zero_inactive_components():
    a = torch.full((1, 4, 7), 3, dtype=torch.long)
    a[0, 0, 0] = 0   # NOOP  -> everything conditional zeroed
    a[0, 1, 0] = 1   # move  -> only comp 1 (move dir) kept
    a[0, 2, 0] = 4   # produce -> comps 4 (dir) and 5 (unit type) kept
    a[0, 3, 0] = 5   # attack -> only comp 6 kept
    z = zero_inactive_components(a)
    assert z[0, 0].tolist() == [0, 0, 0, 0, 0, 0, 0]
    assert z[0, 1].tolist() == [1, 3, 0, 0, 0, 0, 0]
    assert z[0, 2].tolist() == [4, 0, 0, 0, 3, 3, 0]
    assert z[0, 3].tolist() == [5, 0, 0, 0, 0, 0, 3]


def test_opponent_source_cells_reads_enemy_idle_planes():
    obs = _obs_with_sources(1, 2, [(1, 2), (5, 5)])
    # a busy enemy unit (no idle plane) must NOT count
    obs[..., OWNER_ENEMY_CH, 7, 7] = 1.0
    src = opponent_source_cells(obs)
    assert src.shape == (1, 2, 64)
    assert bool(src[0, 0, 1 * 8 + 2]) and bool(src[0, 0, 5 * 8 + 5])
    assert not bool(src[0, 0, 7 * 8 + 7])
    assert int(src[0, 0].sum()) == 2


def test_bc_loss_only_reads_source_cells_and_valid_slots():
    m = _model()
    wm = m.world_model
    B, T = 2, 4
    feats = torch.randn(B, T, wm.n_spatial, wm.cfg.d_model)
    obs = _obs_with_sources(B, T, [(1, 2)])
    opp = _rand_actions(B, T, seed=3)
    src_idx, junk_idx = 1 * 8 + 2, 6 * 8 + 6

    loss_a, metrics = opponent_bc_loss(wm, feats, opp, obs)
    # perturbing a NON-source cell's label changes nothing
    opp2 = opp.clone()
    opp2[:, :, junk_idx] = (opp2[:, :, junk_idx] + 1) % 4
    loss_b, _ = opponent_bc_loss(wm, feats, opp2, obs)
    assert torch.allclose(loss_a, loss_b)
    # perturbing the SOURCE cell's type does change the loss
    opp3 = opp.clone()
    opp3[:, :, src_idx, 0] = (opp3[:, :, src_idx, 0] + 1) % 6
    loss_c, _ = opponent_bc_loss(wm, feats, opp3, obs)
    assert not torch.allclose(loss_a, loss_c)
    assert 0.0 <= metrics["opp_bc/type_acc"] <= 1.0

    # terminal-splice slot: is_first[t+1] invalidates the label at slot t
    is_first = torch.zeros(B, T, dtype=torch.bool)
    is_first[:, 2] = True                       # slot 1 is a terminal splice
    loss_d, _ = opponent_bc_loss(wm, feats, opp, obs, is_first)
    opp4 = opp.clone()
    opp4[:, 1, src_idx, 0] = (opp4[:, 1, src_idx, 0] + 1) % 6
    loss_e, _ = opponent_bc_loss(wm, feats, opp4, obs, is_first)
    assert torch.allclose(loss_d, loss_e), \
        "label at a terminal-splice slot leaked into the BC loss"


def test_dynamics_loss_integration_and_grads():
    m = _model()
    B, T = 3, 5
    obs = _obs_with_sources(B, T, [(1, 2), (4, 4)])
    a, o = _rand_actions(B, T, seed=1), _rand_actions(B, T, seed=2)
    reward, cont = torch.randn(B, T), torch.ones(B, T)
    is_first = torch.zeros(B, T, dtype=torch.bool)
    z = m.tokenizer.encode(obs).detach()

    loss, metrics = dynamics_loss(m, z, a, reward, cont, is_first,
                                  opponent_action=o, obs=obs, opp_bc_coef=1.0)
    assert "opp_bc/loss" in metrics and torch.isfinite(loss)
    loss.backward()
    wm = m.world_model
    head_g = [p.grad for p in wm.opp_head.parameters() if p.grad is not None]
    trunk_g = [p.grad for p in wm.transformer.parameters() if p.grad is not None]
    assert head_g and any(g.abs().sum() > 0 for g in head_g)
    assert trunk_g and any(g.abs().sum() > 0 for g in trunk_g)

    # coef 0 (or no obs) skips the BC term entirely
    _, metrics0 = dynamics_loss(m, z, a, reward, cont, is_first,
                                opponent_action=o, obs=obs, opp_bc_coef=0.0)
    assert "opp_bc/loss" not in metrics0


def test_sample_opponent_masks_sources_and_components():
    m = _model()
    wm = m.world_model
    feats = torch.randn(4, wm.n_spatial, wm.cfg.d_model)
    src = torch.zeros(4, NVEC[0], dtype=torch.bool)
    src[:, 10] = True
    act = wm.sample_opponent(feats, src)
    assert act.shape == (4, NVEC[0], 7)
    off_src = act.clone()
    off_src[:, 10] = 0
    assert off_src.abs().sum() == 0, "non-source cells must be NOOP"
    assert torch.equal(act, zero_inactive_components(act)), \
        "sampled conditional components must follow the sampled type"


def test_imagine_uses_head_and_can_disable():
    m = _model()
    obs = _obs_with_sources(2, 2, [(1, 2)])
    z = m.tokenizer.encode(obs)
    a = _rand_actions(2, 2, seed=5)
    im = m.imagine(z, horizon=3, ctx_action=a,
                   ctx_opponent_action=_rand_actions(2, 2, seed=6))
    assert im["z"].shape[1] == 4 and im["reward"].shape == (2, 3)
    im2 = m.imagine(z, horizon=3, ctx_action=a, use_opp_head=False)
    assert im2["z"].shape[1] == 4

    # explicit opponent_policy still takes precedence (no head sampling needed)
    def opp_policy(z_t, mask):
        return torch.zeros(z_t.shape[0], NVEC[0], 7, dtype=torch.long)

    im3 = m.imagine(z, horizon=2, ctx_action=a, opponent_policy=opp_policy)
    assert im3["z"].shape[1] == 3
