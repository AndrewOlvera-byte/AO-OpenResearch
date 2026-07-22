"""Joint-action world-model tests (v3, NEXT_PLAN.md Workstream D).

Pure-tensor (no JVM). Verifies the properties the v3 change exists for:

- the two action channels are *distinguishable* (separate embedding tables:
  swapping self/opponent streams changes the encoder output),
- the opponent channel actually conditions the transformer pass,
- ``unknown_opp`` handling: ``opp_action=None`` equals an all-unknown mask,
- opponent dropout fires in training mode only and only when configured,
- the full plumbing (loss / sample_next / open_loop / imagine / probe) accepts
  the opponent stream end to end.
"""

import torch

import models.dreamer as D
from loss.dreamer import dynamics_loss, shortcut_forcing_loss, world_model_loss

OBS = (6, 8, 8)
NVEC = [64, 6, 4, 4, 4, 4, 7, 49]
MASK_W = 1 + sum(NVEC[1:])  # 79


def _cfg(**dyn):
    d = {"d_model": 32, "depth": 2, "n_heads": 4, "n_register": 2,
         "k_max": 4, "action_emb": 8, "action_channels": 16, "opp_dropout": 0.0}
    d.update(dyn)
    return D.DreamerV4Config.from_dict({
        "tokenizer": {"d_latent": 8, "enc_channels": 16},
        "dynamics": d,
        "actor_critic": {"dec_channels": 16, "imagine_horizon": 3,
                         "imagine_flow_steps": 2, "imagine_context": 2},
    })


def _model(**dyn):
    return D.DreamerV4(OBS, NVEC, _cfg(**dyn))


def _rand_actions(B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    comps = [torch.randint(0, n, (B, T, NVEC[0], 1), generator=g)
             for n in NVEC[1:]]
    return torch.cat(comps, dim=-1)


def test_action_encoder_channels_are_distinguishable():
    """emb(self=a, opp=b) must differ from emb(self=b, opp=a): a shared-table
    sum would be order-invariant and erase who did what."""
    m = _model()
    enc = m.world_model.action_encoder
    a = _rand_actions(2, 3, seed=1)
    b = _rand_actions(2, 3, seed=2)
    # Break the zero-init symmetry: fresh embeddings are random, but unknown/first
    # params are zeros — actual tables differ, which is what we exercise.
    out_ab = enc(a, None, opp_action=b)
    out_ba = enc(b, None, opp_action=a)
    assert out_ab.shape == out_ba.shape
    assert not torch.allclose(out_ab, out_ba), \
        "swapping self/opponent action streams left the encoder output unchanged"


def test_opponent_channel_conditions_the_pass():
    m = _model()
    wm = m.world_model
    B, T = 2, 4
    z = torch.randn(B, T, wm.n_spatial, wm.d_latent)
    a = _rand_actions(B, T, seed=3)
    o1 = _rand_actions(B, T, seed=4)
    o2 = _rand_actions(B, T, seed=5)
    ctx1 = wm.contextualize(z, a, opponent_action=o1)
    ctx2 = wm.contextualize(z, a, opponent_action=o2)
    assert not torch.allclose(ctx1["h"], ctx2["h"]), \
        "different opponent actions produced identical transformer outputs"


def test_opp_none_equals_all_unknown():
    m = _model()
    enc = m.world_model.action_encoder
    a = _rand_actions(2, 3, seed=6)
    o = _rand_actions(2, 3, seed=7)
    unk = torch.ones(2, 3, dtype=torch.bool)
    out_none = enc(a, None, opp_action=None)
    out_unk = enc(a, None, opp_action=o, opp_unknown=unk)
    assert torch.allclose(out_none, out_unk, atol=1e-6), \
        "opp_action=None must equal an explicit all-unknown opponent"


def test_opp_dropout_train_only():
    from loss.dreamer import _opp_dropout_mask

    torch.manual_seed(0)
    opp = _rand_actions(4, 8)
    m = _model(opp_dropout=0.5)
    m.train()
    mask = _opp_dropout_mask(m, opp, 4, 8, "cpu")
    assert mask is not None and mask.dtype == torch.bool and mask.any()
    m.eval()
    assert _opp_dropout_mask(m, opp, 4, 8, "cpu") is None
    m0 = _model(opp_dropout=0.0)
    m0.train()
    assert _opp_dropout_mask(m0, opp, 4, 8, "cpu") is None
    assert _opp_dropout_mask(m, None, 4, 8, "cpu") is None


def test_losses_accept_opponent_stream_and_grads_flow():
    torch.manual_seed(0)
    m = _model(opp_dropout=0.15)
    m.train()
    B, T = 3, 6
    obs = torch.rand(B, T, *OBS)
    a = _rand_actions(B, T, seed=8)
    o = _rand_actions(B, T, seed=9)
    reward = torch.randn(B, T)
    cont = torch.ones(B, T)
    mask = torch.ones(B, T, NVEC[0], MASK_W)
    is_first = torch.zeros(B, T, dtype=torch.bool)
    is_first[:, 0] = True

    loss, metrics, z = world_model_loss(m, obs, a, reward, cont, mask, is_first,
                                        opponent_action=o)
    assert torch.isfinite(loss)

    z = m.tokenizer.encode(obs).detach()
    dloss, dm = dynamics_loss(m, z, a, reward, cont, is_first, opponent_action=o)
    assert torch.isfinite(dloss)
    dloss.backward()
    enc = m.world_model.action_encoder
    assert enc.opp_embeds[0].weight.grad is not None, \
        "no gradient reached the opponent embedding tables"
    assert enc.embeds[0].weight.grad is not None

    floss, fm = shortcut_forcing_loss(m, z, a, is_first, opponent_action=o)
    assert torch.isfinite(floss)


def test_generation_paths_accept_opponent_stream():
    torch.manual_seed(0)
    m = _model()
    m.eval()
    wm = m.world_model
    B, T = 2, 5
    z = torch.randn(B, T, wm.n_spatial, wm.d_latent).clamp(-1, 1)
    a = _rand_actions(B, T, seed=10)
    o = _rand_actions(B, T, seed=11)
    is_first = torch.zeros(B, T, dtype=torch.bool)

    nxt = wm.sample_next(z, a, is_first, steps=2, opponent_action=o)
    assert nxt.shape == (B, wm.n_spatial, wm.d_latent)

    pred = m.open_loop(z, a, is_first, context=2, opponent_action=o)
    assert pred.shape == (B, T - 2, wm.n_spatial, wm.d_latent)

    im = m.imagine(z[:, :2], horizon=3, ctx_action=a[:, :2],
                   ctx_is_first=is_first[:, :2], ctx_opponent_action=o[:, :2])
    assert im["z"].shape[1] == 4  # last ctx frame + 3 imagined

    # Two-sided dream: a callable opponent on the shared latent.
    calls = []

    def opp_policy(z_t, mask):
        calls.append(z_t.shape)
        return _rand_actions(z_t.shape[0], 1, seed=12)[:, 0]

    im2 = m.imagine(z[:, :2], horizon=3, ctx_action=a[:, :2],
                    ctx_is_first=is_first[:, :2],
                    ctx_opponent_action=o[:, :2], opponent_policy=opp_policy)
    assert len(calls) == 3
    assert im2["z"].shape == im["z"].shape


def test_counterfactual_probe_reports_gaps():
    from entrypoints.util.probes import counterfactual_action_probe

    torch.manual_seed(0)
    m = _model()
    m.eval()
    wm = m.world_model
    B, T = 4, 6
    z = torch.randn(B, T, wm.n_spatial, wm.d_latent).clamp(-1, 1)
    a = _rand_actions(B, T, seed=13)
    o = _rand_actions(B, T, seed=14)
    out = counterfactual_action_probe(m, z, a, o, context=2, flow_steps=2)
    for k in ("probe/mse_true", "probe/self_gap", "probe/opp_gap",
              "probe/both_gap", "probe/self_gap_growth", "probe/opp_gap_growth",
              "probe/self_gap_issued", "probe/opp_gap_issued",
              "probe/issued_cell_frac"):
        assert k in out and out[k] == out[k]           # present and not NaN
    # Buffer path: no opponent stream -> self-channel probe only.
    out2 = counterfactual_action_probe(m, z, a, None, context=2, flow_steps=2)
    assert "probe/self_gap" in out2 and "probe/opp_gap" not in out2
    assert "probe/self_gap_issued" in out2


def test_issued_cell_pooling_alignment():
    from entrypoints.util.probes import _issued_latent_cells

    B, T, ctx = 1, 4, 2
    a = torch.zeros(B, T, 64, 7, dtype=torch.long)     # 8x8 grid -> 2x2 latent
    a[0, ctx - 1, 3 * 8 + 5, 0] = 1                    # issue at (3,5) -> latent (0,1)
    iss = _issued_latent_cells(a, ctx, T - ctx, 8, 8)
    assert iss.shape == (B, T - ctx, 4)
    assert iss[0, 0].tolist() == [False, True, False, False]
    assert not iss[0, 1].any()                         # no issue driving frame ctx+1


def test_action_inject_add_is_identity_at_init_and_learns():
    torch.manual_seed(0)
    m_add = _model(action_inject="add")
    m_tok = _model(action_inject="tokens")
    # same weights for everything the two share
    sd = {k: v for k, v in m_add.state_dict().items()
          if "action_inject_proj" not in k}
    m_tok.load_state_dict(sd, strict=True)
    m_add.eval(); m_tok.eval()
    B, T = 2, 3
    z = torch.randn(B, T, m_add.world_model.n_spatial, m_add.world_model.d_latent)
    a = _rand_actions(B, T)
    sig = torch.zeros(B, T, dtype=torch.long)
    stp = torch.zeros(B, T, dtype=torch.long)
    with torch.no_grad():
        x_add = m_add.world_model.denoise(z, a, None, sig, stp)
        x_tok = m_tok.world_model.denoise(z, a, None, sig, stp)
    # zero-init projection: "add" starts exactly as "tokens"
    assert torch.allclose(x_add, x_tok, atol=1e-6)
    # and the projection is in the gradient path once the loss flows
    m_add.train()
    out = m_add.world_model.denoise(z, a, None, sig, stp)
    out.pow(2).mean().backward()
    g = m_add.world_model.action_inject_proj.weight.grad
    assert g is not None and torch.isfinite(g).all()


def test_action_inject_add_gives_percell_action_route():
    """With a nonzero projection, changing ONE cell's action must change the
    prediction — the direct route the tokens-only path never delivered."""
    torch.manual_seed(0)
    m = _model(action_inject="add")
    wm = m.world_model
    with torch.no_grad():
        wm.action_inject_proj.weight.normal_(0, 0.1)
        wm.flow_x_head.weight.normal_(0, 0.1)   # zero-init head would mask the route
    m.eval()
    B, T = 1, 3
    z = torch.randn(B, T, wm.n_spatial, wm.d_latent)
    a = _rand_actions(B, T)
    a2 = a.clone()
    a2[0, 1, 5] = (a2[0, 1, 5] + 1) % torch.tensor(NVEC[1:])  # perturb one cell
    sig = torch.zeros(B, T, dtype=torch.long)
    stp = torch.zeros(B, T, dtype=torch.long)
    with torch.no_grad():
        x1 = wm.denoise(z, a, None, sig, stp)
        x2 = wm.denoise(z, a2, None, sig, stp)
    assert not torch.allclose(x1, x2)
