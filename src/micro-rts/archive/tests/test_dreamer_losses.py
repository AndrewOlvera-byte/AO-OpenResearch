"""DreamerV4 loss tests — correctness of returns, shortcut-forcing semantics, and
gradient routing per optimizer.

Pure-tensor (no JVM). The key invariants:
- ``lambda_return`` matches a hand-computed value,
- the world-model loss trains only tokenizer + world model, with the reward /
  continue heads ARRIVE-aligned (slot t predicts the t-1 -> t transition),
- the shortcut-forcing loss trains only the world model (latents detached) and
  conditions on valid signal/step indices (the dyadic grid, upstream semantics),
- the actor/critic losses train only the action expert (world model held fixed),
- the percentile ``ReturnNormalizer`` tracks the return spread.
"""

import math

import torch

import models.dreamer as D
from loss.dreamer import (
    ReturnNormalizer, actor_critic_losses, cell_weights, dynamics_loss,
    lambda_return, mask_actions_to_sources, shortcut_forcing_loss,
    tokenizer_loss, world_model_loss,
)
from shared.dreamerv4 import TwoHot

OBS = (6, 8, 8)
NVEC = [64, 6, 4, 4, 4, 4, 7, 49]
MASK_W = 1 + sum(NVEC[1:])


def _model():
    cfg = D.DreamerV4Config.from_dict({
        "tokenizer": {"d_latent": 8, "enc_channels": 16},
        "dynamics": {"d_model": 32, "depth": 2, "n_heads": 4, "n_register": 2,
                     "k_max": 4, "action_emb": 8, "action_channels": 16},
        "actor_critic": {"dec_channels": 16, "imagine_horizon": 4,
                         "imagine_flow_steps": 2},
    })
    return D.DreamerV4(OBS, NVEC, cfg)


def _batch(B=2, T=4):
    return {
        "obs": torch.rand(B, T, *OBS),
        "action": torch.zeros(B, T, NVEC[0], 7, dtype=torch.long),
        "reward": torch.randn(B, T),
        "cont": torch.ones(B, T),
        "mask": torch.randint(0, 2, (B, T, NVEC[0], MASK_W)).float(),
        "is_first": torch.zeros(B, T, dtype=torch.bool),
    }


# --- lambda return ---------------------------------------------------------
def test_lambda_return_hand_computed():
    reward = torch.tensor([[0.0, 0.0]])
    value = torch.tensor([[1.0, 2.0, 3.0]])
    disc = torch.tensor([[1.0, 1.0]])       # gamma*cont = 1
    out = lambda_return(reward, value, disc, lam=0.5)
    # R1 = 0 + 1*3 = 3 ; R0 = 0 + 1*(0.5*2 + 0.5*3) = 2.5
    assert torch.allclose(out, torch.tensor([[2.5, 3.0]]), atol=1e-6)


def test_lambda_return_zero_discount_is_reward():
    reward = torch.randn(3, 5)
    value = torch.randn(3, 6)
    disc = torch.zeros(3, 5)
    assert torch.allclose(lambda_return(reward, value, disc), reward)


# --- world model loss --------------------------------------------------------
def test_world_model_loss_trains_only_world_params():
    m = _model()
    b = _batch()
    loss, metrics, z = world_model_loss(
        m, b["obs"], b["action"], b["reward"], b["cont"], b["mask"], b["is_first"])
    assert torch.isfinite(loss)
    for key in ("wm/recon", "wm/mask_bce", "wm/reward", "wm/continue"):
        assert key in metrics and metrics[key] == metrics[key]
    loss.backward()
    assert m.tokenizer.to_latent.weight.grad is not None
    assert m.world_model.reward_head.weight.grad is not None
    # The action expert is not part of the world-model graph.
    assert m.action_expert.action_conv.weight.grad is None
    assert m.action_expert.critic.out.weight.grad is None


def test_world_model_loss_reward_is_arrive_aligned():
    torch.manual_seed(0)
    m = _model()
    # Un-zero the reward head: with zero logits the two-hot CE is constant in the
    # target, which would make the "earlier reward matters" check vacuous.
    torch.nn.init.normal_(m.world_model.reward_head.weight, std=0.1)
    b = _batch()
    loss1, _, _ = world_model_loss(
        m, b["obs"], b["action"], b["reward"], b["cont"], b["mask"], b["is_first"])
    # The LAST step's reward/cont arrive after the window — they must be unused.
    r2, c2 = b["reward"].clone(), b["cont"].clone()
    r2[:, -1] += 100.0
    c2[:, -1] = 0.0
    loss2, _, _ = world_model_loss(
        m, b["obs"], b["action"], r2, c2, b["mask"], b["is_first"])
    assert torch.allclose(loss1, loss2, atol=1e-6)
    # An EARLIER reward must matter.
    r3 = b["reward"].clone()
    r3[:, 0] += 100.0
    loss3, _, _ = world_model_loss(
        m, b["obs"], b["action"], r3, b["cont"], b["mask"], b["is_first"])
    assert not torch.allclose(loss1, loss3, atol=1e-4)


def test_detach_latents_blocks_encoder_grads_from_scalar_heads():
    m = _model()
    assert m.cfg.detach_latents
    b = _batch()
    loss, _, _ = world_model_loss(
        m, b["obs"], b["action"], b["reward"], b["cont"], b["mask"], b["is_first"],
        recon_coef=0.0, mask_coef=0.0)     # only reward/continue remain
    loss.backward()
    # The zeroed recon/mask terms leave a zero (or absent) encoder grad; the point
    # is the scalar heads contribute nothing through the detached latents.
    g = m.tokenizer.to_latent.weight.grad
    assert g is None or torch.all(g == 0)                  # encoder untouched
    assert m.world_model.reward_head.weight.grad is not None


# --- shortcut forcing ---------------------------------------------------------
def test_shortcut_forcing_loss_finite_and_grads_flow():
    m = _model()
    b = _batch(B=4)
    z = m.tokenizer.encode(b["obs"])
    loss, metrics = shortcut_forcing_loss(m, z, b["action"], b["is_first"])
    assert torch.isfinite(loss)
    assert metrics["flow/matching"] >= 0
    assert "flow/consistency" in metrics
    loss.backward()
    wm = m.world_model
    assert wm.flow_x_head.weight.grad is not None
    assert wm.latent_in.weight.grad is not None
    assert wm.action_encoder.to_tokens.weight.grad is not None
    # Flow loss detaches latents -> encoder untouched.
    assert m.tokenizer.to_latent.weight.grad is None


def test_shortcut_forcing_uses_valid_dyadic_conditioning():
    m = _model()
    wm = m.world_model
    b = _batch(B=8, T=3)
    z = m.tokenizer.encode(b["obs"]).detach()
    calls = []
    orig = wm.denoise

    def spy(z_tilde, action, is_first, signal_idx, step_idx, **kw):
        calls.append((signal_idx.clone(), step_idx.clone()))
        return orig(z_tilde, action, is_first, signal_idx, step_idx, **kw)

    wm.denoise = spy
    try:
        torch.manual_seed(0)
        shortcut_forcing_loss(m, z, b["action"], b["is_first"], self_frac=0.5)
    finally:
        wm.denoise = orig
    assert len(calls) == 4  # empirical + two half steps (no-grad) + big step
    for sig, stp in calls:
        assert sig.min() >= 0 and sig.max() <= wm.k_max - 1   # trained signal bins only
        assert stp.min() >= 0 and stp.max() <= wm.emax
    # Empirical rows train at the finest step d = 1/k_max (step_idx = emax).
    assert torch.all(calls[0][1] == wm.emax)
    # Self-consistency: the half steps are one exponent finer than the big step.
    sig_h1, stp_h1 = calls[1]
    sig_big, stp_big = calls[3]
    assert torch.all(stp_h1 == stp_big + 1)
    assert torch.equal(sig_h1, sig_big)     # same start sigma, smaller step
    # Signal levels sit on the big step's dyadic grid: sigma is a multiple of d.
    d_idx = wm.k_max // (2 ** stp_big)      # k_max * d in index units
    assert torch.all(sig_big % d_idx == 0)


def _mrts_obs(B, T, H=4, W=4, occ_hw=(0, 0), change_t=1):
    """Synthetic 27-channel one-hot MicroRTS-shaped obs (groups 5/5/3/8/6) for
    :func:`cell_weights` tests: every cell is background (unit_type "none",
    channel 13) except ``occ_hw`` which holds a unit (some non-"none" unit type)
    from frame ``change_t`` onward (background before it)."""
    obs = torch.zeros(B, T, 27, H, W)
    obs[:, :, 0] = 1.0    # hp bucket 0
    obs[:, :, 5] = 1.0    # resources bucket 0
    obs[:, :, 10] = 1.0   # owner: none
    obs[:, :, 13] = 1.0   # unit_type: none (background)
    obs[:, :, 21] = 1.0   # action: none
    y, x = occ_hw
    obs[:, change_t:, 13, y, x] = 0.0
    obs[:, change_t:, 14, y, x] = 1.0   # some occupied unit type from change_t on
    return obs


def test_cell_weights_uniform_when_boosts_disabled():
    obs = _mrts_obs(2, 3)
    w = cell_weights(obs, occ_boost=1.0, changed_boost=1.0, floor=1.0, downsample=1)
    assert torch.allclose(w, torch.ones_like(w))


def test_cell_weights_upweights_occupied_and_changed_cells():
    B, T, H, W = 1, 3, 4, 4
    obs = _mrts_obs(B, T, H, W, occ_hw=(1, 2), change_t=1)
    w = cell_weights(obs, occ_boost=4.0, changed_boost=16.0, floor=1.0, downsample=1)
    w = w.reshape(B, T, H, W)
    occ_idx = 1 * W + 2
    # frame 0: cell is background everywhere -> uniform weight.
    assert torch.allclose(w[:, 0], w[:, 0, 0, 0].expand_as(w[:, 0]))
    # frame 1: the occupied+changed cell dominates (changed_boost > occ_boost),
    # and is strictly heavier than any background cell in the same frame.
    w1 = w[:, 1].reshape(B, -1)
    assert (w1[:, occ_idx] > w1[:, :occ_idx].max()).all()
    assert (w1[:, occ_idx] > w1[:, occ_idx + 1:].max()).all()
    # frame 2: still occupied but unchanged from frame 1 -> occ_boost, not
    # changed_boost, so lighter than frame 1's weight at the same cell.
    assert w[:, 2, 1, 2] < w[:, 1, 1, 2]
    assert w[:, 2, 1, 2] > w[:, 2, 0, 0]
    # per-(B,T) mean-normalized to 1.
    assert torch.allclose(w.reshape(B, T, -1).mean(dim=2), torch.ones(B, T), atol=1e-5)


def test_cell_weights_downsample_pools_to_latent_grid():
    # downsample=4 on an 8x8 raw grid -> 2x2 latent grid (matches
    # GridTokenizer.n_spatial = (h//4)*(w//4)); a unit anywhere in a 4x4 block
    # marks that whole pooled cell occupied.
    B, T, H, W = 1, 2, 8, 8
    obs = _mrts_obs(B, T, H, W, occ_hw=(3, 5), change_t=1)  # block (0,1) in 2x2 grid
    w = cell_weights(obs, occ_boost=4.0, changed_boost=16.0, floor=1.0, downsample=4)
    assert w.shape == (B, T, 4)
    w1 = w.reshape(B, T, 2, 2)[:, 1]
    assert w1[:, 0, 1] > w1[:, 0, 0]
    assert w1[:, 0, 1] > w1[:, 1, 0]
    assert w1[:, 0, 1] > w1[:, 1, 1]


def test_shortcut_forcing_loss_accepts_cell_weight_and_shifts_loss():
    m = _model()
    b = _batch(B=4, T=3)
    z = m.tokenizer.encode(b["obs"]).detach()
    torch.manual_seed(0)
    loss_uniform, _ = shortcut_forcing_loss(m, z, b["action"], b["is_first"])
    torch.manual_seed(0)
    cw = torch.ones(4, 3, z.shape[2])
    cw[:, :, 0] = 100.0  # heavily upweight one spatial cell
    cw = cw / cw.mean(dim=2, keepdim=True)
    loss_weighted, metrics = shortcut_forcing_loss(
        m, z, b["action"], b["is_first"], cell_weight=cw)
    assert torch.isfinite(loss_weighted)
    assert not torch.allclose(loss_uniform, loss_weighted)
    loss_weighted.backward()
    assert m.world_model.flow_x_head.weight.grad is not None


def test_dynamics_loss_accepts_cell_weight():
    m = _model()
    b = _batch(B=4, T=3)
    b["opponent_action"] = torch.zeros_like(b["action"])
    z = m.tokenizer.encode(b["obs"]).detach()
    # _model()'s OBS is (6,8,8) -> tokenizer n_spatial = (8//4)*(8//4) = 4;
    # match that grid so cell_weight aligns with z's spatial axis.
    cw = cell_weights(_mrts_obs(4, 3, 8, 8), occ_boost=4.0, changed_boost=16.0)
    loss, metrics = dynamics_loss(
        m, z, b["action"], b["reward"], b["cont"], b["is_first"],
        opponent_action=b["opponent_action"], cell_weight=cw)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.world_model.flow_x_head.weight.grad is not None


def _mrts_model():
    """Tiny model over the REAL 27-channel MicroRTS obs layout (8x8 board) so
    the group-CE path (which asserts the 27-channel grouping) is exercised."""
    cfg = D.DreamerV4Config.from_dict({
        "tokenizer": {"d_latent": 8, "enc_channels": 16},
        "dynamics": {"d_model": 32, "depth": 2, "n_heads": 4, "n_register": 2,
                     "k_max": 4, "action_emb": 8, "action_channels": 16},
        "actor_critic": {"dec_channels": 16, "imagine_horizon": 4,
                         "imagine_flow_steps": 2},
    })
    return D.DreamerV4((27, 8, 8), NVEC, cfg)


def test_mask_actions_to_sources_keeps_only_idle_own_units():
    B, T, H, W = 2, 3, 4, 4
    obs = _mrts_obs(B, T, H, W, occ_hw=(1, 2), change_t=0)   # own unit? plane 14 set...
    # _mrts_obs sets owner=none everywhere; make the occupied cell OWN and idle,
    # and add a second own unit that is BUSY (executing, action plane != none).
    obs[:, :, 10, 1, 2] = 0.0
    obs[:, :, 11, 1, 2] = 1.0            # own + idle (action-none stays 1)
    obs[:, :, 10, 3, 3] = 0.0
    obs[:, :, 11, 3, 3] = 1.0            # own but busy:
    obs[:, :, 21, 3, 3] = 0.0
    obs[:, :, 22, 3, 3] = 1.0            # executing some action
    action = torch.randint(1, 4, (B, T, H * W, 7))           # junk everywhere
    out = mask_actions_to_sources(action, obs)
    idle_own = 1 * W + 2
    busy_own = 3 * W + 3
    assert (out[:, :, idle_own] == action[:, :, idle_own]).all()   # kept
    assert (out[:, :, busy_own] == 0).all()                        # busy -> NOOP
    keep = torch.zeros(H * W, dtype=torch.bool)
    keep[idle_own] = True
    assert (out[:, :, ~keep] == 0).all()                           # junk -> NOOP
    assert out.dtype == action.dtype


def test_mask_actions_to_sources_is_noop_on_all_own_idle_board():
    obs = _mrts_obs(1, 2, 4, 4)
    obs[:, :, 10] = 0.0
    obs[:, :, 11] = 1.0                  # every cell own + idle
    action = torch.randint(0, 4, (1, 2, 16, 7))
    assert torch.equal(mask_actions_to_sources(action, obs), action)


def test_tokenizer_loss_v4_defaults_reproduce_v3():
    m = _mrts_model()
    obs = _mrts_obs(2, 3, 8, 8)
    mask = torch.randint(0, 2, (2, 3, 64, MASK_W)).float()
    torch.manual_seed(0)
    l_old, m_old, _ = tokenizer_loss(m, obs, mask)
    torch.manual_seed(0)
    l_new, m_new, _ = tokenizer_loss(m, obs, mask, group_ce_coef=0.0, cell_weight=None)
    assert torch.allclose(l_old, l_new)
    assert "tok/group_ce" not in m_new


def test_tokenizer_loss_group_ce_adds_term_and_grads():
    m = _mrts_model()
    obs = _mrts_obs(2, 3, 8, 8)
    mask = torch.randint(0, 2, (2, 3, 64, MASK_W)).float()
    loss, metrics, _ = tokenizer_loss(m, obs, mask, group_ce_coef=1.0)
    assert torch.isfinite(loss)
    assert metrics["tok/group_ce"] > 0
    assert loss.item() > metrics["tok/recon"] + metrics["tok/mask_bce"] - 1e-6
    loss.backward()
    assert m.tokenizer.to_latent.weight.grad is not None


def test_tokenizer_loss_cell_weight_shifts_recon():
    m = _mrts_model()
    obs = _mrts_obs(2, 3, 8, 8, occ_hw=(1, 2), change_t=1)
    mask = torch.randint(0, 2, (2, 3, 64, MASK_W)).float()
    cw = cell_weights(obs, occ_boost=4.0, changed_boost=16.0, downsample=1)
    assert cw.shape == (2, 3, 64)
    torch.manual_seed(0)
    l_uniform, mu, _ = tokenizer_loss(m, obs, mask)
    torch.manual_seed(0)
    l_weighted, mw, _ = tokenizer_loss(m, obs, mask, cell_weight=cw)
    assert torch.isfinite(l_weighted)
    assert mu["tok/recon"] != mw["tok/recon"]
    # mask BCE untouched by the recon cell weighting.
    assert abs(mu["tok/mask_bce"] - mw["tok/mask_bce"]) < 1e-6


# --- masked scalar heads -------------------------------------------------------
def test_twohot_mask_selects_elements():
    coder = TwoHot(31)
    logits = torch.randn(2, 4, 31)
    target = torch.randn(2, 4)
    ones = torch.ones(2, 4, dtype=torch.bool)
    assert torch.allclose(coder.loss(logits, target, mask=ones),
                          coder.loss(logits, target), atol=1e-6)
    # Zeroing one element removes exactly its contribution.
    m = ones.clone()
    m[0, 0] = False
    tweaked = target.clone()
    tweaked[0, 0] += 100.0
    assert torch.allclose(coder.loss(logits, target, mask=m),
                          coder.loss(logits, tweaked, mask=m), atol=1e-6)


def test_dynamics_loss_masks_episode_boundary_targets():
    torch.manual_seed(0)
    m = _model()
    torch.nn.init.normal_(m.world_model.reward_head.weight, std=0.1)
    b = _batch(B=2, T=4)
    b["is_first"][:, 2] = True          # slot 2 starts a new episode

    def run(reward, cont):
        torch.manual_seed(123)          # identical flow-noise draws per call
        loss, _ = dynamics_loss(m, m.tokenizer.encode(b["obs"]).detach(),
                                b["action"], reward, cont, b["is_first"],
                                self_frac=0.0)
        return loss

    loss1 = run(b["reward"], b["cont"])
    # reward_1 / cont_1 target slot 2 == the masked episode-start slot: unused.
    r2, c2 = b["reward"].clone(), b["cont"].clone()
    r2[:, 1] += 100.0
    c2[:, 1] = 0.0
    assert torch.allclose(loss1, run(r2, c2), atol=1e-6)
    # reward_0 targets slot 1 (valid): it must matter.
    r3 = b["reward"].clone()
    r3[:, 0] += 100.0
    assert not torch.allclose(loss1, run(r3, b["cont"]), atol=1e-4)


def test_dynamics_loss_routes_grads_and_reports_metrics():
    m = _model()
    b = _batch(B=4)
    z = m.tokenizer.encode(b["obs"]).detach()
    loss, metrics = dynamics_loss(m, z, b["action"], b["reward"], b["cont"],
                                  b["is_first"])
    assert torch.isfinite(loss)
    for key in ("flow/mse", "flow/matching", "flow/consistency",
                "wm/reward", "wm/continue", "wm/total"):
        assert key in metrics
    loss.backward()
    wm = m.world_model
    # Heads read the forcing pass's registers: grads reach heads AND denoiser.
    assert wm.reward_head.weight.grad is not None
    assert wm.continue_head.weight.grad is not None
    assert wm.flow_x_head.weight.grad is not None
    assert m.tokenizer.to_latent.weight.grad is None      # frozen-phase contract


# --- latent normalization --------------------------------------------------------
def test_shortcut_forcing_is_invariant_to_matched_latent_scale():
    m = _model()
    wm = m.world_model
    b = _batch(B=4)
    z = m.tokenizer.encode(b["obs"]).detach()
    torch.manual_seed(7)
    loss1, _ = shortcut_forcing_loss(m, z, b["action"], b["is_first"])
    # Scaling the latents AND the normalization scale together changes nothing:
    # the denoiser only ever sees unit-RMS space.
    wm.set_latent_scale(3.0)
    torch.manual_seed(7)
    loss2, _ = shortcut_forcing_loss(m, 3.0 * z, b["action"], b["is_first"])
    assert torch.allclose(loss1, loss2, atol=1e-5)


def test_sample_next_round_trips_latent_scale():
    m = _model()
    wm = m.world_model
    z_ctx = torch.rand(2, 3, wm.n_spatial, wm.d_latent)
    act = torch.zeros(2, 3, NVEC[0], 7, dtype=torch.long)
    torch.manual_seed(11)
    out1 = wm.sample_next(z_ctx, act, None, steps=2)
    wm.set_latent_scale(5.0)
    torch.manual_seed(11)
    out2 = wm.sample_next(5.0 * z_ctx, act, None, steps=2)
    assert torch.allclose(out2, 5.0 * out1, atol=1e-4)


# --- tokenizer latent-noise augmentation ------------------------------------------
def test_tokenizer_loss_latent_noise_perturbs_decode_only():
    torch.manual_seed(0)
    m = _model()
    b = _batch()
    torch.manual_seed(1)
    loss_clean, _, z_clean = tokenizer_loss(m, b["obs"], b["mask"])
    torch.manual_seed(1)
    loss_noisy, _, z_noisy = tokenizer_loss(m, b["obs"], b["mask"], latent_noise=0.5)
    assert torch.allclose(z_clean, z_noisy)          # encoder output unchanged
    assert not torch.allclose(loss_clean, loss_noisy)
    loss_noisy.backward()
    assert m.tokenizer.to_latent.weight.grad is not None


# --- actor / critic -----------------------------------------------------------
def test_actor_critic_losses_route_grads_and_spare_world_model():
    m = _model()
    z0 = torch.rand(6, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    imagined = m.imagine(z0, horizon=4)
    actor_loss, critic_loss, metrics = actor_critic_losses(m, imagined)
    assert torch.isfinite(actor_loss) and torch.isfinite(critic_loss)

    actor_loss.backward(retain_graph=True)
    critic_loss.backward()
    # Actor grads on the action-expert decoder / action head.
    assert m.action_expert.action_conv.weight.grad is not None
    # Critic grads on the value MLP.
    assert m.action_expert.critic.out.weight.grad is not None
    # World model held fixed: imagined latents were detached.
    assert m.world_model.flow_x_head.weight.grad is None
    assert m.tokenizer.to_latent.weight.grad is None
    # Target critic never receives gradient.
    assert all(p.grad is None for p in m.action_expert.target_critic.parameters())


def test_return_normalizer_tracks_percentile_spread():
    norm = ReturnNormalizer(rate=1.0, limit=1.0, low=5.0, high=95.0)
    returns = torch.linspace(0.0, 100.0, steps=1001).view(1, -1)
    scale = norm(returns)
    assert abs(scale - 90.0) < 1.0          # P95 - P5 of U[0,100]
    # Tiny spreads are floored at `limit` so noise is not amplified.
    norm2 = ReturnNormalizer(rate=1.0, limit=1.0)
    assert norm2(torch.zeros(4, 4)) == 1.0
    # EMA moves gradually at small rates.
    norm3 = ReturnNormalizer(rate=0.5, limit=1.0)
    norm3(returns)
    s2 = norm3(torch.zeros(4, 4) + 50.0)    # spread 0
    assert 1.0 < s2 < 90.0


def test_actor_loss_uses_normalized_advantage():
    torch.manual_seed(0)
    m = _model()
    z0 = torch.rand(4, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    imagined = m.imagine(z0, horizon=3)
    imagined["reward"] = imagined["reward"] + 100.0  # inflate the return spread... scale
    norm = ReturnNormalizer(rate=1.0, limit=1.0)
    _, _, metrics = actor_critic_losses(m, imagined, return_normalizer=norm)
    assert metrics["ac/return_scale"] >= 1.0


def test_target_critic_ema_moves_toward_critic():
    m = _model()
    before = m.action_expert.target_critic.out.weight.detach().clone()
    # Perturb the online critic, then EMA the target toward it.
    with torch.no_grad():
        m.action_expert.critic.out.weight.add_(1.0)
    m.action_expert.update_target(decay=0.5)
    after = m.action_expert.target_critic.out.weight.detach()
    assert torch.all(after > before)  # moved toward the (increased) online critic
