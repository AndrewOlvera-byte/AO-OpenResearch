"""Dreamer 4 loss functions — one per optimized subsystem.

Three optimization problems, matching the three optimizers in
``DreamerV4.build_optimizers``:

- **world model** (``world_model_loss`` + ``shortcut_forcing_loss``) — the
  tokenizer reconstruction and predicted action-mask BCE; the reward (symlog
  two-hot) and continue (BCE) heads, arrive-aligned (slot ``t`` predicts the
  ``t-1 -> t`` transition, so slot 0 is skipped and episode-start slots are
  masked); and the Dreamer 4 shortcut forcing objective on the transformer
  denoiser, computed in the world model's unit-RMS normalized latent space.
  Latents entering the world model are stop-gradient'd by default
  (``detach_latents``) so the encoder is shaped by reconstruction, as in
  Dreamer 4's separate tokenizer phase. In the pretraining ``dynamics_loss`` the
  scalar heads read the empirical forcing pass's registers (no clean pass).
- **generative dynamics** (``shortcut_forcing_loss``) — verified against upstream
  (nicklashansen/dreamer4 ``train_dynamics.py``): x-prediction with diffusion
  forcing. Every frame is noised with its own signal level ``sigma`` on the dyadic
  grid; *empirical* rows train at the finest step (``d = 1/k_max``) with the
  ``0.9*sigma + 0.1`` weighting; *self-consistency* rows train a bigger shortcut
  step against two half-steps (velocity space, ``(1-sigma)^2`` weight, no-grad
  target) so few-step sampling stays consistent with many-step sampling.
- **actor / critic** (``actor_critic_losses``) — imagined-rollout policy
  improvement: the critic two-hot-regresses the lambda-return; the actor follows
  REINFORCE on the (returns - value) advantage, scaled by the DreamerV3
  percentile ``ReturnNormalizer``, with an entropy bonus.

``lambda_return`` is factored out and unit-tested against a hand-computed case.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _valid_transitions(is_first, n_rows=None):
    """Arrive-aligned target validity: slot ``t+1`` predicting the ``t -> t+1``
    transition is invalid when frame ``t+1`` starts a new episode (the target
    reward/continue belong to the previous episode, whose terminal arrival frame
    was never stored). Returns a (B,T-1) bool mask or None."""
    if is_first is None:
        return None
    v = ~is_first[:, 1:].bool()
    return v[:n_rows] if n_rows is not None else v


def _masked_bce(logits, target, valid):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if valid is None:
        return loss.mean()
    v = valid.to(loss.dtype)
    return (loss * v).sum() / v.sum().clamp_min(1.0)


# unit_type group is channels [13:21) of the 27-channel obs; channel 13 is the
# "none" (empty-cell) one-hot class (see eval_dreamer_tokenizer.py:NONE_UNIT_TYPE_CH).
_NONE_UNIT_TYPE_CH = 13
# The 27 MicroRTS obs channels as one-hot groups: hp(5) resources(5) owner(3)
# unit_type(8) current_action(6) — each group is exactly one-hot per cell.
_OBS_GROUPS = ((0, 5), (5, 10), (10, 13), (13, 21), (21, 27))
_OWNER_OWN_CH = 11          # owner group [10:13): none / own / enemy
_OWNER_ENEMY_CH = 12
_ACTION_NONE_CH = 21        # current-action group [21:27): channel 21 = idle


def mask_actions_to_sources(action, obs):
    """Zero (NOOP) the per-cell action at every cell that cannot issue one.

    GridNet policies emit an action for ALL H*W cells; the engine executes only
    the ones at legal source cells (~0.4% of cells), and the collector stores
    the raw policy output — so ~97% of nonzero stored action entries sit on
    cells with no controllable unit and were never executed. Measured on the v3
    store: P(cell changes next frame | stored action != noop) has a 1.3x lift
    over other cells (i.e. the channel is noise as stored), but a 70x lift once
    restricted to legal-source cells. Feeding the raw tensor makes ignoring the
    action channel the statistically correct behavior for the world model —
    the root cause of the flat CF-PROBE self_gap through v3/v4/v4.1.

    A cell can issue an action iff it holds one of the acting player's units
    AND that unit is idle (current-action plane = none) — both readable from
    the obs the batch already carries (own-unit plane ∧ action-none plane
    ≈ the engine's source-legality mask, which the dynamics task's loader
    doesn't load). Returns a new tensor; all 7 components zeroed (type 0 =
    NOOP) at non-source cells.

    Apply to the SELF action only: the opponent channel comes from the jar
    patch reading engine-executed actions and is already clean (which is why
    opp_gap outran self_gap in every run so far).

    ``action`` (B,[T],H*W,7) long; ``obs`` (B,[T],C,H,W).
    """
    src = (obs[..., _OWNER_OWN_CH, :, :] > 0.5) & \
          (obs[..., _ACTION_NONE_CH, :, :] > 0.5)            # (B,[T],H,W)
    return action * src.flatten(-2).unsqueeze(-1).to(action.dtype)


def opponent_bc_loss(wm, feats, opponent_action, obs, is_first=None, *,
                     slot_weight=None):
    """Opponent-policy BC (v4.4): per-cell CE of the head's logits against the
    engine-executed opponent action AT each frame.

    ``feats`` (B,T,n_spatial,d_model) trunk spatial outputs whose slot ``t`` saw
    frame ``t`` (labels are NOT shifted: predicting frame ``t``'s own action
    from the state at ``t`` is a policy, unlike the shifted conditioning
    inputs). ``opponent_action`` (B,T,H*W,n_comp) long; ``obs`` (B,T,C,H,W)
    raw frames (27-channel MicroRTS layout — the opponent's legal-source cells
    are read off the enemy-owner ∧ idle planes, the opponent-side twin of
    :func:`mask_actions_to_sources`).

    CE runs ONLY at opponent source cells (~1% of the board — everywhere else
    the label is trivially NOOP and would drown the signal, the same imbalance
    the cell-weighted flow loss fixes), with conditional components (move dir,
    attack target, ...) scored only where the true type selects them.
    Slots whose label belongs to a different frame are masked: a
    terminal-arrival splice at ``t`` keeps the NEXT episode's stored action,
    identified by ``is_first[t+1]`` (see mrts_dataset terminal substitution).
    ``slot_weight`` (B,T) optionally downweights slots (the caller passes the
    forcing pass's ``0.9*sigma + 0.1`` so heavily-noised inputs, whose trunk
    barely saw the frame, don't dominate).

    The privileged labels are only needed HERE — at deployment/imagination the
    head's own samples stand in for the opponent (fog-of-war ready: LIAM-style
    training-time-only opponent supervision). Returns ``(loss, metrics)``.
    """
    assert wm.opp_head is not None, "opponent_bc_loss needs a model built with opp_head"
    C = obs.shape[-3]
    assert C == _OBS_GROUPS[-1][1], \
        f"opponent source cells assume the {_OBS_GROUPS[-1][1]}-channel MicroRTS layout, got C={C}"
    B, T = opponent_action.shape[:2]
    logits = wm.opp_head(feats)                                   # (B,T,H*W,total)

    src = (obs[..., _OWNER_ENEMY_CH, :, :] > 0.5) & \
          (obs[..., _ACTION_NONE_CH, :, :] > 0.5)                 # (B,T,H,W)
    valid = torch.ones(B, T, dtype=torch.bool, device=obs.device)
    if is_first is not None and T > 1:
        valid[:, :-1] &= ~is_first[:, 1:].bool()                  # terminal splices
    w_cell = (src.flatten(-2) & valid[..., None]).float()         # (B,T,H*W)
    if slot_weight is not None:
        w_cell = w_cell * slot_weight.to(w_cell.dtype)[..., None]

    atype = opponent_action[..., 0]
    comp_sizes = wm.opp_head.comp_sizes
    # comp -> action TYPE that reads it (0=type itself, always live); mirrors
    # models.dreamer.world_model._COMPONENT_ACTIVE_TYPE.
    active_types = (None, 1, 2, 3, 4, 4, 5)
    loss_sum = logits.new_zeros(())
    w_sum = logits.new_zeros(())
    off = 0
    for i, n in enumerate(comp_sizes):
        ce = F.cross_entropy(
            logits[..., off:off + n].reshape(-1, n).float(),
            opponent_action[..., i].reshape(-1).long(),
            reduction="none").reshape(B, T, -1)
        w = w_cell if active_types[i] is None else \
            w_cell * (atype == active_types[i]).float()
        loss_sum = loss_sum + (ce * w).sum()
        w_sum = w_sum + w.sum()
        off += n
    loss = loss_sum / w_sum.clamp_min(1.0)

    with torch.no_grad():
        at_src = w_cell > 0
        n_src = at_src.sum().clamp_min(1)
        type_pred = logits[..., :comp_sizes[0]].argmax(dim=-1)
        metrics = {
            "opp_bc/loss": float(loss.detach()),
            "opp_bc/type_acc": float((type_pred == atype)[at_src].float().sum() / n_src),
            "opp_bc/source_frac": float(src.float().mean()),
        }
    return loss, metrics


def cell_weights(obs, *, occ_boost=4.0, changed_boost=16.0, floor=1.0, downsample=4):
    """Per-spatial-cell flow-loss weight (B,T,n_spatial) from raw obs frames.

    MicroRTS boards are ~99% empty background cells whose one-hot state a bot
    or player action almost never touches; uniformly averaging the flow-matching
    MSE over all cells dilutes the action-caused signal (the occupied, changing
    cells) 100:1 into the static background, so the model can hit a very low
    aggregate loss while never learning to condition on the action (see
    NEXT_PLAN.md dynamics-v4 diagnosis: CF-PROBE self_gap stays ~0 the whole v3
    run). This mirrors the standard fix for extreme foreground/background
    imbalance in dense prediction (focal-loss-style reweighting, occupancy-only
    losses in voxel/video world models): background cells get weight ``floor``,
    occupied cells (unit present, any owner) get ``occ_boost``, and cells whose
    one-hot state differs from the previous frame (the actual transition the
    action caused) get ``changed_boost`` — the three are maxed, not summed, so a
    changed-and-occupied cell isn't double-counted.

    ``downsample`` must match ``TokenizerConfig.downsample`` (the H,W -> H/d,W/d
    reduction the tokenizer's latent grid uses, GridTokenizer.n_spatial): a raw
    cell is occupied/changed if ANY cell in its downsample x downsample block is,
    max-pooled to the latent grid's row-major flatten order so the result lines
    up 1:1 with ``z``'s ``n_spatial`` axis. Per-(B,T) mean-normalized to 1 so the
    overall loss scale (and therefore ``flow_coef``) is unaffected — only the
    intra-frame *distribution* of gradient shifts toward the cells actions
    actually move.
    """
    B, T, C, H, W = obs.shape
    occ = obs[:, :, _NONE_UNIT_TYPE_CH] < 0.5                       # (B,T,H,W)
    changed = torch.zeros(B, T, H, W, dtype=torch.bool, device=obs.device)
    changed[:, 1:] = (obs[:, 1:] != obs[:, :-1]).any(dim=2)

    def pool(x):
        if downsample <= 1:
            return x
        assert H % downsample == 0 and W % downsample == 0
        x = x.reshape(B, T, H // downsample, downsample, W // downsample, downsample)
        return x.amax(dim=(3, 5))

    occ, changed = pool(occ), pool(changed)
    Hp, Wp = occ.shape[-2:]
    w = torch.full((B, T, Hp, Wp), float(floor), device=obs.device, dtype=torch.float32)
    w = torch.where(occ, torch.maximum(w, torch.tensor(float(occ_boost), device=obs.device)), w)
    w = torch.where(changed, torch.maximum(w, torch.tensor(float(changed_boost), device=obs.device)), w)
    w = w.flatten(2)                                                # (B,T,n_spatial)
    return w / w.mean(dim=2, keepdim=True).clamp_min(1e-8)


# --- world model (clean pass) ---------------------------------------------
def _opp_dropout_mask(model, opponent_action, B, T, device, opponent_valid=None):
    """Train-time opponent dropout (B,T) or None: with prob ``opp_dropout`` a
    frame's opponent channel becomes the learned ``unknown_opp`` embedding, so
    one model trains both the conditional and the marginal dynamics."""
    p = float(getattr(model.cfg.dynamics, "opp_dropout", 0.0))
    if opponent_action is None:
        return None
    if opponent_valid is None and (not model.training or p <= 0.0):
        return None
    unknown = (~opponent_valid.bool().to(device) if opponent_valid is not None else
               torch.zeros(B, T, dtype=torch.bool, device=device))
    if model.training and p > 0.0:
        unknown |= torch.rand(B, T, device=device) < p
    return unknown


def world_model_loss(model, obs, action, reward, cont, mask, is_first=None, *,
                     opponent_action=None,
                     recon_coef=1.0, mask_coef=1.0, reward_coef=1.0, cont_coef=1.0):
    """Tokenizer + scalar-head losses over a batch of sequences.

    Shapes: ``obs`` (B,T,C,H,W); ``action`` (B,T,H*W,7) long, the action chosen AT
    each frame; ``opponent_action`` same shape (None = opponent unknown);
    ``reward``/``cont`` (B,T) for the transition taken at each frame;
    ``mask`` (B,T,H*W,mask_width); ``is_first`` (B,T) bool. Returns
    ``(loss, metrics, z)`` where ``z`` are the encoded latents (with grad; the
    caller detaches for imagination seeding).
    """
    tok = model.tokenizer
    z = tok.encode(obs)                                     # (B,T,n_spatial,d_latent)

    recon = tok.decode(z)
    recon_loss = F.mse_loss(recon, obs)

    mask_logits = tok.decode_mask(z)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask.float())

    z_wm = z.detach() if model.cfg.detach_latents else z
    opp_unknown = _opp_dropout_mask(model, opponent_action, *z.shape[:2], z.device)
    ctx = model.world_model.contextualize(z_wm, action, is_first,
                                          opponent_action=opponent_action,
                                          opp_unknown=opp_unknown)
    # Arrive alignment: the readout at slot t+1 (which has seen action_t via the
    # shift) predicts reward_t / cont_t; slot 0 has no transition target, and
    # slots that start a new episode have no valid target either (masked).
    valid = _valid_transitions(is_first)
    reward_loss = model.world_model.reward_coder.loss(
        ctx["reward_logits"][:, 1:], reward[:, :-1], mask=valid)
    cont_loss = _masked_bce(ctx["continue_logit"][:, 1:], cont[:, :-1].float(), valid)

    loss = (recon_coef * recon_loss + mask_coef * mask_loss
            + reward_coef * reward_loss + cont_coef * cont_loss)
    metrics = {
        "wm/recon": float(recon_loss.detach()),
        "wm/mask_bce": float(mask_loss.detach()),
        "wm/reward": float(reward_loss.detach()),
        "wm/continue": float(cont_loss.detach()),
        "wm/total": float(loss.detach()),
    }
    return loss, metrics, z


# --- pretraining phases (frozen-tokenizer staging) ------------------------
def tokenizer_loss(model, obs, mask, *, recon_coef=1.0, mask_coef=1.0,
                   latent_noise=0.0, group_ce_coef=0.0, cell_weight=None):
    """Tokenizer-only phase (Dreamer 4 trains the autoencoder as its own stage).

    The reconstruction MSE + predicted action-mask BCE slice of
    :func:`world_model_loss`, without touching the world model — the objective for
    ``train_dreamer_tokenizer``. ``obs`` (B,[T],C,H,W); ``mask`` (B,[T],H*W,mask_width).
    Returns ``(loss, metrics, z)`` with the encoded latents ``z`` (with grad).

    ``latent_noise`` adds Gaussian noise (absolute std) to ``z`` before decoding:
    the decoder/mask head become robust to the imperfect latents the flow model
    will hand them at rollout, and the encoder must keep its latents spread out
    relative to the noise floor — anchoring the latent scale that reconstruction
    MSE alone leaves free to collapse.

    v4 objective sharpeners (both default-off == exact v3 behavior):

    ``group_ce_coef`` adds per-group categorical cross-entropy: each of the 27
    MicroRTS channels belongs to a one-hot group (hp/resources/owner/unit_type/
    action, ``_OBS_GROUPS``), so the decoder output within a group is a
    categorical distribution and CE is its proper likelihood — it puts real
    gradient on rare classes (actual units) that a plain per-channel MSE
    underweights, the same reason discrete-token world models (IRIS et al.)
    train with CE. Applied only when the obs channel count matches the MicroRTS
    layout; the MSE term is kept (it anchors the tanh-latent scale together
    with ``latent_noise``).

    ``cell_weight`` (B,[T],H*W) — typically :func:`cell_weights` at
    ``downsample=1`` (raw grid, NOT the latent grid) — reweights both recon
    terms toward occupied/changed cells so encoder capacity goes to the
    dynamic content the downstream dynamics model must predict, not the ~99%
    static background (Delta-IRIS trains its autoencoder on inter-frame
    deltas for the same reason).
    """
    tok = model.tokenizer
    z = tok.encode(obs)                                     # (B,[T],n_spatial,d_latent)
    z_dec = z + latent_noise * torch.randn_like(z) if latent_noise > 0 else z
    recon = tok.decode(z_dec)

    seq = obs.dim() == 5
    cw = None
    if cell_weight is not None:
        H, W = obs.shape[-2:]
        cw = cell_weight.reshape(*obs.shape[:-3], H, W)      # (B,[T],H,W)

    if cw is None:
        recon_loss = F.mse_loss(recon, obs)
    else:
        per_cell = (recon - obs).pow(2).mean(dim=-3)         # (B,[T],H,W)
        recon_loss = (per_cell * cw).mean()

    ce_loss = None
    if group_ce_coef > 0.0:
        C = obs.shape[-3]
        assert C == _OBS_GROUPS[-1][1], \
            f"group CE assumes the {_OBS_GROUPS[-1][1]}-channel MicroRTS layout, got C={C}"
        ce_terms = []
        for lo, hi in _OBS_GROUPS:
            logits = recon[..., lo:hi, :, :]
            target = obs[..., lo:hi, :, :].argmax(dim=-3)     # (B,[T],H,W)
            ce = F.cross_entropy(
                logits.flatten(0, 1).contiguous() if seq else logits,
                target.flatten(0, 1) if seq else target,
                reduction="none")                             # (N,H,W)
            if seq:
                ce = ce.reshape(*obs.shape[:2], *ce.shape[1:])
            ce_terms.append((ce * cw).mean() if cw is not None else ce.mean())
        ce_loss = torch.stack(ce_terms).mean()

    mask_logits = tok.decode_mask(z_dec)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask.float())
    loss = recon_coef * recon_loss + mask_coef * mask_loss
    if ce_loss is not None:
        loss = loss + group_ce_coef * ce_loss
    metrics = {
        "tok/recon": float(recon_loss.detach()),
        "tok/mask_bce": float(mask_loss.detach()),
        "tok/total": float(loss.detach()),
    }
    if ce_loss is not None:
        metrics["tok/group_ce"] = float(ce_loss.detach())
    return loss, metrics, z


def dynamics_loss(model, z, action, reward, cont, is_first=None, *,
                  opponent_action=None, opponent_valid=None, cell_weight=None, obs=None,
                  reward_coef=1.0, cont_coef=1.0, flow_coef=1.0,
                  opp_bc_coef=0.0, self_frac=0.25):
    """World-model phase over a **frozen** tokenizer's latents.

    The world-model slice of the DreamerV4 objective without the tokenizer terms:
    the Dreamer 4 shortcut-forcing flow objective on the transformer denoiser, plus
    the arrive-aligned reward (symlog two-hot) and continue (BCE) heads read from
    the **empirical forcing pass's** register tokens (upstream: no separate clean
    pass, saving a full transformer forward per step). Slot ``t`` predicts the
    ``t-1 -> t`` transition, so slot 0 is skipped and episode-start slots are
    masked. ``z`` are raw tokenizer latents (B,T,n_spatial,d_latent); grads flow
    only into the world model (the caller keeps the tokenizer frozen).
    ``cell_weight`` (B,T,n_spatial), typically from :func:`cell_weights` on the
    same batch's raw obs — see :func:`shortcut_forcing_loss`.

    ``opp_bc_coef`` > 0 (model built with ``dynamics.opp_head``, ``obs`` and
    ``opponent_action`` given) adds :func:`opponent_bc_loss` on the empirical
    forcing pass's spatial trunk outputs, sigma-weighted like the flow rows.
    Returns ``(loss, metrics)``.
    """
    wm = model.world_model
    want_bc = (opp_bc_coef > 0.0 and getattr(wm, "opp_head", None) is not None
               and opponent_action is not None and obs is not None)
    floss, fm, extras = shortcut_forcing_loss(
        model, z, action, is_first, opponent_action=opponent_action,
        opponent_valid=opponent_valid,
        cell_weight=cell_weight, self_frac=self_frac,
        return_extras=True, return_spatial=want_bc)
    reg_emp, n_emp = extras["registers"], extras["n_emp"]

    reward_logits = wm.reward_head(reg_emp)                  # (n_emp,T,bins)
    cont_logit = wm.continue_head(reg_emp).squeeze(-1)       # (n_emp,T)
    valid = _valid_transitions(is_first, n_rows=n_emp)
    reward_loss = wm.reward_coder.loss(
        reward_logits[:, 1:], reward[:n_emp, :-1], mask=valid)
    cont_loss = _masked_bce(cont_logit[:, 1:], cont[:n_emp, :-1].float(), valid)

    loss = reward_coef * reward_loss + cont_coef * cont_loss + flow_coef * floss
    metrics = {
        "wm/reward": float(reward_loss.detach()),
        "wm/continue": float(cont_loss.detach()),
        **fm,
    }
    if want_bc:
        bc_loss, bc_m = opponent_bc_loss(
            wm, extras["spatial"], opponent_action[:n_emp], obs[:n_emp],
            is_first[:n_emp] if is_first is not None else None,
            slot_weight=(0.9 * extras["sigma"] + 0.1) *
                        (opponent_valid[:n_emp].float() if opponent_valid is not None else 1.0))
        loss = loss + opp_bc_coef * bc_loss
        metrics.update(bc_m)
    metrics["wm/total"] = float(loss.detach())
    return loss, metrics


# --- generative dynamics (shortcut forcing, upstream-faithful) -------------
def shortcut_forcing_loss(model, z, action, is_first=None, *, opponent_action=None,
                          opponent_valid=None, cell_weight=None, self_frac=0.25, return_registers=False,
                          return_extras=False, return_spatial=False):
    """Dreamer 4 shortcut forcing on the transformer denoiser.

    ``z`` are RAW tokenizer latents (B,T,n_spatial,d_latent) — detached and
    normalized to the world model's unit-RMS space here, so the unit-variance
    noise and the clean targets live on the same scale. The world model
    x-predicts the (normalized) clean latents of every frame from per-frame
    noised inputs (diffusion forcing). The first ``B - round(B*self_frac)`` rows
    train the empirical flow at the finest step; the rest train dyadic
    self-consistency.

    ``cell_weight`` (B,T,n_spatial), or None for the plain uniform-over-cells
    average: reweights the per-cell squared error before the spatial mean (see
    :func:`cell_weights` — background/occupied/changed cells get different
    weight so the ~1%-of-cells action signal isn't diluted by the empty board).
    Mean-normalized to 1 over the spatial axis, so passing ``None`` and passing
    an all-ones tensor are equivalent.

    With ``return_registers`` the return is ``(loss, metrics, reg_emp, n_emp)``
    where ``reg_emp`` (n_emp,T,d_model) are the empirical pass's pooled register
    outputs — :func:`dynamics_loss` reads the reward/continue heads off them.
    With ``return_extras`` (supersedes ``return_registers``) the return is
    ``(loss, metrics, extras)`` with ``extras = {"registers", "spatial",
    "sigma", "n_emp"}`` — ``spatial`` (n_emp,T,n_spatial,d_model) is the
    empirical pass's trunk output for the opponent BC head (None unless
    ``return_spatial``), ``sigma`` its (n_emp,T) per-slot signal levels.
    """
    wm = model.world_model
    k_max, emax = wm.k_max, wm.emax
    z = wm.normalize(z.detach())
    B, T = z.shape[:2]
    device = z.device
    noise = torch.randn_like(z)
    opp_unknown = _opp_dropout_mask(model, opponent_action, B, T, device,
                                    opponent_valid=opponent_valid)

    n_self = int(round(B * self_frac)) if emax > 0 else 0
    n_self = min(n_self, B - 1) if B > 1 else 0
    n_emp = B - n_self

    def interp(zc, nz, sigma):
        return (1.0 - sigma)[..., None, None] * nz + sigma[..., None, None] * zc

    # Empirical rows: finest step d = 1/k_max, sigma = j/k_max, j ~ U{0..k_max-1};
    # weight w = 0.9*sigma + 0.1 (upstream).
    ze, ne = z[:n_emp], noise[:n_emp]
    ae = action[:n_emp]
    oe = opponent_action[:n_emp] if opponent_action is not None else None
    ue = opp_unknown[:n_emp] if opp_unknown is not None else None
    fe = is_first[:n_emp] if is_first is not None else None
    cwe = cell_weight[:n_emp] if cell_weight is not None else None
    j = torch.randint(0, k_max, (n_emp, T), device=device)
    sigma_e = j.float() / k_max
    step_e = torch.full((n_emp, T), emax, dtype=torch.long, device=device)
    want_reg = return_registers or return_extras
    want_spatial = return_spatial and return_extras
    out = wm.denoise(interp(ze, ne, sigma_e), ae, fe, j, step_e,
                     return_registers=want_reg, return_spatial=want_spatial,
                     opponent_action=oe, opp_unknown=ue)
    if want_spatial:
        x1_hat, reg_emp, spat_emp = out
    elif want_reg:
        x1_hat, reg_emp = out
        spat_emp = None
    else:
        x1_hat, reg_emp, spat_emp = out, None, None
    per_cell = (x1_hat - ze).pow(2).mean(dim=3)              # (n_emp, T, n_spatial)
    per = (per_cell * cwe).mean(dim=2) if cwe is not None else per_cell.mean(dim=2)
    w = 0.9 * sigma_e + 0.1
    emp_loss = (w * per).mean()

    metrics = {"flow/matching": float(emp_loss.detach())}
    with torch.no_grad():
        # Unweighted x-prediction MSE, overall and by signal-level third — the
        # weighted `flow/matching` is dominated by the (easy) high-noise regime.
        mse = per.detach()
        metrics["flow/mse"] = float(mse.mean())
        for name, sel in (("lo", sigma_e < 1 / 3),
                          ("mid", (sigma_e >= 1 / 3) & (sigma_e < 2 / 3)),
                          ("hi", sigma_e >= 2 / 3)):
            metrics[f"flow/mse_sigma_{name}"] = \
                float(mse[sel].mean()) if sel.any() else 0.0

    def _ret(loss, metrics):
        if return_extras:
            return loss, metrics, {"registers": reg_emp, "spatial": spat_emp,
                                   "sigma": sigma_e, "n_emp": n_emp}
        if return_registers:
            return loss, metrics, reg_emp, n_emp
        return loss, metrics

    if n_self == 0:
        metrics["flow/consistency"] = 0.0
        metrics["flow/total"] = float(emp_loss.detach())
        return _ret(emp_loss, metrics)

    # Self-consistency rows: step exponent e ~ U{0..emax-1} -> d = 2^-e; sigma on
    # that step's grid (multiples of d, sigma <= 1-d). One big step (grad) must
    # match two half-steps at e+1 (no-grad), compared in velocity space with the
    # (1-sigma)^2 weight (upstream ``boot_per``).
    zs, ns = z[n_emp:], noise[n_emp:]
    as_, fs = action[n_emp:], (is_first[n_emp:] if is_first is not None else None)
    os_ = opponent_action[n_emp:] if opponent_action is not None else None
    us = opp_unknown[n_emp:] if opp_unknown is not None else None
    cws = cell_weight[n_emp:] if cell_weight is not None else None
    e = torch.randint(0, emax, (n_self, T), device=device)
    d = torch.pow(2.0, -e.float())
    ksteps = torch.pow(2.0, e.float())
    j2 = torch.floor(torch.rand(n_self, T, device=device) * ksteps)
    sigma = j2 * d                                          # in [0, 1-d]
    sig_idx = torch.round(sigma * k_max).long()
    z_tilde = interp(zs, ns, sigma)
    eps = 1e-4

    with torch.no_grad():
        half = e + 1
        d_half = 0.5 * d
        x1_h1 = wm.denoise(z_tilde, as_, fs, sig_idx, half,
                           opponent_action=os_, opp_unknown=us)
        b1 = (x1_h1 - z_tilde) / (1.0 - sigma).clamp_min(eps)[..., None, None]
        z_mid = z_tilde + b1 * d_half[..., None, None]
        sigma_mid = sigma + d_half
        sig_mid_idx = torch.round(sigma_mid * k_max).long()
        x1_h2 = wm.denoise(z_mid, as_, fs, sig_mid_idx, half,
                           opponent_action=os_, opp_unknown=us)
        b2 = (x1_h2 - z_mid) / (1.0 - sigma_mid).clamp_min(eps)[..., None, None]
        vbar = 0.5 * (b1 + b2)

    x1_big = wm.denoise(z_tilde, as_, fs, sig_idx, e,
                        opponent_action=os_, opp_unknown=us)
    vhat = (x1_big - z_tilde) / (1.0 - sigma).clamp_min(eps)[..., None, None]
    boot_per_cell = (vhat - vbar).pow(2).mean(dim=3)          # (n_self, T, n_spatial)
    boot_per = (boot_per_cell * cws).mean(dim=2) if cws is not None else boot_per_cell.mean(dim=2)
    boot_per = boot_per * (1.0 - sigma).pow(2)
    sc_loss = boot_per.mean()

    # Row-weighted combination = mean over the whole batch (upstream averages
    # per-row losses over all B rows).
    loss = (n_emp * emp_loss + n_self * sc_loss) / B
    metrics["flow/consistency"] = float(sc_loss.detach())
    metrics["flow/total"] = float(loss.detach())
    return _ret(loss, metrics)


# --- returns -------------------------------------------------------------
def lambda_return(reward, value, disc, lam=0.95):
    """Dreamer lambda-return.

    ``reward`` (B,H), ``value`` (B,H+1) (bootstrap value at each latent incl. the
    final one), ``disc`` (B,H) per-step discount (``gamma * continue``). Returns
    (B,H) targets ``R_t = r_t + disc_t * ((1-lam) V_{t+1} + lam R_{t+1})`` with
    ``R_{H-1}`` bootstrapping on ``V_H``.
    """
    H = reward.shape[1]
    returns = torch.zeros_like(reward)
    nxt = value[:, -1]
    for t in reversed(range(H)):
        returns[:, t] = reward[:, t] + disc[:, t] * ((1 - lam) * value[:, t + 1] + lam * nxt)
        nxt = returns[:, t]
    return returns


class ReturnNormalizer:
    """DreamerV3 percentile return scale.

    Tracks an EMA of ``P[high] - P[low]`` over lambda-return batches; advantages
    are divided by ``max(limit, S)`` so small (noise-level) return spreads are not
    amplified while large ones are normalized to unit scale.
    """

    def __init__(self, rate=0.01, limit=1.0, low=5.0, high=95.0):
        self.rate, self.limit, self.low, self.high = rate, limit, low, high
        self.scale: float | None = None

    def __call__(self, returns: torch.Tensor) -> float:
        flat = returns.detach().float().flatten()
        lo = torch.quantile(flat, self.low / 100.0)
        hi = torch.quantile(flat, self.high / 100.0)
        s = float(hi - lo)
        self.scale = s if self.scale is None else \
            (1.0 - self.rate) * self.scale + self.rate * s
        return max(self.limit, self.scale)

    def state_dict(self) -> dict:
        return {"scale": self.scale}

    def load_state_dict(self, d: dict) -> None:
        self.scale = d.get("scale")


# --- actor / critic ------------------------------------------------------
def actor_critic_losses(model, imagined, *, gamma=0.99, lam=0.95, entropy_coef=3e-3,
                        return_normalizer: ReturnNormalizer | None = None):
    """Policy-improvement losses on an imagined rollout (world model held fixed).

    ``imagined`` is the dict from ``DreamerV4.imagine``: ``z`` (B,H+1,...),
    ``action`` (B,H,...), ``mask`` (B,H,...), ``reward`` (B,H), ``cont`` (B,H).
    Returns ``(actor_loss, critic_loss, metrics)``.
    """
    ae = model.action_expert
    z = imagined["z"]                                       # detached latents
    B, Hp1 = z.shape[:2]
    H = Hp1 - 1

    # Scalar values (advantage/metrics) and slow target values, both without grad;
    # the critic's gradient comes from the two-hot logits below.
    z_flat = z.reshape(B * Hp1, *z.shape[2:])
    with torch.no_grad():
        values = ae.value(z_flat).reshape(B, Hp1)
        tgt_values = ae.target_value(z_flat).reshape(B, Hp1)

    disc = gamma * imagined["cont"]                         # (B,H)
    returns = lambda_return(imagined["reward"], tgt_values, disc, lam=lam)  # (B,H)

    # Critic: symlog two-hot classification of the lambda-return on the first H latents.
    value_logits = ae.value_logits(z[:, :H].reshape(B * H, *z.shape[2:]))
    critic_loss = ae.critic_coder.loss(value_logits, returns.detach().reshape(B * H))

    # Actor: REINFORCE on the (percentile-normalized) imagined advantage.
    zs = z[:, :H].reshape(B * H, *z.shape[2:])
    acts = imagined["action"].reshape(B * H, *imagined["action"].shape[2:])
    masks = imagined["mask"].reshape(B * H, *imagined["mask"].shape[2:])
    logprob, entropy, _ = ae.evaluate(zs, acts, masks)
    logprob = logprob.reshape(B, H)
    entropy = entropy.reshape(B, H)
    advantage = (returns - values[:, :H]).detach()
    ret_scale = return_normalizer(returns) if return_normalizer is not None else 1.0
    advantage = advantage / ret_scale
    actor_loss = -(logprob * advantage).mean() - entropy_coef * entropy.mean()

    metrics = {
        "ac/critic_loss": float(critic_loss.detach()),
        "ac/actor_loss": float(actor_loss.detach()),
        "ac/return_mean": float(returns.mean().detach()),
        "ac/return_scale": float(ret_scale),
        "ac/value_mean": float(values.mean().detach()),
        "ac/entropy": float(entropy.mean().detach()),
        "ac/advantage_std": float(advantage.std().detach()),
    }
    return actor_loss, critic_loss, metrics


def real_actor_critic_losses(model, z, batch, *, gamma=0.99, lam=0.95,
                             entropy_coef=3e-3,
                             return_normalizer: ReturnNormalizer | None = None):
    """Online-RL baseline: the same lambda-return critic + REINFORCE actor as
    :func:`actor_critic_losses`, but on **real replay sequences** instead of
    imagination — values from the critic on encoded real latents, the env's
    rewards/continues (``cont=0`` at a done row cuts the bootstrap across the
    autoreset boundary), the actions actually taken, and the **engine** action
    mask. No world model in the gradient path, so this isolates what the actor
    can learn from real experience alone — the sample-efficiency baseline the
    imagination modes are compared against.

    ``z`` (B,T,...) detached latents of ``batch["obs"]``; ``batch`` needs
    ``action``/``reward``/``cont``/``mask``/``is_first`` as collected. Slots whose
    stored mask has no legal action (terminal-spliced rows, dead player) are
    excluded from the actor loss and the critic targets.
    """
    ae = model.action_expert
    B, T = z.shape[:2]
    H = T - 1
    assert H >= 1, "need at least 2 frames for a return"
    action, mask = batch["action"], batch["mask"].float()
    reward, cont = batch["reward"].float(), batch["cont"].float()

    z_flat = z.reshape(B * T, *z.shape[2:])
    with torch.no_grad():
        values = ae.value(z_flat).reshape(B, T)
        tgt_values = ae.target_value(z_flat).reshape(B, T)

    disc = gamma * cont[:, :H]
    returns = lambda_return(reward[:, :H], tgt_values, disc, lam=lam)  # (B,H)

    # Rows with no legal action carry no self-consistent (state, action) pair.
    valid = mask[:, :H].flatten(2).any(dim=2).float()                  # (B,H)

    value_logits = ae.value_logits(z[:, :H].reshape(B * H, *z.shape[2:]))
    critic_loss = ae.critic_coder.loss(
        value_logits.reshape(B, H, -1), returns.detach(), mask=valid.bool())

    logprob, entropy, _ = ae.evaluate(
        z[:, :H].reshape(B * H, *z.shape[2:]),
        action[:, :H].reshape(B * H, *action.shape[2:]),
        mask[:, :H].reshape(B * H, *mask.shape[2:]))
    logprob = logprob.reshape(B, H)
    entropy = entropy.reshape(B, H)
    advantage = (returns - values[:, :H]).detach()
    ret_scale = return_normalizer(returns) if return_normalizer is not None else 1.0
    advantage = advantage / ret_scale
    denom = valid.sum().clamp_min(1.0)
    actor_loss = -((logprob * advantage + entropy_coef * entropy) * valid).sum() / denom

    metrics = {
        "ac/critic_loss": float(critic_loss.detach()),
        "ac/actor_loss": float(actor_loss.detach()),
        "ac/return_mean": float(returns.mean().detach()),
        "ac/return_scale": float(ret_scale),
        "ac/value_mean": float(values.mean().detach()),
        "ac/entropy": float(entropy.mean().detach()),
        "ac/advantage_std": float(advantage.std().detach()),
        "ac/valid_frac": float(valid.mean().detach()),
    }
    return actor_loss, critic_loss, metrics
