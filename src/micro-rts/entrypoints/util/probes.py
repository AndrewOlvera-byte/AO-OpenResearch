"""Action-causality probes for the joint-action world model (NEXT_PLAN.md, E).

The v2 failure mode was an *action-ignoring* world model: excellent
teacher-forced metrics, useless generatively, and an actor that maximized reward
inside action-insensitive mush. These probes measure the one thing that matters
before any imagination RL: **does the open-loop rollout actually depend on the
actions it is conditioned on?**

``counterfactual_action_probe`` rolls the model open-loop four ways — true
actions; batch-shuffled player-1 actions; batch-shuffled opponent actions; both
shuffled — and reports the MSE gaps vs the true-action rollout. Shuffling across
the batch keeps each stream's marginal statistics (same actions, same rates)
while destroying its pairing with the state, so a positive gap can only come
from the model *using* the channel.

The aggregate whole-frame gap has a built-in SNR ceiling: a newly issued action
touches ~1 latent cell in 16, so even a PERFECT model's frame-mean gap is tiny
(measured: a +3.7% local effect dilutes to ~+0.0005 aggregate — below the noise
floor of any affordable probe batch). The **issued-cell-conditioned** gaps are
therefore the primary gate: the same MSE difference measured only at latent
cells whose driving frame issued an action there, where a real effect cannot
hide. The aggregate gaps are kept as secondary context.

Gates (do not start RL until both hold):
  1. ``probe/self_gap_issued`` > 0 and growing over training (primary; the
     aggregate ``self_gap``/growth are context only),
  2. ``probe/opp_gap_issued``  > 0 likewise.

NOTE: issuance is read as ``action[..., 0] != 0`` on the tensors given —
callers running ``mask_junk_actions`` pass already-masked actions, making this
exact; unmasked callers get a junk-diluted (conservative) version.
"""

from __future__ import annotations

import torch


def _issued_latent_cells(action, ctx, T_gen, h, w, down=4):
    """(B,T_gen,n_spatial) bool: latent cells whose DRIVING frame issued an
    action there. Generated frame ctx+i is driven (arrive-aligned shift) by the
    action at frame ctx+i-1; the raw H*W issuance grid is max-pooled by the
    tokenizer's downsample factor so it lines up with per-latent-cell MSE."""
    B = action.shape[0]
    iss = (action[:, ctx - 1:ctx - 1 + T_gen, :, 0] != 0)          # (B,T_gen,H*W)
    iss = iss.reshape(B, T_gen, h // down, down, w // down, down)
    return iss.amax(dim=(3, 5)).flatten(2)                          # (B,T_gen,n_spatial)


@torch.no_grad()
def counterfactual_action_probe(model, z, action, opponent_action, is_first=None,
                                *, context: int = 8, flow_steps: int = 4,
                                head_slots: int = 3) -> dict[str, float]:
    """Open-loop rollout MSE under true vs shuffled action streams.

    ``z`` raw tokenizer latents (B,T,n_spatial,d_latent); ``action`` /
    ``opponent_action`` (B,T,H*W,7) frame-aligned; ``context`` real frames seed
    the rollout, the remaining ``T-context`` slots are generated. Returns
    ``probe/*`` metrics: per-variant MSE, gaps vs true, and the late-minus-early
    gap growth over the horizon (mean of the last vs first ``head_slots``
    generated slots).
    """
    wm = model.world_model
    B, T = z.shape[:2]
    ctx = min(context, T - 1)
    T_gen = T - ctx
    z_tgt = wm.normalize(z[:, ctx:])
    if B > 1:
        perm_a = torch.randperm(B, device=z.device)
        perm_o = torch.roll(perm_a, 1)          # independent-ish second derangement
    else:
        perm_a = perm_o = torch.zeros(1, dtype=torch.long, device=z.device)

    def rollout_mse(act, opp):
        pred = model.open_loop(z, act, is_first, context=ctx,
                               flow_steps=flow_steps, opponent_action=opp)
        return (wm.normalize(pred) - z_tgt).pow(2).mean(dim=3)     # (B,T_gen,n_spatial)

    variants = {
        "true": (action, opponent_action),
        "self_shuffled": (action[perm_a], opponent_action),
    }
    # Without an opponent stream (e.g. the RL replay buffer) only the
    # self-channel probe runs; the opp variants need recorded opponent actions.
    if opponent_action is not None:
        variants["opp_shuffled"] = (action, opponent_action[perm_o])
        variants["both_shuffled"] = (action[perm_a], opponent_action[perm_o])
    mse_c = {name: rollout_mse(a, o) for name, (a, o) in variants.items()}
    mse_t = {name: m.mean(dim=(0, 2)) for name, m in mse_c.items()}  # (T_gen,)

    enc = wm.action_encoder
    iss_self = _issued_latent_cells(action, ctx, T_gen, enc.h, enc.w)
    iss_opp = _issued_latent_cells(opponent_action, ctx, T_gen, enc.h, enc.w) \
        if opponent_action is not None else None

    h = min(head_slots, T_gen)
    out = {}
    for name, m in mse_t.items():
        out[f"probe/mse_{name}"] = float(m.mean())
    for name, iss in (("self", iss_self), ("opp", iss_opp), ("both", iss_self)):
        key = f"{name}_shuffled"
        if key not in mse_c:
            continue
        gap_t = mse_t[key] - mse_t["true"]
        out[f"probe/{name}_gap"] = float(gap_t.mean())
        out[f"probe/{name}_gap_growth"] = float(gap_t[-h:].mean() - gap_t[:h].mean())
        # PRIMARY gate: the same gap at issued cells only (see module docstring).
        if iss is not None and bool(iss.any()):
            out[f"probe/{name}_gap_issued"] = \
                float(mse_c[key][iss].mean() - mse_c["true"][iss].mean())
    out["probe/issued_cell_frac"] = float(iss_self.float().mean())
    return out
