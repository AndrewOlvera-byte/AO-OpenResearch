from __future__ import annotations

import torch
import torch.nn.functional as F

from core.registry import register

from .encoder import masked_action_pool, split_branches


def _norm(value):
    return F.layer_norm(value.float(), value.shape[-1:])


def _jepa(prediction, target):
    return F.smooth_l1_loss(_norm(prediction), _norm(target.detach()))


def _event_values(prediction):
    """Convert change-support logits into the semantic event feature space."""
    return torch.cat((prediction[..., :1].sigmoid(), prediction[..., 1:]), dim=-1)


def _sparse_event_loss(prediction, target, valid=None):
    """Balance sparse change support and regress magnitudes on changed patches."""
    support = target.float().abs().amax(-1) > 1e-6
    logits = prediction[..., 0].float()
    positives = support.sum().float()
    negatives = support.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 32.0)
    support_per = F.binary_cross_entropy_with_logits(
        logits,
        support.float(),
        pos_weight=pos_weight,
        reduction="none",
    ).mean(-1)
    regression_per = F.smooth_l1_loss(
        prediction[..., 1:].float(), target[..., 1:].float(), reduction="none"
    ).mean(-1)
    patch_weight = 0.1 + 4.0 * support.float()
    regression_per = (regression_per * patch_weight).sum(-1) / patch_weight.sum(-1)
    per = support_per + regression_per
    if valid is None:
        return per.mean()
    weight = valid.to(per.dtype)
    return (per * weight).sum() / weight.sum().clamp_min(1.0)


def _window_sum(value, horizon, discount=1.0):
    pieces = [value[:, index : value.shape[1] - horizon + index] for index in range(horizon)]
    weights = value.new_tensor([discount**index for index in range(horizon)])
    stacked = torch.stack(pieces, dim=-1)
    return (stacked * weights).sum(-1)


def semantic_transition_target(state, next_state, globals_, next_globals, downsample=2):
    """Compact transition events; no absolute hidden-state reconstruction target."""
    present0, present1 = state[..., 1].float(), next_state[..., 1].float()
    self0 = present0 * (state[..., 3] == 0).float()
    self1 = present1 * (next_state[..., 3] == 0).float()
    opp0 = present0 * (state[..., 3] == 1).float()
    opp1 = present1 * (next_state[..., 3] == 1).float()
    hp0, hp1 = state[..., 5].float(), next_state[..., 5].float()
    changed = (state[..., 1:] != next_state[..., 1:]).any(-1).float()
    features = torch.stack(
        (
            changed,
            present1 - present0,
            self1 - self0,
            opp1 - opp0,
            (hp1 - hp0) / 16.0,
            (next_state[..., 4] != state[..., 4]).float() * changed,
        ),
        dim=-1,
    )
    h = w = int(state.shape[-2] ** 0.5)
    lead = features.shape[:-2]
    grid = features.reshape(-1, h, w, features.shape[-1]).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(grid, downsample, downsample).permute(0, 2, 3, 1)
    pooled = pooled.reshape(*lead, -1, features.shape[-1])
    resource = (next_globals[..., 1:3] - globals_[..., 1:3]).float() / 16.0
    resource = resource[..., None, :].expand(*pooled.shape[:-1], 2)
    return torch.cat((pooled, resource), dim=-1)


@register("loss", "predictive_belief")
def predictive_belief_loss(
    model,
    batch,
    *,
    future_jepa_coef=1.0,
    variance_coef=0.05,
    self_inverse_coef=0.5,
    opponent_plan_coef=0.5,
    events_coef=0.5,
    reward_coef=0.5,
    return_coef=0.5,
    continue_coef=0.1,
    counterfactual_coef=1.0,
    counterfactual_effect_coef=1.0,
):
    encoded = model(batch)
    online, target = encoded["online"], encoded["target"]
    cfg = model.cfg
    zero = online["tokens"].new_zeros(())

    jepa_parts = []
    reward_loss = return_loss = continue_loss = zero
    reward_count = 0
    for horizon, prediction in encoded["predictions"].items():
        target_tokens = target["tokens"][:, horizon:]
        jepa_parts.append(_jepa(prediction, target_tokens))
        scalar = model.scalar_predictions(prediction)
        returns = _window_sum(batch["reward"], horizon, cfg.discount)
        continues = torch.stack(
            [
                batch["cont"][:, index : batch["cont"].shape[1] - horizon + index]
                for index in range(horizon)
            ],
            dim=-1,
        ).prod(-1)
        return_loss = return_loss + F.smooth_l1_loss(scalar[..., 1], returns)
        continue_loss = continue_loss + F.binary_cross_entropy_with_logits(
            scalar[..., 2], continues
        )
        reward_count += 1
        if horizon == 1:
            reward_loss = F.smooth_l1_loss(scalar[..., 0], batch["reward"][:, :-1])
    jepa = torch.stack(jepa_parts).mean()
    return_loss = return_loss / max(reward_count, 1)
    continue_loss = continue_loss / max(reward_count, 1)

    flat = online["tokens"].float().flatten(0, -2)
    variance = F.relu(1.0 - flat.std(0, unbiased=False)).mean()

    pred1 = encoded["predictions"][1]
    future = split_branches(pred1, cfg.branch_sizes)
    source = {name: value[:, :-1] for name, value in online.items() if name != "tokens"}
    inverse_prediction = model.inverse_action(source["self"], future["self"])
    inverse_target = encoded["action_pool"][:, :-1]
    inverse = _jepa(inverse_prediction, inverse_target)

    anchors = batch["state"].shape[1] - model.opponent_tokenizer.max_horizon
    with torch.no_grad():
        opponent_target = model.opponent_tokenizer.encode(
            batch["state"], batch["opponent_action"]
        )[0]
    opponent_prediction = model.opponent_plan_head(online["opponent"][:, :anchors])
    opponent = _jepa(opponent_prediction, opponent_target)

    factual_target = semantic_transition_target(
        batch["state"][:, :-1],
        batch["next_state"][:, :-1],
        batch["globals"][:, :-1],
        batch["next_globals"][:, :-1],
        model.ego_tokenizer.cfg.downsample,
    )
    event_prediction = model.event_head(future["interaction"])
    event = _sparse_event_loss(event_prediction, factual_target)

    counterfactual = counterfactual_effect = effect_cosine = zero
    if "counterfactual_valid" in batch:
        cf_prediction_tokens = model.predict_counterfactual(encoded, batch)
        cf_parts = split_branches(cf_prediction_tokens, cfg.branch_sizes)
        cf_prediction = model.event_head(cf_parts["interaction"])
        cf_target = semantic_transition_target(
            batch["state"][:, :-1],
            batch["counterfactual_next_state"][:, :-1],
            batch["globals"][:, :-1],
            batch["counterfactual_next_globals"][:, :-1],
            model.ego_tokenizer.cfg.downsample,
        )
        valid = batch["counterfactual_valid"][:, :-1].float()
        counterfactual = _sparse_event_loss(cf_prediction, cf_target, valid)
        predicted_effect = _event_values(cf_prediction).float() - _event_values(
            event_prediction
        ).float()
        target_effect = cf_target.float() - factual_target.float()
        effect_patch = target_effect.abs().amax(-1) > 1e-6
        patch_weight = 0.05 + 8.0 * effect_patch.float()
        effect_per = (
            (predicted_effect - target_effect).square().mean(-1) * patch_weight
        ).sum(-1) / patch_weight.sum(-1)
        active = valid * (target_effect.flatten(-2).norm(dim=-1) > 1e-5).float()
        active_denom = active.sum().clamp_min(1.0)
        cosine = F.cosine_similarity(
            predicted_effect.flatten(-2), target_effect.flatten(-2), dim=-1, eps=1e-8
        )
        pred_norm = predicted_effect.flatten(-2).norm(dim=-1)
        target_norm = target_effect.flatten(-2).norm(dim=-1)
        direction = ((1.0 - cosine) * active).sum() / active_denom
        magnitude = (
            F.smooth_l1_loss(
                pred_norm.clamp_min(1e-6).log(),
                target_norm.clamp_min(1e-6).log(),
                reduction="none",
            )
            * active
        ).sum() / active_denom
        counterfactual_effect = (
            (effect_per * active).sum() / active_denom
            + direction
            + 0.25 * magnitude
        )
        effect_cosine = (cosine * active).sum() / active_denom

    total = (
        float(future_jepa_coef) * jepa
        + float(variance_coef) * variance
        + float(self_inverse_coef) * inverse
        + float(opponent_plan_coef) * opponent
        + float(events_coef) * event
        + float(reward_coef) * reward_loss
        + float(return_coef) * return_loss
        + float(continue_coef) * continue_loss
        + float(counterfactual_coef) * counterfactual
        + float(counterfactual_effect_coef) * counterfactual_effect
    )
    metrics = {
        "world_action_encoder/total": total.detach(),
        "world_action_encoder/future_jepa": jepa.detach(),
        "world_action_encoder/variance": variance.detach(),
        "world_action_encoder/self_inverse": inverse.detach(),
        "world_action_encoder/opponent_plan": opponent.detach(),
        "world_action_encoder/events": event.detach(),
        "world_action_encoder/reward": reward_loss.detach(),
        "world_action_encoder/return": return_loss.detach(),
        "world_action_encoder/continue": continue_loss.detach(),
        "world_action_encoder/counterfactual": counterfactual.detach(),
        "world_action_encoder/counterfactual_effect": counterfactual_effect.detach(),
        "world_action_encoder/counterfactual_effect_cosine": effect_cosine.detach(),
        "world_action_encoder/latent_rms": online["tokens"].float().square().mean().sqrt().detach(),
    }
    return total, metrics


def _static_zeroed(delta, sizes):
    """Zero the static branch of a branch-factorized tensor (static is immutable)."""
    parts = split_branches(delta, sizes)
    parts["static"] = torch.zeros_like(parts["static"])
    return torch.cat(tuple(parts.values()), dim=-2)


@register("loss", "causal_world_action_dynamics")
def factorized_world_action_dynamics_loss(
    model,
    batch,
    *,
    flow_transition_coef=1.0,
    multi_horizon_coef=1.0,
    self_inverse_coef=0.5,
    opponent_inverse_coef=0.5,
    events_coef=0.5,
    reward_coef=0.5,
    return_coef=0.5,
    continue_coef=0.1,
    counterfactual_coef=1.0,
    counterfactual_effect_coef=1.0,
    counterfactual_preference_coef=0.5,
    reconstruction_coef=0.0,
    horizons=(1, 2, 4),
):
    """Composed transition loss for ``FactorizedWorldActionDynamics``.

    Trains the intrinsic/extrinsic/interaction flow to transport the frozen
    stage-1 belief from ``b_t`` to ``b_{t+1}`` under the simultaneous ego action
    and opponent plan, plus readout heads (events / scalars / inverse) decoded
    from the *predicted* next belief and counterfactual objectives.  Emits
    ``world_action_dynamics/*`` metrics (``monitor: world_action_dynamics/total``).
    """
    dynamics = model.dynamics
    sizes = dynamics.cfg.branch_sizes
    downsample = model.ego_tokenizer.cfg.downsample
    discount = float(model.belief_cfg.discount)

    encoded = model(batch)  # frozen tokenization (no grad)
    belief, action, valid, plan = (
        encoded["belief"],
        encoded["action"],
        encoded["valid"],
        encoded["plan"],
    )
    zero = belief.new_zeros(())
    # Aligned transition window t -> t+1 with an opponent plan available at t.
    length = min(plan.shape[1], belief.shape[1] - 1)
    departure = belief[:, :length]
    next_belief = belief[:, 1:length + 1]
    ego_action = action[:, :length]
    plan_t = plan[:, :length]
    state = batch["state"][:, :length]
    next_state = batch["next_state"][:, :length]
    globals_ = batch["globals"][:, :length]
    next_globals = batch["next_globals"][:, :length]

    # --- flow-matching transition (rectified flow on branch deltas) ----------
    delta = _static_zeroed(next_belief - departure, sizes)
    noise = _static_zeroed(torch.randn_like(delta), sizes)
    tau = torch.rand(departure.shape[:-2], device=delta.device, dtype=delta.dtype)
    interpolated = (1.0 - tau[..., None, None]) * noise + tau[..., None, None] * delta
    velocity = dynamics.velocity(interpolated, tau, departure, ego_action, plan_t)
    flow_transition = F.smooth_l1_loss(velocity.float(), (delta - noise).float())

    # --- single-step prediction reused by the readouts / counterfactuals -----
    predicted = dynamics.transition(departure, ego_action, plan_t)
    predicted_parts = split_branches(predicted, sizes)
    source_parts = split_branches(departure, sizes)

    # --- multi-horizon rollout consistency (truncated BPTT) ------------------
    # The rollout prefix runs under no_grad; only the final transition at each
    # horizon carries gradient, which keeps the retained graph (and memory)
    # bounded to a single flow integration regardless of horizon.
    multi_parts = []
    for horizon in horizons:
        span = length - horizon + 1
        if span < 1:
            continue
        rolled = belief[:, :span]
        with torch.no_grad():
            for step in range(horizon - 1):
                rolled = dynamics.transition(
                    rolled, action[:, step:step + span], plan[:, step:step + span]
                )
        rolled = dynamics.transition(
            rolled,
            action[:, horizon - 1:horizon - 1 + span],
            plan[:, horizon - 1:horizon - 1 + span],
        )
        multi_parts.append(_jepa(rolled, belief[:, horizon:horizon + span]))
    multi_horizon = torch.stack(multi_parts).mean() if multi_parts else zero

    # --- inverse heads off (b_t, predicted b_{t+1}) --------------------------
    self_inverse = _jepa(
        model.self_inverse(
            torch.cat(
                (source_parts["self"].mean(-2), predicted_parts["self"].mean(-2)), dim=-1
            )
        ),
        masked_action_pool(ego_action, valid[:, :length]),
    )
    opponent_inverse = _jepa(
        model.opponent_inverse(
            torch.cat((source_parts["opponent"], predicted_parts["opponent"]), dim=-1)
        ),
        plan_t,
    )

    # --- events / scalars off the predicted next belief ----------------------
    factual_target = semantic_transition_target(
        state, next_state, globals_, next_globals, downsample
    )
    event_prediction = model.event_head(predicted_parts["interaction"])
    events = _sparse_event_loss(event_prediction, factual_target)

    scalar = model.scalar_predictions(predicted)
    reward_loss = F.smooth_l1_loss(scalar[..., 0], batch["reward"][:, :length])
    returns = _window_sum(batch["reward"], 1, discount)[:, :length]
    return_loss = F.smooth_l1_loss(scalar[..., 1], returns)
    continue_loss = F.binary_cross_entropy_with_logits(
        scalar[..., 2], batch["cont"][:, :length]
    )

    # --- counterfactual action intervention ----------------------------------
    counterfactual = counterfactual_effect = counterfactual_preference = zero
    effect_cosine = zero
    if "counterfactual_valid" in batch:
        cf_action, _ = model.encode_actions(batch, "counterfactual_action")
        cf_predicted = dynamics.transition(departure, cf_action[:, :length], plan_t)
        cf_parts = split_branches(cf_predicted, sizes)
        cf_prediction = model.event_head(cf_parts["interaction"])
        cf_target = semantic_transition_target(
            state,
            batch["counterfactual_next_state"][:, :length],
            globals_,
            batch["counterfactual_next_globals"][:, :length],
            downsample,
        )
        cf_valid = batch["counterfactual_valid"][:, :length].float()
        counterfactual = _sparse_event_loss(cf_prediction, cf_target, cf_valid)

        predicted_effect = _event_values(cf_prediction).float() - _event_values(
            event_prediction
        ).float()
        target_effect = cf_target.float() - factual_target.float()
        patch_weight = 0.05 + 8.0 * (target_effect.abs().amax(-1) > 1e-6).float()
        effect_per = (
            (predicted_effect - target_effect).square().mean(-1) * patch_weight
        ).sum(-1) / patch_weight.sum(-1)
        active = cf_valid * (target_effect.flatten(-2).norm(dim=-1) > 1e-5).float()
        denom = active.sum().clamp_min(1.0)
        cosine = F.cosine_similarity(
            predicted_effect.flatten(-2), target_effect.flatten(-2), dim=-1, eps=1e-8
        )
        direction = ((1.0 - cosine) * active).sum() / denom
        counterfactual_effect = (effect_per * active).sum() / denom + direction
        effect_cosine = (cosine * active).sum() / denom

        # Preference: predicted return must order factual vs counterfactual the
        # same way the realized resource outcome does.
        cf_scalar = model.scalar_predictions(cf_predicted)
        factual_gain = (next_globals[..., 1:3] - globals_[..., 1:3]).float().sum(-1)
        cf_gain = (
            batch["counterfactual_next_globals"][:, :length, 1:3] - globals_[..., 1:3]
        ).float().sum(-1)
        target_sign = torch.sign(cf_gain - factual_gain)
        pred_diff = cf_scalar[..., 1] - scalar[..., 1]
        pref_active = cf_valid * (target_sign.abs() > 0).float()
        counterfactual_preference = (
            F.relu(0.1 - target_sign * pred_diff) * pref_active
        ).sum() / pref_active.sum().clamp_min(1.0)

    reconstruction = zero
    if reconstruction_coef:
        reconstruction = _jepa(predicted, next_belief)

    total = (
        float(flow_transition_coef) * flow_transition
        + float(multi_horizon_coef) * multi_horizon
        + float(self_inverse_coef) * self_inverse
        + float(opponent_inverse_coef) * opponent_inverse
        + float(events_coef) * events
        + float(reward_coef) * reward_loss
        + float(return_coef) * return_loss
        + float(continue_coef) * continue_loss
        + float(counterfactual_coef) * counterfactual
        + float(counterfactual_effect_coef) * counterfactual_effect
        + float(counterfactual_preference_coef) * counterfactual_preference
        + float(reconstruction_coef) * reconstruction
    )
    metrics = {
        "world_action_dynamics/total": total.detach(),
        "world_action_dynamics/flow_transition": flow_transition.detach(),
        "world_action_dynamics/multi_horizon": multi_horizon.detach(),
        "world_action_dynamics/self_inverse": self_inverse.detach(),
        "world_action_dynamics/opponent_inverse": opponent_inverse.detach(),
        "world_action_dynamics/events": events.detach(),
        "world_action_dynamics/reward": reward_loss.detach(),
        "world_action_dynamics/return": return_loss.detach(),
        "world_action_dynamics/continue": continue_loss.detach(),
        "world_action_dynamics/counterfactual": counterfactual.detach(),
        "world_action_dynamics/counterfactual_effect": counterfactual_effect.detach(),
        "world_action_dynamics/counterfactual_effect_cosine": effect_cosine.detach(),
        "world_action_dynamics/counterfactual_preference": counterfactual_preference.detach(),
        "world_action_dynamics/reconstruction": reconstruction.detach(),
    }
    return total, metrics
