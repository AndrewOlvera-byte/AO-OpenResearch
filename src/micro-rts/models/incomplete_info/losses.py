from __future__ import annotations

import torch
import torch.nn.functional as F

from models.dreamer_v2 import structured_reconstruction_loss

from .ego_tokenizer import OBS_GROUPS, EgoTokenizerPretrainer
from .heads import event_targets


def _weighted_mean(value, weight):
    weight = weight.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _latent_distance(prediction, target):
    """Scale-insensitive JEPA distance without collapsing token geometry."""
    prediction = F.layer_norm(prediction.float(), prediction.shape[-1:])
    target = F.layer_norm(target.float(), target.shape[-1:])
    return F.smooth_l1_loss(prediction, target)


def event_reconstruction_loss(decoder, logits, events, valid, prefix="events"):
    valid_f = valid.to(logits["valid"].dtype)
    losses = {
        "valid": F.binary_cross_entropy_with_logits(
            logits["valid"].squeeze(-1), valid_f
        )
    }
    targets = event_targets(events, decoder)
    denom = valid_f.sum().clamp_min(1.0)
    metrics = {}
    for name, target in targets.items():
        per = F.cross_entropy(
            logits[name].flatten(0, -2), target.flatten(), reduction="none"
        ).reshape_as(target)
        losses[name] = (per * valid_f).sum() / denom
        metrics[f"{prefix}/{name}_acc"] = (
            ((logits[name].argmax(-1) == target) & valid).sum() / denom
        ).detach()
    total = torch.stack(tuple(losses.values())).mean()
    metrics.update(
        {
            f"{prefix}/loss": total.detach(),
            f"{prefix}/valid_acc": ((logits["valid"].squeeze(-1) >= 0) == valid)
            .float()
            .mean()
            .detach(),
        }
    )
    return total, metrics


def ego_tokenizer_loss(
    model: EgoTokenizerPretrainer,
    batch: dict,
    *,
    full_state_tokenizer=None,
    reconstruction_coef=1.0,
    visibility_coef=0.1,
    jepa_coef=0.25,
    teacher_coef=0.25,
):
    obs = batch["local_obs"].float()
    visibility = batch["local_visibility"].float()
    lead = obs.shape[:-3]
    visible_blocks = (
        F.avg_pool2d(
            visibility.reshape(-1, *visibility.shape[-3:]),
            model.tokenizer.cfg.downsample,
            model.tokenizer.cfg.downsample,
        ).flatten(1)
        > 0
    )
    patch_mask = (
        torch.rand_like(visible_blocks.float())
        < float(model.tokenizer.cfg.mask_fraction)
    ) & visible_blocks
    patch_mask = patch_mask.reshape(*lead, -1)

    decoded, spatial, _registers, _ = model.tokenizer(obs, visibility)
    masked_spatial, _, _ = model.tokenizer.encode(obs, visibility, patch_mask)
    with torch.no_grad():
        target_spatial, _, _ = model.target.encode(obs, visibility)

    target_groups = torch.split(obs, OBS_GROUPS, dim=-3)
    prediction_groups = torch.split(decoded["obs_logits"], OBS_GROUPS, dim=-3)
    visible = visibility.squeeze(-3)
    recon_parts = []
    for prediction, target in zip(prediction_groups, target_groups):
        per = F.cross_entropy(
            prediction.movedim(-3, -1).reshape(-1, prediction.shape[-3]),
            target.argmax(-3).reshape(-1),
            reduction="none",
        ).reshape_as(visible)
        recon_parts.append(_weighted_mean(per, visible))
    reconstruction = torch.stack(recon_parts).mean()
    visibility_loss = F.binary_cross_entropy_with_logits(
        decoded["visibility_logits"], visibility
    )

    predicted_masked = model.jepa_predictor(masked_spatial)
    mask_f = patch_mask[..., None].to(predicted_masked.dtype)
    jepa = _weighted_mean(
        F.smooth_l1_loss(
            F.layer_norm(predicted_masked.float(), predicted_masked.shape[-1:]),
            F.layer_norm(target_spatial.float(), target_spatial.shape[-1:]),
            reduction="none",
        ).mean(-1),
        mask_f.squeeze(-1),
    )

    teacher = reconstruction.new_zeros(())
    if full_state_tokenizer is not None:
        with torch.no_grad():
            full = full_state_tokenizer.encode(batch["state"], batch["globals"])[
                ..., : model.tokenizer.n_spatial, :
            ]
        per_token = F.smooth_l1_loss(
            F.layer_norm(spatial.float(), spatial.shape[-1:]),
            F.layer_norm(full.float(), full.shape[-1:]),
            reduction="none",
        ).mean(-1)
        teacher = _weighted_mean(per_token, visible_blocks.reshape(*lead, -1))

    total = (
        float(reconstruction_coef) * reconstruction
        + float(visibility_coef) * visibility_loss
        + float(jepa_coef) * jepa
        + float(teacher_coef) * teacher
    )
    metrics = {
        "ego/total": total.detach(),
        "ego/reconstruction": reconstruction.detach(),
        "ego/visibility": visibility_loss.detach(),
        "ego/jepa": jepa.detach(),
        "ego/full_teacher": teacher.detach(),
        "ego/masked_fraction": patch_mask.float().mean().detach(),
    }
    return total, metrics


def self_action_tokenizer_loss(
    model,
    full_state_tokenizer,
    batch,
    *,
    reconstruction_coef=1.0,
    forward_coef=1.0,
    changed_token_boost=8.0,
):
    tokens, events, valid, overflow = model(batch["local_obs"], batch["action"])
    reconstruction, metrics = event_reconstruction_loss(
        model.decoder, model.decode_events(tokens), events, valid, "self_action"
    )
    with torch.no_grad():
        z0 = F.layer_norm(
            full_state_tokenizer.encode(batch["state"], batch["globals"]).float(),
            (full_state_tokenizer.d_latent,),
        )
        z1 = F.layer_norm(
            full_state_tokenizer.encode(
                batch["next_state"], batch["next_globals"]
            ).float(),
            (full_state_tokenizer.d_latent,),
        )
        target_delta = z1 - z0
    predicted_delta = model.predict_effect(z0, tokens, valid)
    changed = target_delta.float().square().mean(-1) > 1e-6
    weights = 1.0 + (float(changed_token_boost) - 1.0) * changed.float()
    forward = _weighted_mean(
        (predicted_delta.float() - target_delta.float()).square().mean(-1), weights
    )
    total = float(reconstruction_coef) * reconstruction + float(forward_coef) * forward
    metrics.update(
        {
            "self_action/total": total.detach(),
            "self_action/forward": forward.detach(),
            "self_action/overflow": overflow.float().mean().detach(),
        }
    )
    return total, metrics


def opponent_plan_tokenizer_loss(
    model,
    full_state_tokenizer,
    batch,
    *,
    event_coef=1.0,
    future_state_coef=0.5,
):
    plan, events, valid, overflow = model.encode(
        batch["state"], batch["opponent_action"]
    )
    event_loss, metrics = event_reconstruction_loss(
        model.event_decoder,
        model.decode_events(plan),
        events,
        valid,
        "opponent_plan",
    )
    anchors = plan.shape[1]
    horizon = model.max_horizon
    with torch.no_grad():
        target = full_state_tokenizer.encode(
            batch["state"][:, horizon : horizon + anchors],
            batch["globals"][:, horizon : horizon + anchors],
        )
    future = _latent_distance(model.predict_future_state(plan), target)
    total = float(event_coef) * event_loss + float(future_state_coef) * future
    metrics.update(
        {
            "opponent_plan/total": total.detach(),
            "opponent_plan/future_state": future.detach(),
            "opponent_plan/overflow": overflow.float().mean().detach(),
        }
    )
    return total, metrics


def opponent_intent_prior_loss(
    model,
    batch,
    *,
    latent_coef=1.0,
    event_coef=1.0,
    mode_coef=0.25,
    balance_coef=0.1,
    contrastive_coef=0.5,
    shuffled_margin_coef=0.5,
    diversity_coef=0.05,
    shuffled_margin=0.25,
    diversity_floor=0.75,
    contrastive_temperature=0.1,
):
    """Match privileged plan targets with explicit history-conditioned modes."""
    plans, mode_logits, history_embedding, _ = model(batch)
    with torch.no_grad():
        target_plan, target_events, target_valid, overflow = (
            model.opponent_tokenizer.encode(
                batch["state"], batch["opponent_action"]
            )
        )
        target = model.normalize_plan(target_plan).float()

    per_mode = F.smooth_l1_loss(
        plans.float(), target[:, :, None].expand_as(plans).float(), reduction="none"
    ).mean(dim=(-1, -2))
    winner = per_mode.argmin(-1)
    best_distance = per_mode.gather(-1, winner[..., None]).squeeze(-1)
    latent = best_distance.mean()
    mode = F.cross_entropy(mode_logits.flatten(0, -2), winner.flatten())
    selected, _ = model.intent_prior.select(plans, F.one_hot(
        winner, plans.shape[-3]
    ).to(plans.dtype) * 1e4)
    selected_raw = model.denormalize_plan(selected)
    event, event_metrics = event_reconstruction_loss(
        model.opponent_tokenizer.event_decoder,
        model.opponent_tokenizer.decode_events(selected_raw),
        target_events,
        target_valid,
        "intent_best",
    )

    # Contrast histories only against other sequences at the same anchor time;
    # neighboring anchors from one episode are not treated as false negatives.
    target_embedding = model.intent_prior.target_embedding(target)
    if target.shape[0] > 1:
        similarity = torch.einsum(
            "abc,adc->abd",
            history_embedding.transpose(0, 1),
            target_embedding.transpose(0, 1),
        ) / float(contrastive_temperature)
        labels = torch.arange(target.shape[0], device=target.device)
        labels = labels[None].expand(target.shape[1], -1).reshape(-1)
        contrastive = 0.5 * (
            F.cross_entropy(similarity.flatten(0, 1), labels)
            + F.cross_entropy(similarity.transpose(-1, -2).flatten(0, 1), labels)
        )
        shuffled = plans.roll(1, 0)
        shuffled_distance = F.smooth_l1_loss(
            shuffled.float(),
            target[:, :, None].expand_as(shuffled).float(),
            reduction="none",
        ).mean(dim=(-1, -2)).amin(-1)
        shuffled_rank = F.relu(
            float(shuffled_margin) + best_distance - shuffled_distance
        ).mean()
    else:
        contrastive = latent.new_zeros(())
        shuffled_rank = latent.new_zeros(())

    flat_modes = plans.float().flatten(-2)
    pairwise = torch.cdist(flat_modes, flat_modes) / flat_modes.shape[-1] ** 0.5
    mask = ~torch.eye(plans.shape[-3], dtype=torch.bool, device=plans.device)
    diversity_rms = pairwise[..., mask].mean()
    diversity = F.relu(float(diversity_floor) - pairwise[..., mask]).mean()

    probabilities = mode_logits.softmax(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1).mean()
    mean_probability = probabilities.flatten(0, -2).mean(0)
    balance = (
        mean_probability
        * (
            mean_probability.clamp_min(1e-8).log()
            + torch.log(mean_probability.new_tensor(float(plans.shape[-3])))
        )
    ).sum()
    total = (
        float(latent_coef) * latent
        + float(event_coef) * event
        + float(mode_coef) * mode
        + float(balance_coef) * balance
        + float(contrastive_coef) * contrastive
        + float(shuffled_margin_coef) * shuffled_rank
        + float(diversity_coef) * diversity
    )
    top_plan, _ = model.intent_prior.select(plans, mode_logits)
    with torch.no_grad():
        top_event, top_metrics = event_reconstruction_loss(
            model.opponent_tokenizer.event_decoder,
            model.opponent_tokenizer.decode_events(model.denormalize_plan(top_plan)),
            target_events,
            target_valid,
            "intent_top1",
        )
    metrics = {
        "intent/total": total.detach(),
        "intent/latent": latent.detach(),
        "intent/event": event.detach(),
        "intent/mode_ce": mode.detach(),
        "intent/mode_balance": balance.detach(),
        "intent/contrastive": contrastive.detach(),
        "intent/shuffled_rank": shuffled_rank.detach(),
        "intent/diversity_penalty": diversity.detach(),
        "intent/diversity_rms": diversity_rms.detach(),
        "intent/mode_entropy": entropy.detach(),
        "intent/active_modes": winner.unique().numel(),
        "intent/top1_event": top_event.detach(),
        "intent/overflow": overflow.float().mean().detach(),
    }
    metrics.update(event_metrics)
    metrics.update(top_metrics)
    return total, metrics


def belief_dynamics_loss(
    model,
    batch,
    *,
    flow_coef=1.0,
    prior_coef=1.0,
    grounding_coef=0.25,
    history_rank_coef=1.0,
    intent_rank_coef=1.0,
    action_rank_coef=0.0,
    condition_margin=0.05,
    visible_boost=3.0,
    occupied_boost=0.0,
    hidden_occupied_boost=0.0,
    rank_anchors=1,
    action_residual_coef=0.0,
    anchor_coef=0.0,
    anchor_grounding_coef=0.0,
    counterfactual_coef=0.0,
    counterfactual_effect_coef=0.0,
):
    """Train next-state flow from only execution-time information.

    The ordinary random-time flow term learns the vector field.  A separate
    tau=0 branch directly trains genuine noise-to-state generation, and
    cross-episode condition swaps require both causal history and the promoted
    intent latent to improve that generation.  The swaps use only the final
    anchor to keep the intervention cheap relative to the full sequence loss.
    """
    with torch.no_grad():
        condition = model.encode_condition(batch, sample_intent=False)
        registers = condition["history_registers"]
        plan = condition["plan_tokens"]
        action = condition["action_tokens"]
        action_valid = condition["action_valid"]
        anchors = registers.shape[1]
        current_raw = model.full_state_tokenizer.encode(
            batch["state"][:, :anchors],
            batch["globals"][:, :anchors],
        )
        current_target = model.normalize_state(current_raw)
        target_raw = model.full_state_tokenizer.encode(
            batch["state"][:, 1 : anchors + 1],
            batch["globals"][:, 1 : anchors + 1],
        )
        absolute_target = model.normalize_state(target_raw)

    anchor = model.flow.current_anchor(registers)
    residual_flow = model.flow.current_belief_anchor
    target = absolute_target - current_target if residual_flow else absolute_target
    anchor_loss = (
        F.mse_loss(anchor.float(), current_target.float())
        if residual_flow
        else absolute_target.new_zeros(())
    )

    noise = torch.randn_like(target)
    tau = torch.rand(*target.shape[:-2], device=target.device, dtype=target.dtype)
    y_tau = noise.lerp(target, tau[..., None, None])
    target_velocity = target - noise
    velocity = model.flow(
        y_tau, tau, registers, plan, action, action_valid
    )
    flow = F.mse_loss(velocity.float(), target_velocity.float())

    prior_noise = torch.randn_like(target)
    tau_zero = torch.zeros_like(tau)
    prior_velocity = model.flow(
        prior_noise, tau_zero, registers, plan, action, action_valid
    )
    residual_estimate = prior_noise + prior_velocity
    prior_residual_per = (
        residual_estimate.float() - target.float()
    ).square().mean((-1, -2))
    prior = prior_residual_per.mean()
    prior_estimate = residual_estimate + anchor if residual_flow else residual_estimate
    prior_per = (
        prior_estimate.float() - absolute_target.float()
    ).square().mean((-1, -2))

    raw_state = model.denormalize_state(prior_estimate)
    decoded = model.full_state_tokenizer.decode(raw_state)
    next_visibility = batch["local_visibility"][:, 1 : anchors + 1]
    next_visibility = next_visibility.squeeze(-3).flatten(-2).bool()
    target_state = batch["state"][:, 1 : anchors + 1]
    target_occupied = target_state[..., 1].bool()
    cell_weights = (
        1.0
        + float(visible_boost) * next_visibility
        + float(occupied_boost) * target_occupied
        + float(hidden_occupied_boost) * (target_occupied & ~next_visibility)
    )
    grounding, grounding_metrics = structured_reconstruction_loss(
        model.full_state_tokenizer,
        decoded,
        target_state,
        batch["globals"][:, 1 : anchors + 1],
        cell_weights=cell_weights,
    )

    anchor_grounding = prior.new_zeros(())
    anchor_grounding_metrics = {}
    if residual_flow and float(anchor_grounding_coef):
        decoded_anchor = model.full_state_tokenizer.decode(
            model.denormalize_state(anchor)
        )
        anchor_grounding, anchor_grounding_metrics = structured_reconstruction_loss(
            model.full_state_tokenizer,
            decoded_anchor,
            batch["state"][:, :anchors],
            batch["globals"][:, :anchors],
        )

    zero = prior.new_zeros(())
    history_rank = zero
    intent_rank = zero
    action_rank = zero
    history_advantage = zero
    intent_advantage = zero
    action_advantage = zero
    action_output_delta = zero
    if target.shape[0] > 1 and any(
        float(value)
        for value in (history_rank_coef, intent_rank_coef, action_rank_coef)
    ):
        n_rank = min(max(int(rank_anchors), 1), anchors)
        last_noise = prior_noise[:, -n_rank:]
        last_tau = tau_zero[:, -n_rank:]
        last_target = absolute_target[:, -n_rank:]
        last_action = action[:, -n_rank:]
        last_valid = action_valid[:, -n_rank:]
        matched = prior_per[:, -n_rank:]

        def rank_metrics(error, eligible=None):
            advantage = error - matched
            penalty = F.relu(float(condition_margin) - advantage)
            if eligible is not None:
                weight = eligible.to(penalty.dtype)
                denom = weight.sum().clamp_min(1.0)
                return (penalty * weight).sum() / denom, (
                    advantage * weight
                ).sum() / denom
            return penalty.mean(), advantage.mean()

        history_shuffled = model.flow(
            last_noise,
            last_tau,
            registers[:, -n_rank:].roll(1, 0),
            plan[:, -n_rank:],
            last_action,
            last_valid,
        ) + last_noise
        if residual_flow:
            history_shuffled = history_shuffled + model.flow.current_anchor(
                registers[:, -n_rank:].roll(1, 0)
            )
        history_error = (
            history_shuffled.float() - last_target.float()
        ).square().mean((-1, -2))
        history_rank, history_advantage = rank_metrics(history_error)

        intent_shuffled = model.flow(
            last_noise,
            last_tau,
            registers[:, -n_rank:],
            plan[:, -n_rank:].roll(1, 0),
            last_action,
            last_valid,
        ) + last_noise
        if residual_flow:
            intent_shuffled = intent_shuffled + anchor[:, -n_rank:]
        intent_error = (
            intent_shuffled.float() - last_target.float()
        ).square().mean((-1, -2))
        intent_rank, intent_advantage = rank_metrics(intent_error)

        shuffled_action = last_action.roll(1, 0)
        shuffled_valid = last_valid.roll(1, 0)
        action_shuffled = model.flow(
            last_noise,
            last_tau,
            registers[:, -n_rank:],
            plan[:, -n_rank:],
            shuffled_action,
            shuffled_valid,
        ) + last_noise
        if residual_flow:
            action_shuffled = action_shuffled + anchor[:, -n_rank:]
        action_error = (
            action_shuffled.float() - last_target.float()
        ).square().mean((-1, -2))
        action_output_delta = (
            action_shuffled.float() - prior_estimate[:, -n_rank:].float()
        ).square().mean((-1, -2)).mean()
        union = last_valid | shuffled_valid
        token_change = (
            (last_action.float() - shuffled_action.float()).square().mean(-1)
        )
        action_eligible = (
            (last_valid != shuffled_valid).any(-1)
            | ((token_change > 1e-8) & union).any(-1)
        )
        action_rank, action_advantage = rank_metrics(
            action_error, action_eligible
        )

    action_residual_penalty = zero
    if model.flow.explicit_action_residual and float(action_residual_coef):
        n_probe = min(max(int(rank_anchors), 1), anchors)
        residual = model.flow.action_residual(
            prior_noise[:, -n_probe:],
            action[:, -n_probe:],
            action_valid[:, -n_probe:],
        )
        action_residual_penalty = residual.float().square().mean()

    counterfactual = zero
    counterfactual_effect = zero
    counterfactual_effect_cosine = zero
    counterfactual_effect_norm_ratio = zero
    counterfactual_effect_active = zero
    if float(counterfactual_coef) or float(counterfactual_effect_coef):
        required = (
            "counterfactual_action",
            "counterfactual_next_state",
            "counterfactual_next_globals",
            "counterfactual_valid",
        )
        missing = [name for name in required if name not in batch]
        if missing:
            raise ValueError(
                f"counterfactual belief loss is missing batch fields {missing}"
            )
        with torch.no_grad():
            cf_action, _, cf_valid_action, _ = model.intent_model.self_action_tokenizer(
                batch["local_obs"][:, :anchors],
                batch["counterfactual_action"][:, :anchors],
            )
            cf_absolute_target = model.normalize_state(
                model.full_state_tokenizer.encode(
                    batch["counterfactual_next_state"][:, :anchors],
                    batch["counterfactual_next_globals"][:, :anchors],
                )
            )
            cf_target = (
                cf_absolute_target - current_target
                if residual_flow
                else cf_absolute_target
            )
            cf_valid = batch["counterfactual_valid"][:, :anchors].bool()

        cf_y_tau = noise.lerp(cf_target, tau[..., None, None])
        cf_velocity = model.flow(
            cf_y_tau,
            tau,
            registers,
            plan,
            cf_action,
            cf_valid_action,
        )
        cf_flow_per = (
            cf_velocity.float() - (cf_target - noise).float()
        ).square().mean((-1, -2))
        cf_residual_estimate = prior_noise + model.flow(
            prior_noise,
            tau_zero,
            registers,
            plan,
            cf_action,
            cf_valid_action,
        )
        cf_prior_per = (
            cf_residual_estimate.float() - cf_target.float()
        ).square().mean((-1, -2))
        cf_weight = cf_valid.to(cf_flow_per.dtype)
        cf_denom = cf_weight.sum().clamp_min(1.0)
        counterfactual = (
            ((cf_flow_per + cf_prior_per) * cf_weight).sum() / cf_denom
        )
        cf_estimate = (
            cf_residual_estimate + anchor
            if residual_flow
            else cf_residual_estimate
        )
        predicted_effect = cf_estimate.float() - prior_estimate.float()
        target_effect = cf_absolute_target.float() - absolute_target.float()
        pred_flat = predicted_effect.flatten(-2)
        target_flat = target_effect.flatten(-2)
        effect_per = (predicted_effect - target_effect).square().mean((-1, -2))
        target_norm = target_flat.norm(dim=-1)
        effect_valid = cf_valid & (target_norm > 1.0e-4)
        effect_weight = effect_valid.to(effect_per.dtype)
        effect_denom = effect_weight.sum().clamp_min(1.0)
        counterfactual_effect = (
            effect_per * effect_weight
        ).sum() / effect_denom
        cosine = F.cosine_similarity(pred_flat, target_flat, dim=-1, eps=1e-8)
        pred_norm = pred_flat.norm(dim=-1)
        counterfactual_effect_cosine = (
            cosine * effect_weight
        ).sum() / effect_denom
        counterfactual_effect_norm_ratio = (
            (pred_norm / target_norm.clamp_min(1e-8)) * effect_weight
        ).sum() / effect_denom
        counterfactual_effect_active = effect_weight.sum() / cf_denom

    total = (
        float(flow_coef) * flow
        + float(prior_coef) * prior
        + float(grounding_coef) * grounding
        + float(anchor_coef) * anchor_loss
        + float(anchor_grounding_coef) * anchor_grounding
        + float(history_rank_coef) * history_rank
        + float(intent_rank_coef) * intent_rank
        + float(action_rank_coef) * action_rank
        + float(action_residual_coef) * action_residual_penalty
        + float(counterfactual_coef) * counterfactual
        + float(counterfactual_effect_coef) * counterfactual_effect
    )
    with torch.no_grad():
        predicted_state, predicted_globals = model.full_state_tokenizer.discretize(
            decoded
        )
        true_state = batch["state"][:, 1 : anchors + 1].clone()
        true_state[..., 2] = -1
        exact_cell = (predicted_state == true_state).all(-1)
        visible = next_visibility
        hidden = ~visible
        occupied = true_state[..., 1].bool()

        def masked_mean(value, mask):
            return (value & mask).sum().float() / mask.sum().clamp_min(1)

        probabilities = condition["mode_probabilities"]
        entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(-1).mean()
        metrics = {
            "belief_dynamics/total": total.detach(),
            "belief_dynamics/flow": flow.detach(),
            "belief_dynamics/prior_latent": prior.detach(),
            "belief_dynamics/prior_grounding": grounding.detach(),
            "belief_dynamics/current_anchor": anchor_loss.detach(),
            "belief_dynamics/current_anchor_grounding": anchor_grounding.detach(),
            "belief_dynamics/transition_target_rms": target.float().square().mean().sqrt(),
            "belief_dynamics/history_rank": history_rank.detach(),
            "belief_dynamics/intent_rank": intent_rank.detach(),
            "belief_dynamics/action_rank": action_rank.detach(),
            "belief_dynamics/history_advantage": history_advantage.detach(),
            "belief_dynamics/intent_advantage": intent_advantage.detach(),
            "belief_dynamics/action_advantage": action_advantage.detach(),
            "belief_dynamics/action_output_delta": action_output_delta.detach(),
            "belief_dynamics/action_residual_penalty": (
                action_residual_penalty.detach()
            ),
            "belief_dynamics/counterfactual": counterfactual.detach(),
            "belief_dynamics/counterfactual_effect": counterfactual_effect.detach(),
            "belief_dynamics/counterfactual_effect_cosine": (
                counterfactual_effect_cosine.detach()
            ),
            "belief_dynamics/counterfactual_effect_norm_ratio": (
                counterfactual_effect_norm_ratio.detach()
            ),
            "belief_dynamics/counterfactual_effect_active": (
                counterfactual_effect_active.detach()
            ),
            "belief_dynamics/visible_exact_cell": masked_mean(exact_cell, visible),
            "belief_dynamics/hidden_exact_cell": masked_mean(exact_cell, hidden),
            "belief_dynamics/hidden_occupied_exact": masked_mean(
                exact_cell, hidden & occupied
            ),
            "belief_dynamics/exact_globals": (
                predicted_globals == batch["globals"][:, 1 : anchors + 1]
            ).all(-1).float().mean(),
            "belief_dynamics/intent_entropy": entropy.detach(),
            "belief_dynamics/target_rms": absolute_target.float().square().mean().sqrt(),
        }
    metrics.update(
        {f"belief_dynamics/{key}": value for key, value in grounding_metrics.items()}
    )
    metrics.update(
        {
            f"belief_dynamics/anchor_{key}": value
            for key, value in anchor_grounding_metrics.items()
        }
    )
    return total, metrics


def joint_flow_world_model_loss(
    model,
    batch,
    *,
    flow_coef=1.0,
    grounding_coef=0.25,
    opponent_event_coef=0.5,
    future_jepa_coef=0.25,
):
    """Joint conditional flow over hidden state and opponent plan.

    Mechanics are deliberately absent from this objective: sampled opponent
    actions are decoded from plan tokens and passed to the separately frozen
    mechanics model at rollout time. This prevents an intent shortcut into the
    transition function.
    """
    horizon = model.opponent_tokenizer.max_horizon
    anchors = batch["state"].shape[1] - horizon
    if anchors <= 0:
        raise ValueError("dynamics sequence must exceed opponent maximum horizon")
    with torch.no_grad():
        target_state_raw = model.full_state_tokenizer.encode(
            batch["state"][:, :anchors], batch["globals"][:, :anchors]
        )
        target_state = model.normalize_state(target_state_raw)
        target_plan, target_events, target_valid, _ = model.opponent_tokenizer.encode(
            batch["state"], batch["opponent_action"]
        )
        target_plan = model.normalize_plan(target_plan)
    history = model.encode_history(batch)
    registers = history["registers"][:, :anchors]
    target = torch.cat((target_state, target_plan), dim=-2)
    noise = torch.randn_like(target)
    tau = torch.rand(*target.shape[:-2], device=target.device, dtype=target.dtype)
    y_tau = noise.lerp(target, tau[..., None, None])
    target_velocity = target - noise
    velocity = model.flow(y_tau, tau, registers)
    flow = F.mse_loss(velocity.float(), target_velocity.float())
    estimate = y_tau + (1.0 - tau[..., None, None]) * velocity
    predicted_state, predicted_plan = model.split_joint(estimate)

    raw_state = model.denormalize_state(predicted_state)
    decoded = model.full_state_tokenizer.decode(raw_state)
    visibility = batch["local_visibility"][:, :anchors].flatten(-2).squeeze(-2)
    cell_weights = 1.0 + 3.0 * visibility
    grounding, grounding_metrics = structured_reconstruction_loss(
        model.full_state_tokenizer,
        decoded,
        batch["state"][:, :anchors],
        batch["globals"][:, :anchors],
        cell_weights=cell_weights,
    )
    denorm_plan = model.denormalize_plan(predicted_plan)
    plan_logits = model.opponent_tokenizer.decode_events(denorm_plan)
    event_loss, event_metrics = event_reconstruction_loss(
        model.opponent_tokenizer.event_decoder,
        plan_logits,
        target_events,
        target_valid,
        "joint_plan",
    )
    with torch.no_grad():
        future_target = model.normalize_state(
            model.full_state_tokenizer.encode(
                batch["state"][:, horizon : horizon + anchors],
                batch["globals"][:, horizon : horizon + anchors],
            )
        )
    future = _latent_distance(model.future_predictor(registers), future_target)
    total = (
        float(flow_coef) * flow
        + float(grounding_coef) * grounding
        + float(opponent_event_coef) * event_loss
        + float(future_jepa_coef) * future
    )
    metrics = {
        "joint/total": total.detach(),
        "joint/flow": flow.detach(),
        "joint/grounding": grounding.detach(),
        "joint/opponent_event": event_loss.detach(),
        "joint/future_jepa": future.detach(),
        "joint/target_rms": target.float().square().mean().sqrt().detach(),
    }
    metrics.update({f"joint/{key}": value for key, value in grounding_metrics.items()})
    metrics.update(event_metrics)
    return total, metrics
