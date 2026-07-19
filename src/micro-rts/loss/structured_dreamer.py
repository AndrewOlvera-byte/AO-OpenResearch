"""Online world-model, agent-head, and imagination losses for structured Dreamer."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.dreamer_v2.dynamics import structured_causal_paired_loss
from loss.dreamer import lambda_return


def _flatten_valid(x, valid):
    return x.reshape(-1, *x.shape[2:])[valid.reshape(-1)]


def sequence_transition_batch(batch: dict) -> dict:
    """Convert online replay sequences into factual one-step transition rows."""
    state, glob = batch["full_state"], batch["full_globals"]
    if state.shape[1] < 2:
        raise ValueError("structured online dynamics needs replay seq_len >= 2")
    valid = ~batch["is_first"][:, 1:].bool()
    if not valid.any():
        raise ValueError("sample contains no within-episode transition")
    out = {
        "state": _flatten_valid(state[:, :-1], valid),
        "globals": _flatten_valid(glob[:, :-1], valid),
        "next_state": _flatten_valid(state[:, 1:], valid),
        "next_globals": _flatten_valid(glob[:, 1:], valid),
        "action": _flatten_valid(batch["action"][:, :-1], valid),
        "opponent_action": _flatten_valid(batch["opponent_action"][:, :-1], valid),
    }
    n = out["state"].shape[0]
    out.update({
        "counterfactual_action": out["action"].clone(),
        "counterfactual_opponent_action": out["opponent_action"].clone(),
        "counterfactual_next_state": out["next_state"].clone(),
        "counterfactual_next_globals": out["next_globals"].clone(),
        "counterfactual_valid": torch.zeros(n, dtype=torch.bool, device=state.device),
    })
    return out


def structured_online_world_loss(model, batch: dict, **coefficients):
    transitions = sequence_transition_batch(batch)
    return structured_causal_paired_loss(model, transitions, **coefficients)


def _head_rows(model, batch):
    """Return departure/arrival latents and aligned scalar/action labels."""
    if "full_state" in batch:
        state, glob = batch["full_state"], batch["full_globals"]
        valid = ~batch["is_first"][:, 1:].bool()
        with torch.no_grad():
            z0 = model.tokenizer.encode(state[:, :-1], glob[:, :-1])
            z1 = model.tokenizer.encode(state[:, 1:], glob[:, 1:])
        return {
            "state": _flatten_valid(state[:, :-1], valid),
            "z0": _flatten_valid(z0, valid),
            "z1": _flatten_valid(z1, valid),
            "action": _flatten_valid(batch["action"][:, :-1], valid),
            "opponent_action": _flatten_valid(batch["opponent_action"][:, :-1], valid),
            "opponent_valid": _flatten_valid(
                batch.get("opponent_valid", torch.ones_like(batch["is_first"]))[:, :-1], valid
            ).bool(),
            "mask": _flatten_valid(batch["mask"][:, :-1], valid).float(),
            "reward": _flatten_valid(batch["reward"][:, :-1, None], valid).squeeze(-1),
            "cont": _flatten_valid(batch["cont"][:, :-1, None], valid).squeeze(-1),
        }

    state, glob = batch["state"], batch["globals"]
    nxt, nglob = batch["next_state"], batch["next_globals"]
    with torch.no_grad():
        z0 = model.tokenizer.encode(state, glob).flatten(0, 1)
        z1 = model.tokenizer.encode(nxt, nglob).flatten(0, 1)
    n = z0.shape[0]
    return {
        "state": state.flatten(0, 1), "z0": z0, "z1": z1,
        "action": batch["action"].flatten(0, 1),
        "opponent_action": batch["opponent_action"].flatten(0, 1),
        "opponent_valid": torch.ones(n, dtype=torch.bool, device=z0.device),
        "mask": batch["mask"].flatten(0, 1).float(),
        "reward": batch["reward"].flatten(), "cont": batch["cont"].flatten(),
    }


def structured_agent_head_loss(
    model,
    batch: dict,
    *,
    reward_coef=1.0,
    continue_coef=1.0,
    prior_bc_coef=1.0,
    opponent_bc_coef=0.5,
):
    """Phase-2 Dreamer heads: reward/continue and own/opponent behavior priors."""
    rows = _head_rows(model, batch)
    reward_logits, continue_logits = model.scalar_heads(rows["z1"])
    reward_loss = model.scalar_heads.reward_coder.loss(reward_logits, rows["reward"])
    continue_loss = F.binary_cross_entropy_with_logits(continue_logits, rows["cont"])

    prior_logp, prior_entropy, _ = model.action_expert.behavior_prior.evaluate(
        rows["z0"], rows["action"], rows["mask"]
    )
    prior_bc = -prior_logp.mean()
    opponent_action, opp_mask = model.role_bc_inputs(
        rows["state"], rows["opponent_action"], 2
    )
    opp_logp, opp_entropy, _ = model.action_expert.opponent_policy.evaluate(
        rows["z0"], opponent_action, opp_mask
    )
    opp_valid = rows["opponent_valid"].float()
    opponent_bc = -(opp_logp * opp_valid).sum() / opp_valid.sum().clamp_min(1.0)
    total = (
        float(reward_coef) * reward_loss
        + float(continue_coef) * continue_loss
        + float(prior_bc_coef) * prior_bc
        + float(opponent_bc_coef) * opponent_bc
    )
    with torch.no_grad():
        pred_reward = model.scalar_heads.reward_coder.mean(reward_logits)
        reward_mae = (pred_reward - rows["reward"]).abs().mean()
        cont_acc = ((continue_logits > 0) == rows["cont"].bool()).float().mean()
    return total, {
        "agent/total": total.detach(),
        "agent/reward": reward_loss.detach(),
        "agent/reward_mae": reward_mae.detach(),
        "agent/continue": continue_loss.detach(),
        "agent/continue_acc": cont_acc.detach(),
        "agent/prior_bc": prior_bc.detach(),
        "agent/prior_entropy": prior_entropy.mean().detach(),
        "agent/opponent_bc": opponent_bc.detach(),
        "agent/opponent_entropy": opp_entropy.mean().detach(),
    }


def structured_pmpo_losses(model, imagined: dict):
    """Dreamer-4 sign-balanced PMPO actor and two-hot lambda-return critic."""
    cfg = model.cfg.actor_critic
    z = imagined["z"]
    b, hp1 = z.shape[:2]
    horizon = hp1 - 1
    flat = z.reshape(b * hp1, *z.shape[2:])
    with torch.no_grad():
        values = model.action_expert.value(flat).reshape(b, hp1)
        targets = model.action_expert.target_value(flat).reshape(b, hp1)
        returns = lambda_return(
            imagined["reward"], targets, cfg.gamma * imagined["cont"], cfg.lam
        )
        advantage = returns - values[:, :horizon]

    z_actor = z[:, :horizon].reshape(b * horizon, *z.shape[2:])
    action = imagined["action"].reshape(b * horizon, *imagined["action"].shape[2:])
    mask = imagined["mask"].reshape(b * horizon, *imagined["mask"].shape[2:])
    logp, entropy, _ = model.action_expert.actor_policy.evaluate(z_actor, action, mask)
    adv = advantage.reshape(-1)
    pos, neg = adv >= 0, adv < 0
    positive = -logp[pos].mean() if pos.any() else logp.new_zeros(())
    negative = logp[neg].mean() if neg.any() else logp.new_zeros(())
    alpha = float(cfg.pmpo_alpha)
    pmpo = alpha * positive + (1.0 - alpha) * negative
    reverse_kl = model.action_expert.actor_policy.reverse_kl(
        z_actor, model.action_expert.behavior_prior, mask
    ).mean()
    actor_loss = (
        pmpo + float(cfg.prior_kl_coef) * reverse_kl
        - float(cfg.entropy_coef) * entropy.mean()
    )

    value_logits = model.action_expert.value_logits(z_actor)
    critic_loss = model.action_expert.coder.loss(value_logits, returns.reshape(-1))
    return actor_loss, critic_loss, {
        "ac/actor_loss": actor_loss.detach(),
        "ac/pmpo": pmpo.detach(),
        "ac/pmpo_positive": positive.detach(),
        "ac/pmpo_negative": negative.detach(),
        "ac/prior_kl": reverse_kl.detach(),
        "ac/critic_loss": critic_loss.detach(),
        "ac/return_mean": returns.mean().detach(),
        "ac/value_mean": values.mean().detach(),
        "ac/advantage_mean": advantage.mean().detach(),
        "ac/advantage_positive_frac": pos.float().mean().detach(),
        "ac/entropy": entropy.mean().detach(),
    }


@torch.no_grad()
def structured_world_metrics(model, batch: dict, flow_steps=1):
    """Deterministic factual metrics for online/fixed guardrails."""
    if "full_state" in batch:
        batch = sequence_transition_batch(batch)
    state = batch["state"].reshape(-1, *batch["state"].shape[-2:])
    glob = batch["globals"].reshape(-1, batch["globals"].shape[-1])
    nxt = batch["next_state"].reshape(-1, *batch["next_state"].shape[-2:])
    nglob = batch["next_globals"].reshape(-1, batch["next_globals"].shape[-1])
    action = batch["action"].reshape(-1, *batch["action"].shape[-2:])
    opp = batch["opponent_action"].reshape(-1, *batch["opponent_action"].shape[-2:])
    z0 = model.tokenizer.encode(state, glob)
    z1 = model.tokenizer.encode(nxt, nglob)
    events, valid, _ = model.action_events(state, action, opp)
    pred = model.dynamics.sample_next(
        z0, events, valid, flow_steps,
        state_token_valid=model.state_token_valid(state),
    )
    norm = model.dynamics.normalize
    mse = F.mse_loss(norm(pred), norm(z1))
    copy = F.mse_loss(norm(z0), norm(z1))
    decoded_state, _ = model.tokenizer.discretize(model.tokenizer.decode(pred))
    true_state = nxt.clone()
    true_state[..., 2] = -1
    exact = (decoded_state == true_state).all(-1).float().mean()
    changed = (state != true_state).any(-1)
    pred_changed = (state != decoded_state).any(-1)
    tp = (changed & pred_changed).sum().float()
    fp = (~changed & pred_changed).sum().float()
    fn = (changed & ~pred_changed).sum().float()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    return {
        "guard/unweighted_mse": float(mse),
        "guard/copy_mse": float(copy),
        "guard/copy_ratio": float(mse / copy.clamp_min(1e-8)),
        "guard/exact_cell": float(exact),
        "guard/changed_f1": float(f1),
    }
