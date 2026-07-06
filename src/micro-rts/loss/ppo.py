"""Clipped PPO surrogate loss over a flattened minibatch TensorDict.

Best-practice PPO objective (Schulman et al. 2017, with the implementation details
from the "PPO that matters" line of work):

- advantages normalized per minibatch,
- clipped policy surrogate,
- optional clipped value loss,
- entropy bonus,
- diagnostic metrics (approx KL, clip fraction) for logging.

Returns ``(loss, metrics)`` where ``loss`` is the scalar to backprop and
``metrics`` is a dict of detached floats for W&B.
"""

from __future__ import annotations

import torch


def ppo_loss(
    mb,
    policy,
    clip: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    clip_vloss: bool = True,
):
    new_logp, entropy, value = policy.evaluate_actions(
        mb["obs"], mb["action"], mb.get("mask", None)
    )

    adv = mb["advantage"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    log_ratio = new_logp - mb["logprob"]
    ratio = log_ratio.exp()

    # Policy (clipped surrogate) loss.
    pg_unclipped = ratio * adv
    pg_clipped = ratio.clamp(1 - clip, 1 + clip) * adv
    policy_loss = -torch.min(pg_unclipped, pg_clipped).mean()

    # Value loss, optionally clipped around the old value estimate.
    returns = mb["return"]
    if clip_vloss:
        old_value = mb["value"]
        v_unclipped = (value - returns).pow(2)
        v_clipped = (old_value + (value - old_value).clamp(-clip, clip) - returns).pow(2)
        value_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
    else:
        value_loss = 0.5 * (value - returns).pow(2).mean()

    entropy_mean = entropy.mean()
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean

    with torch.no_grad():
        # http://joschu.net/blog/kl-approx.html — low-variance KL estimator.
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clipfrac = ((ratio - 1).abs() > clip).float().mean()

    metrics = {
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy_mean.detach()),
        "approx_kl": float(approx_kl),
        "clipfrac": float(clipfrac),
        "total_loss": float(loss.detach()),
    }
    return loss, metrics
