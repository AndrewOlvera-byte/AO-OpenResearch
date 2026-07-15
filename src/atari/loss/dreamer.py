"""DreamerV4 loss functions for Atari — one per optimized subsystem.

Same structure as the MicroRTS losses, minus the action-mask BCE (Atari has no
action mask) and with a single categorical actor:

- ``world_model_loss`` — tokenizer reconstruction (MSE), latent dynamics
  (teacher-forced next-latent MSE, stop-grad target), reward regression, continue BCE.
- ``shortcut_forcing_loss`` — the Dreamer 4 flow-matching + self-consistency dynamics
  objective on the next latent.
- ``lambda_return`` / ``actor_critic_losses`` — imagined lambda-return critic
  regression and a REINFORCE actor with a normalized advantage + entropy bonus.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from shared.dreamerv4 import symlog, two_hot


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=x.dtype)
    denom = mask.expand_as(x).sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def _twohot_xent(coder, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tgt = two_hot(symlog(target), coder.bins).detach()
    return -(tgt * F.log_softmax(logits, dim=-1)).sum(-1)


def world_model_loss(model, obs, action, reward, cont, is_first=None, *,
                     recon_coef=1.0, dyn_coef=1.0, reward_coef=1.0, cont_coef=1.0,
                     recon_dynamic_coef=0.0, reward_balance_nonzero=False):
    """``obs`` (B,T,C,H,W); ``action`` (B,T) long; ``reward``/``cont`` (B,T).
    Returns (loss, metrics, latents z)."""
    tok = model.tokenizer
    z = tok.encode(obs)
    recon = tok.decode(z)
    recon_err = (recon - obs).pow(2)
    recon_pixel_loss = recon_err.mean()
    recon_loss = recon_pixel_loss
    with torch.no_grad():
        mean_img = obs.mean(dim=(0, 1), keepdim=True)
        recon_mean_baseline = F.mse_loss(mean_img.expand_as(obs), obs)
        if obs.shape[1] > 1:
            delta = torch.zeros_like(obs)
            delta[:, 1:] = (obs[:, 1:] - obs[:, :-1]).abs()
            obs_delta = delta.mean()
        else:
            delta = torch.zeros_like(obs)
            obs_delta = obs.new_tensor(0.0)
    if recon_dynamic_coef > 0 and obs.shape[1] > 1:
        rel_delta = (delta / obs_delta.clamp_min(1e-6)).clamp(max=20.0)
        weight = 1.0 + float(recon_dynamic_coef) * rel_delta
        recon_loss = (recon_err * weight).sum() / weight.sum().clamp_min(1.0)

    pred = model.world_model.predict(z, action)
    z_prev = z[:, :-1].detach()
    z_next = z[:, 1:].detach()
    dyn_err = (pred["next_latent"][:, :-1] - z_next).pow(2)
    dyn_zero_err = z_next.pow(2)
    dyn_copy_err = (z_prev - z_next).pow(2)
    if is_first is None:
        dyn_loss = dyn_err.mean()
        dyn_zero_baseline = dyn_zero_err.mean()
        dyn_copy_baseline = dyn_copy_err.mean()
        valid_next = None
    else:
        valid_next = ~is_first[:, 1:].bool()
        dyn_loss = _masked_mean(dyn_err, valid_next)
        dyn_zero_baseline = _masked_mean(dyn_zero_err, valid_next)
        dyn_copy_baseline = _masked_mean(dyn_copy_err, valid_next)
    reward_xent = _twohot_xent(model.world_model.reward_coder, pred["reward_logits"], reward)
    reward_raw_loss = reward_xent.mean()
    with torch.no_grad():
        reward_nonzero = reward.ne(0)
        reward_zero = ~reward_nonzero
    reward_loss_nonzero = reward_xent[reward_nonzero].mean() if bool(reward_nonzero.any()) else reward_xent.new_tensor(0.0)
    reward_loss_zero = reward_xent[reward_zero].mean() if bool(reward_zero.any()) else reward_xent.new_tensor(0.0)
    if reward_balance_nonzero and bool(reward_nonzero.any()) and bool(reward_zero.any()):
        reward_loss = 0.5 * (reward_loss_zero + reward_loss_nonzero)
    else:
        reward_loss = reward_raw_loss
    cont_loss = F.binary_cross_entropy_with_logits(pred["continue_logit"], cont.float())

    loss = (recon_coef * recon_loss + dyn_coef * dyn_loss
            + reward_coef * reward_loss + cont_coef * cont_loss)
    metrics = {
        "wm/recon": float(recon_loss.detach()),
        "wm/recon_pixel": float(recon_pixel_loss.detach()),
        "wm/recon_mean_baseline": float(recon_mean_baseline),
        "wm/obs_delta_abs": float(obs_delta),
        "wm/latent_std": float(z.detach().std()),
        "wm/latent_delta_abs": float((z_next - z_prev).abs().mean()),
        "wm/dynamics": float(dyn_loss.detach()),
        "wm/dynamics_zero_baseline": float(dyn_zero_baseline.detach()),
        "wm/dynamics_copy_baseline": float(dyn_copy_baseline.detach()),
        "wm/dynamics_vs_copy": float((dyn_loss / dyn_copy_baseline.clamp_min(1e-8)).detach()),
        "wm/reward": float(reward_loss.detach()),
        "wm/reward_raw": float(reward_raw_loss.detach()),
        "wm/reward_zero": float(reward_loss_zero.detach()),
        "wm/reward_nonzero": float(reward_loss_nonzero.detach()),
        "wm/reward_nonzero_frac": float(reward_nonzero.float().mean()),
        "wm/reward_pred_mean": float(pred["reward"].detach().mean()),
        "wm/reward_pred_std": float(pred["reward"].detach().std()),
        "wm/reward_target_mean": float(reward.detach().mean()),
        "wm/continue": float(cont_loss.detach()),
        "wm/dynamics_valid_frac": 1.0 if valid_next is None else float(valid_next.float().mean()),
        "wm/total": float(loss.detach()),
    }
    return loss, metrics, z


def shortcut_forcing_loss(model, z, action, is_first=None, *, consistency_frac=0.25, k_max=None):
    """Dreamer 4 shortcut-forcing (flow-matching + self-consistency) on next latents."""
    wm = model.world_model
    k_max = k_max or wm.cfg.k_max
    z = z.detach()
    h = wm.contextualize(z, action)["h"][:, :-1]
    z1 = z[:, 1:].detach()
    B, T = z1.shape[:2]
    valid = None if is_first is None else ~is_first[:, 1:].bool()
    noise = torch.randn_like(z1)

    tau = torch.rand(B, T, device=z.device)
    z_tau = (1 - tau)[..., None, None] * noise + tau[..., None, None] * z1
    d0 = torch.zeros(B, T, device=z.device)
    fm_err = (wm.flow_velocity(h, z_tau, tau, d0) - (z1 - noise)).pow(2)
    fm_loss = fm_err.mean() if valid is None else _masked_mean(fm_err, valid)

    log2k = int(max(1, torch.log2(torch.tensor(float(k_max))).item()))
    j = torch.randint(0, log2k, (B, T), device=z.device)
    d = 2.0 ** (-(j + 1).float())
    tau2 = torch.rand(B, T, device=z.device) * (1 - 2 * d)
    z_t2 = (1 - tau2)[..., None, None] * noise + tau2[..., None, None] * z1
    with torch.no_grad():
        s1 = wm.flow_velocity(h, z_t2, tau2, d)
        s2 = wm.flow_velocity(h, z_t2 + d[..., None, None] * s1, tau2 + d, d)
        target = 0.5 * (s1 + s2)
    sc_err = (wm.flow_velocity(h, z_t2, tau2, 2 * d) - target).pow(2)
    sc_loss = sc_err.mean() if valid is None else _masked_mean(sc_err, valid)

    loss = fm_loss + consistency_frac * sc_loss
    return loss, {"flow/matching": float(fm_loss.detach()),
                  "flow/consistency": float(sc_loss.detach()),
                  "flow/valid_frac": 1.0 if valid is None else float(valid.float().mean()),
                  "flow/total": float(loss.detach())}


class PercentileReturnNormalizer:
    """DreamerV3-style return scale for actor advantages.

    Sparse rewards make per-batch advantage standardization brittle: when imagined
    returns are mostly identical, a tiny batch standard deviation can amplify noise.
    The percentile spread is tracked with an EMA and clamped by ``limit`` so actor
    updates stay on a stable scale even before nonzero rewards are common.
    """

    def __init__(self, *, rate: float = 0.01, limit: float = 1.0,
                 perclo: float = 5.0, perchi: float = 95.0, eps: float = 1e-8) -> None:
        self.rate = float(rate)
        self.limit = float(limit)
        self.perclo = float(perclo)
        self.perchi = float(perchi)
        self.eps = float(eps)
        self.low = 0.0
        self.high = 0.0
        self.initialized = False

    @torch.no_grad()
    def scale(self, returns: torch.Tensor, *, update: bool = True) -> tuple[torch.Tensor, dict]:
        flat = returns.detach().flatten().float()
        if flat.numel() == 0:
            scale = returns.new_tensor(self.limit)
            return scale, {"ac/return_norm_scale": float(scale)}
        lo = float(torch.quantile(flat, self.perclo / 100.0))
        hi = float(torch.quantile(flat, self.perchi / 100.0))
        if update:
            if not self.initialized:
                self.low, self.high = lo, hi
                self.initialized = True
            else:
                self.low = (1.0 - self.rate) * self.low + self.rate * lo
                self.high = (1.0 - self.rate) * self.high + self.rate * hi
        use_lo, use_hi = (self.low, self.high) if self.initialized else (lo, hi)
        spread = max(use_hi - use_lo, self.limit, self.eps)
        scale = returns.new_tensor(spread)
        return scale, {
            "ac/return_norm_low": use_lo,
            "ac/return_norm_high": use_hi,
            "ac/return_norm_scale": spread,
        }


def lambda_return(reward, value, disc, lam=0.95):
    """R_t = r_t + disc_t * ((1-lam) V_{t+1} + lam R_{t+1}); bootstrap on V_H."""
    H = reward.shape[1]
    returns = torch.zeros_like(reward)
    nxt = value[:, -1]
    for t in reversed(range(H)):
        returns[:, t] = reward[:, t] + disc[:, t] * ((1 - lam) * value[:, t + 1] + lam * nxt)
        nxt = returns[:, t]
    return returns


def actor_critic_losses(model, imagined, *, gamma=0.997, lam=0.95, entropy_coef=3e-4,
                        normalize_adv=True, return_normalizer: PercentileReturnNormalizer | None = None):
    """Imagined lambda-return critic + normalized-advantage REINFORCE actor."""
    ae = model.action_expert
    z = imagined["z"]
    B, Hp1 = z.shape[:2]
    H = Hp1 - 1

    z_flat = z.reshape(B * Hp1, *z.shape[2:])
    with torch.no_grad():
        values = ae.value(z_flat).reshape(B, Hp1)              # scalar means (advantage/metrics)
        tgt_values = ae.target_value(z_flat).reshape(B, Hp1)

    disc = gamma * imagined["cont"]
    returns = lambda_return(imagined["reward"], tgt_values, disc, lam=lam)
    # Critic: symlog two-hot classification of the lambda-return on the first H latents.
    value_logits = ae.value_logits(z[:, :H].reshape(B * H, *z.shape[2:]))
    critic_loss = ae.critic_coder.loss(value_logits, returns.detach().reshape(B * H))

    zs = z[:, :H].reshape(B * H, *z.shape[2:])
    acts = imagined["action"].reshape(B * H)
    logprob, entropy, _ = ae.evaluate(zs, acts)
    logprob = logprob.reshape(B, H)
    entropy = entropy.reshape(B, H)
    advantage = (returns - values[:, :H]).detach()
    norm_metrics = {}
    if return_normalizer is not None:
        scale, norm_metrics = return_normalizer.scale(returns.detach(), update=True)
        advantage = advantage / scale.clamp_min(1e-8)
    elif normalize_adv:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    actor_loss = -(logprob * advantage).mean() - entropy_coef * entropy.mean()

    metrics = {
        "ac/critic_loss": float(critic_loss.detach()),
        "ac/actor_loss": float(actor_loss.detach()),
        "ac/return_mean": float(returns.mean().detach()),
        "ac/return_std": float(returns.std().detach()),
        "ac/adv_mean": float(advantage.mean().detach()),
        "ac/adv_std": float(advantage.std().detach()),
        "ac/value_mean": float(values.mean().detach()),
        "ac/entropy": float(entropy.mean().detach()),
    }
    metrics.update(norm_metrics)
    return actor_loss, critic_loss, metrics
