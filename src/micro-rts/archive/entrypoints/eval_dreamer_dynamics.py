"""Entrypoint: RL-readiness eval of a trained DreamerV4 **dynamics** world model.

The dynamics training loop's val probe answers "is the loss going down"; this
script answers the decision question: **is the world model good enough to train
an actor/critic inside it** (`train_dreamer_rl`)? Each section targets one
failure mode that would sink the RL phase, and the script ends with a mechanical
PASS/WARN/FAIL verdict per criterion plus a JSON dump of everything.

1. **Scalar heads, teacher-forced** — decoded reward vs true reward along real
   latents (correlation overall / on nonzero-reward slots, hallucinated reward on
   zero slots) and continue-head probabilities. The reward head is the ONLY
   learning signal the actor gets; if it can't rank real transitions, imagination
   returns are noise.
2. **Terminal transitions** — windows around done rows (terminal-frame stores
   splice the true arrival obs in). Continue-head separation (AUC, recall at 0.5)
   and terminal reward prediction split by win/loss. If p(cont) stays ~1 at
   terminals, imagined episodes never end and value bootstrapping is wrong.
3. **Open-loop rollouts, real actions** — generate `--horizon` frames from
   `--context` real frames at 1/4/16 flow steps, best-of-N samples (the opponent
   is unobserved -> the future is stochastic, a single sample has an MSE floor).
   Per-step decoded-obs accuracy (all cells + occupied-only) against BOTH
   copy-last baselines: frozen last context frame (fair open-loop reference) and
   teacher-forced previous real frame (the strong probe from training). Also the
   reward-head correlation along generated latents.
4. **Action conditioning** — same rollout with actions rolled across the batch.
   If MSE doesn't get worse, the model ignores actions and policy improvement in
   imagination is impossible. This is the hardest go/no-go signal.
5. **Imagination health, RL settings** — `DreamerV4.imagine` exactly as the RL
   loop calls it (predicted masks, actor sampling, `imagine_flow_steps`): latent
   RMS drift (blow-up/collapse), decoded-board occupancy + owner/unit-type
   contradiction rate vs the real-data baseline, reward/continue statistics, and
   predicted-mask sanity.
6. **Renders** — (ground truth | open-loop dream) unit-type/owner strips plus an
   imagination strip, for eyeballing what the numbers mean.

Usage (inside the container)::

    python src/micro-rts/entrypoints/eval_dreamer_dynamics.py \
        --ckpt checkpoints/dreamer_dynamics_v2.pt \
        --data '/data/micro-rts/tokdyn_pretrain_v2__*.h5' \
        --out-dir checkpoints/dynamics_eval_v2

Defaults mirror the RL config: context 8 (`imagine_context`), horizon 15
(`imagine_horizon`), 4 flow steps (`imagine_flow_steps`). The val split
(`--val-frac`, seed 0) matches the training run, so nothing here was trained on.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PKG = _HERE.parents[3]          # src/micro-rts
_SRC = _HERE.parents[3]          # src
for p in (str(_PKG), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from core.registry import build  # noqa: E402
import models.dreamer  # noqa: E402,F401  (registry side effect)
from collectors.offline_data import build_mrts_loader, to_device  # noqa: E402
from collectors.offline_data.mrts_dataset import MRTSSequenceDataset  # noqa: E402
from loss.dreamer import _valid_transitions, mask_actions_to_sources  # noqa: E402


def prep_batch(b, model, device):
    """to_device + the same junk-action masking the model was trained with
    (checkpoints carry ``dynamics.mask_junk_actions`` in their model_cfg), so
    the eval feeds the WM the action distribution it actually saw."""
    b = to_device(b, device)
    if getattr(model.cfg.dynamics, "mask_junk_actions", False):
        b["action"] = mask_actions_to_sources(b["action"], b["obs"])
    return b


from trainers.BaseTrainer import resolve_device  # noqa: E402

# gym_microrts 27-plane GridNet layout (see eval_dreamer_tokenizer.py).
CHANNEL_GROUPS = {
    "hp": (0, 5),
    "resources": (5, 10),
    "owner": (10, 13),
    "unit_type": (13, 21),
    "action": (21, 27),
}
UNIT_LO, UNIT_HI = CHANNEL_GROUPS["unit_type"]
OWNER_LO, OWNER_HI = CHANNEL_GROUPS["owner"]
RESOURCE_UNIT_IDX = 1            # unit_type argmax 1 == neutral resource (owner none)

_PALETTE = np.array([
    (30, 30, 30), (220, 50, 47), (38, 139, 210), (133, 153, 0), (211, 54, 130),
    (181, 137, 0), (108, 113, 196), (42, 161, 152), (203, 75, 22),
], dtype=np.uint8)


def resolve_data(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f"no dataset matches {pattern!r}")
    return matches[-1]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    obs_shape = tuple(ckpt["obs_shape"])
    model_cfg = ckpt["model_cfg"]
    model = build("model", type=model_cfg.get("type", "dreamerv4"),
                  obs_shape=obs_shape, action_nvec=ckpt["action_nvec"],
                  device=str(device), **{k: v for k, v in model_cfg.items() if k != "type"})
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, obs_shape, ckpt.get("step"), ckpt.get("phase")


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    if a.numel() < 2 or a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rank_auc(scores_neg: torch.Tensor, scores_pos: torch.Tensor,
             max_pos: int = 20000) -> float:
    """P(score_neg < score_pos): how reliably the negative class (terminals)
    scores BELOW the positive class (non-terminals). 1.0 = perfect separation."""
    neg = scores_neg.flatten().float()
    pos = scores_pos.flatten().float()
    if neg.numel() == 0 or pos.numel() == 0:
        return float("nan")
    if pos.numel() > max_pos:
        pos = pos[torch.randperm(pos.numel())[:max_pos]]
    # U statistic via ranks of the concatenated sample.
    all_s = torch.cat([neg, pos])
    ranks = all_s.argsort().argsort().float() + 1.0
    r_neg = ranks[: neg.numel()].sum()
    n, m = neg.numel(), pos.numel()
    u_neg = r_neg - n * (n + 1) / 2.0        # pairs where neg > pos (plus ties/2)
    return float(1.0 - u_neg / (n * m))


def occupancy(obs: torch.Tensor) -> torch.Tensor:
    """(...,C,H,W) -> (...,H,W) bool: a unit (incl. resources) sits on the cell."""
    return obs[..., UNIT_LO:UNIT_HI, :, :].argmax(dim=-3) != 0


def contradiction_rate(obs: torch.Tensor) -> float:
    """Owner/unit-type coherence of a decoded board: a cell is contradictory when
    unit_type says empty but owner says owned, or a non-resource unit has no
    owner. ~0 on real boards; rises as imagined boards drift off-manifold."""
    unit = obs[..., UNIT_LO:UNIT_HI, :, :].argmax(dim=-3)
    owner = obs[..., OWNER_LO:OWNER_HI, :, :].argmax(dim=-3)
    bad = ((unit == 0) & (owner != 0)) | \
          ((unit > RESOURCE_UNIT_IDX) & (owner == 0))
    return float(bad.float().mean())


def obs_accuracy(pred_obs, true_obs, cells: torch.Tensor | None = None):
    """Binary-plane accuracy per step: (all_cells, selected_cells), each (B,T).
    ``cells`` (B,T,H,W) restricts the second number (default: occupied cells)."""
    match = (pred_obs > 0.5) == (true_obs > 0.5)                  # (B,T,C,H,W)
    acc_all = match.float().mean(dim=(2, 3, 4))
    if cells is None:
        cells = occupancy(true_obs)                               # (B,T,H,W)
    sel = cells[:, :, None].expand_as(match)
    num = (match & sel).float().sum(dim=(2, 3, 4))
    den = sel.float().sum(dim=(2, 3, 4)).clamp_min(1.0)
    return acc_all, num / den


# --- 1. scalar heads, teacher-forced ---------------------------------------
@torch.no_grad()
def eval_scalar_heads(model, loader, device, batches):
    wm = model.world_model
    pred_r, true_r, cont_p, cont_t = [], [], [], []
    for i, b in zip(range(batches), loader):
        b = prep_batch(b, model, device)
        z = model.tokenizer.encode(b["obs"])
        ctx = wm.contextualize(z, b["action"], b["is_first"],
                               opponent_action=b["opponent_action"])
        valid = _valid_transitions(b["is_first"])
        pred_r.append(ctx["reward"][:, 1:][valid].cpu())
        true_r.append(b["reward"][:, :-1][valid].cpu())
        cont_p.append(torch.sigmoid(ctx["continue_logit"][:, 1:][valid]).cpu())
        cont_t.append(b["cont"][:, :-1][valid].cpu())
    pred_r, true_r = torch.cat(pred_r), torch.cat(true_r)
    cont_p, cont_t = torch.cat(cont_p), torch.cat(cont_t)

    nz = true_r.abs() > 1e-6
    m = {
        "reward/corr_all": corr(pred_r, true_r),
        "reward/corr_nonzero": corr(pred_r[nz], true_r[nz]) if nz.any() else float("nan"),
        "reward/mse": float((pred_r - true_r).pow(2).mean()),
        "reward/nonzero_frac": float(nz.float().mean()),
        "reward/abs_pred_on_zero_slots": float(pred_r[~nz].abs().mean()),
        "reward/true_abs_mean_nonzero": float(true_r[nz].abs().mean()) if nz.any() else 0.0,
        "cont/p_mean_nonterminal": float(cont_p[cont_t > 0.5].mean()),
        "cont/n_slots": int(cont_t.numel()),
    }
    return m, (cont_p, cont_t)


# --- 2. terminal transitions ------------------------------------------------
def terminal_window_indices(ds: MRTSSequenceDataset, limit: int, seed: int = 0):
    """Val-split window indices whose slice contains a done row early enough that
    the (spliced) terminal arrival slot carries a supervisable cont=0 target."""
    import h5py

    with h5py.File(ds.path, "r", locking=False) as f:
        done_rows = np.nonzero(f["done"][:].astype(bool))[0]
    ws = ds._win_start
    lo = np.searchsorted(done_rows, ws)                 # first done >= start
    hi = np.searchsorted(done_rows, ws + ds.seq_len - 1)  # first done >= start+T-1
    idx = np.nonzero(hi > lo)[0]
    if len(idx) > limit:
        idx = np.random.default_rng(seed).choice(idx, size=limit, replace=False)
    return idx


@torch.no_grad()
def eval_terminals(model, ds, idx, device, batch, cont_bg):
    """Continue-head separation + terminal reward prediction on done-containing
    windows. ``cont_bg`` are (p, target) from section 1, pooled into the AUC."""
    wm = model.world_model
    p_term, p_nonterm, pred_r_term, true_r_term = [], [cont_bg[0][cont_bg[1] > 0.5]], [], []
    for i in range(0, len(idx), batch):
        b = default_collate([ds[int(j)] for j in idx[i:i + batch]])
        b = prep_batch(b, model, device)
        z = model.tokenizer.encode(b["obs"])
        ctx = wm.contextualize(z, b["action"], b["is_first"],
                               opponent_action=b["opponent_action"])
        valid = _valid_transitions(b["is_first"])
        p = torch.sigmoid(ctx["continue_logit"][:, 1:])[valid].cpu()
        r = ctx["reward"][:, 1:][valid].cpu()
        t_cont = b["cont"][:, :-1][valid].cpu()
        t_rew = b["reward"][:, :-1][valid].cpu()
        term = t_cont < 0.5
        p_term.append(p[term])
        p_nonterm.append(p[~term])
        pred_r_term.append(r[term])
        true_r_term.append(t_rew[term])
    p_term = torch.cat(p_term)
    p_nonterm = torch.cat(p_nonterm)
    pred_r_term = torch.cat(pred_r_term)
    true_r_term = torch.cat(true_r_term)
    if p_term.numel() == 0:
        return {"cont/n_terminals": 0}
    win, loss = true_r_term > 0, true_r_term < 0
    m = {
        "cont/n_terminals": int(p_term.numel()),
        "cont/p_mean_terminal": float(p_term.mean()),
        "cont/terminal_recall_at_0.5": float((p_term < 0.5).float().mean()),
        "cont/false_terminal_rate_at_0.5": float((p_nonterm < 0.5).float().mean()),
        "cont/auc": rank_auc(p_term, p_nonterm),
        "reward/terminal_pred_mean_win": float(pred_r_term[win].mean()) if win.any() else float("nan"),
        "reward/terminal_true_mean_win": float(true_r_term[win].mean()) if win.any() else float("nan"),
        "reward/terminal_pred_mean_loss": float(pred_r_term[loss].mean()) if loss.any() else float("nan"),
        "reward/terminal_true_mean_loss": float(true_r_term[loss].mean()) if loss.any() else float("nan"),
        "reward/terminal_sign_acc": float(
            (torch.sign(pred_r_term)[win | loss] == torch.sign(true_r_term)[win | loss])
            .float().mean()) if (win | loss).any() else float("nan"),
    }
    return m


# --- 3+4. open-loop rollouts + action conditioning --------------------------
@torch.no_grad()
def eval_open_loop(model, loader, device, batches, context, samples,
                   flow_steps_list, rl_flow_steps):
    wm = model.world_model
    agg: dict[str, list] = {}
    render_pack = None

    def add(key, val):
        agg.setdefault(key, []).append(val)

    for bi, b in zip(range(batches), loader):
        b = prep_batch(b, model, device)
        obs, action, is_first = b["obs"], b["action"], b["is_first"]
        opp_action = b["opponent_action"]
        B, T = obs.shape[:2]
        z = model.tokenizer.encode(obs)
        Hgen = T - context
        z_tgt = wm.normalize(z[:, context:])
        obs_tgt = obs[:, context:]

        # Copy-last baselines, latent + obs space, per step (B,Hgen).
        cl_frozen = wm.normalize(z[:, context - 1:context]).expand_as(z_tgt)
        cl_tf = wm.normalize(z[:, context - 1:-1])
        add("latent/copylast_frozen_mse", (cl_frozen - z_tgt).pow(2).mean(dim=(2, 3)).mean(0).cpu())
        add("latent/copylast_tf_mse", (cl_tf - z_tgt).pow(2).mean(dim=(2, 3)).mean(0).cpu())
        o_frozen = obs[:, context - 1:context].expand_as(obs_tgt)
        for name, o_base in (("frozen", o_frozen), ("tf", obs[:, context - 1:-1])):
            a_all, a_occ = obs_accuracy(o_base, obs_tgt)
            add(f"obs/copylast_{name}_acc", a_all.mean(0).cpu())
            add(f"obs/copylast_{name}_acc_occ", a_occ.mean(0).cpu())

        for k in flow_steps_list:
            per_step_best, per_seq_best, best_pred, best_seq_mse = None, None, None, None
            for _ in range(samples):
                pred = model.open_loop(z, action, is_first, context=context, flow_steps=k,
                                       opponent_action=opp_action)
                mse_step = (wm.normalize(pred) - z_tgt).pow(2).mean(dim=(2, 3))  # (B,Hgen)
                mse_seq = mse_step.mean(dim=1)                                   # (B,)
                per_step_best = mse_step if per_step_best is None else torch.minimum(per_step_best, mse_step)
                if per_seq_best is None:
                    per_seq_best, best_pred = mse_seq, pred
                else:
                    better = mse_seq < per_seq_best
                    per_seq_best = torch.where(better, mse_seq, per_seq_best)
                    best_pred = torch.where(better[:, None, None, None], pred, best_pred)
            add(f"latent/mse_k{k}_step", per_step_best.mean(0).cpu())
            add(f"latent/mse_k{k}", float(per_seq_best.mean()))

            if k == rl_flow_steps:
                recon = model.tokenizer.decode(best_pred)
                a_all, a_occ = obs_accuracy(recon, obs_tgt)
                add("obs/acc_step", a_all.mean(0).cpu())
                add("obs/acc_occ_step", a_occ.mean(0).cpu())
                # Changed-cell accuracy: cells where the board actually changed
                # between consecutive real frames — the motion signal, undiluted
                # by the (mostly static) rest of the board. The teacher-forced
                # copy-last baseline predicts "no change" and bounds it from below.
                prev = obs[:, context - 1:-1]
                changed = ((obs_tgt > 0.5) != (prev > 0.5)).any(dim=2)  # (B,Hgen,H,W)
                add("obs/changed_cell_frac", float(changed.float().mean()))
                for name, o_pred in (("", recon), ("copylast_frozen_", o_frozen),
                                     ("copylast_tf_", prev)):
                    _, a_chg = obs_accuracy(o_pred, obs_tgt, cells=changed)
                    add(f"obs/{name}acc_changed_step", a_chg.mean(0).cpu())
                # Reward head along generated latents (real actions).
                z_full = torch.cat([z[:, :context], best_pred], dim=1)
                ctx_out = wm.contextualize(z_full, action, is_first,
                                           opponent_action=opp_action)
                add("reward/rollout_corr", corr(ctx_out["reward"][:, context:],
                                                b["reward"][:, context - 1:-1]))
                if render_pack is None:
                    render_pack = (obs[0].cpu(), recon[0].cpu(), context)

                # Action conditioning (NEXT_PLAN.md gate 1): the same probe under
                # counterfactual action streams, BOTH channels. "shuffled" rolls
                # another sequence's actions in (largely illegal on this board — a
                # model that learned to ignore illegal actions is RIGHT to be
                # insensitive); "noop" idles the player, a legal counterfactual
                # that must visibly change the future; the opp_* variants perturb
                # the opponent channel with the learner's stream intact; both_*
                # perturbs both. Step 1 is the clean readout: later steps compound
                # the model's own drift.
                for cf_name, act_cf, opp_cf in (
                        ("shuffled", torch.roll(action, 1, dims=0), opp_action),
                        ("noop", torch.zeros_like(action), opp_action),
                        ("opp_shuffled", action, torch.roll(opp_action, 1, dims=0)),
                        ("opp_noop", action, torch.zeros_like(opp_action)),
                        ("both_shuffled", torch.roll(action, 1, dims=0),
                         torch.roll(opp_action, 2, dims=0))):
                    step_best, seq_best, cf_best_pred, cf_best_mse = None, None, None, None
                    for _ in range(samples):
                        predC = model.open_loop(z, act_cf, is_first,
                                                context=context, flow_steps=k,
                                                opponent_action=opp_cf)
                        mse_stepC = (wm.normalize(predC) - z_tgt).pow(2).mean(dim=(2, 3))
                        mse_seqC = mse_stepC.mean(dim=1)
                        step_best = mse_stepC if step_best is None else \
                            torch.minimum(step_best, mse_stepC)
                        if cf_best_mse is None:
                            cf_best_mse, cf_best_pred = mse_seqC, predC
                        else:
                            better = mse_seqC < cf_best_mse
                            cf_best_mse = torch.where(better, mse_seqC, cf_best_mse)
                            cf_best_pred = torch.where(better[:, None, None, None],
                                                       predC, cf_best_pred)
                    add(f"latent/mse_{cf_name}_step", step_best.mean(0).cpu())
                    add(f"latent/mse_{cf_name}", float(cf_best_mse.mean()))
                    # Reward the head reads along counterfactual latents with the
                    # REAL rewards as reference: high corr would mean rewards are
                    # inferred from board dynamics alone, not the agent's actions.
                    z_cf = torch.cat([z[:, :context], cf_best_pred], dim=1)
                    ctx_cf = wm.contextualize(z_cf, act_cf, is_first,
                                              opponent_action=opp_cf)
                    add(f"reward/rollout_corr_{cf_name}",
                        corr(ctx_cf["reward"][:, context:], b["reward"][:, context - 1:-1]))

    out = {}
    for key, vals in agg.items():
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals).mean(0).tolist()
        else:
            out[key] = float(np.nanmean(vals))
    return out, render_pack


# --- 5. imagination health ---------------------------------------------------
@torch.no_grad()
def eval_imagination(model, loader, device, batches, context, horizon):
    wm = model.world_model
    agg: dict[str, list] = {}
    strip = None
    for bi, b in zip(range(batches), loader):
        b = prep_batch(b, model, device)
        obs, action, is_first = b["obs"], b["action"], b["is_first"]
        z0 = model.tokenizer.encode(obs[:, :context])
        try:
            im = model.imagine(z0, horizon=horizon,
                               ctx_action=action[:, :context],
                               ctx_is_first=is_first[:, :context],
                               ctx_opponent_action=b["opponent_action"][:, :context])
        except Exception as e:  # a crash here is itself the finding
            return {"imagine/error": repr(e)}, None
        zn = wm.normalize(im["z"])                                # (B,H+1,S,D)
        rms = zn.pow(2).mean(dim=(2, 3)).sqrt()                   # (B,H+1)
        agg.setdefault("imagine/rms_step", []).append(rms.mean(0).cpu())
        agg.setdefault("imagine/nan_frac", []).append(float(torch.isnan(im["z"]).float().mean()))

        dec = model.tokenizer.decode(im["z"])                     # (B,H+1,C,H,W)
        agg.setdefault("imagine/occupancy_step", []).append(
            occupancy(dec).float().mean(dim=(0, 2, 3)).cpu())
        agg.setdefault("imagine/contradiction_rate", []).append(contradiction_rate(dec))
        agg.setdefault("data/occupancy", []).append(float(occupancy(obs).float().mean()))
        agg.setdefault("data/contradiction_rate", []).append(contradiction_rate(obs))

        agg.setdefault("imagine/reward_mean", []).append(float(im["reward"].mean()))
        agg.setdefault("imagine/reward_std", []).append(float(im["reward"].std()))
        agg.setdefault("imagine/reward_absmax", []).append(float(im["reward"].abs().max()))
        agg.setdefault("imagine/cont_mean", []).append(float(im["cont"].mean()))
        agg.setdefault("imagine/cont_below_0.5_frac", []).append(float((im["cont"] < 0.5).float().mean()))
        agg.setdefault("imagine/mask_positive_frac", []).append(float(im["mask"].float().mean()))
        with torch.no_grad():
            real_mask = torch.sigmoid(model.tokenizer.decode_mask(
                model.tokenizer.encode(obs))) > 0.5
        agg.setdefault("data/mask_positive_frac", []).append(float(real_mask.float().mean()))
        if strip is None:
            strip = dec[0].cpu()
    out = {}
    for key, vals in agg.items():
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals).mean(0).tolist()
        else:
            out[key] = float(np.mean(vals))
    return out, strip


# --- renders -----------------------------------------------------------------
def render_strip(frames_gt, frames_pred, lo, hi, path, cell=10):
    """Time-strip PNG: top row ground truth, bottom row prediction (argmax of one
    channel group per frame). ``frames_*`` are (T,C,H,W); pred may be None."""
    from PIL import Image

    def row(frames):
        boards = []
        for t in range(frames.shape[0]):
            arg = frames[t, lo:hi].argmax(dim=0).numpy()
            rgb = _PALETTE[arg % len(_PALETTE)]
            rgb = np.kron(rgb, np.ones((cell, cell, 1), dtype=np.uint8))
            boards.append(np.pad(rgb, ((1, 1), (1, 1), (0, 0)), constant_values=255))
        return np.concatenate(boards, axis=1)

    rows = [row(frames_gt)]
    if frames_pred is not None:
        rows.append(row(frames_pred))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


# --- verdict ------------------------------------------------------------------
def build_verdict(m):
    """Mechanical PASS/WARN/FAIL per RL-blocking criterion."""
    checks = []

    # Step-1 MSE isolates conditioning (later steps compound the model's own
    # drift); the noop counterfactual is legal, so insensitivity there cannot be
    # excused as "the model learned to ignore illegal actions".
    k_rl = m.get("flow_steps_rl", 4)
    real_curve = next((m[k] for k in (f"latent/mse_k{k_rl}_step", "latent/mse_k1_step")
                       if k in m), None)
    ratios = {}
    for cf in ("noop", "shuffled"):
        curve = m.get(f"latent/mse_{cf}_step")
        if curve and real_curve and real_curve[0] > 0:
            ratios[cf] = curve[0] / real_curve[0]
    ratio = max(ratios.values()) if ratios else float("nan")
    gi = m.get("probe/self_gap_issued")
    if gi is not None:
        # Primary: issued-cell-conditioned gap (probes.py) — the aggregate
        # ratio has an SNR ceiling (~6% of latent cells carry an issued
        # action) and stays as context only.
        base = m.get("probe/mse_true", 0.0) or 1e-8
        rel = gi / base
        checks.append(("action_conditioning",
                       "PASS" if rel >= 0.10 else "WARN" if rel >= 0.03 else "FAIL",
                       f"issued-cell CF gap {gi:+.4f} ({rel * 100:+.1f}% of true-action "
                       f"MSE {base:.4f}); aggregate step-1 ratio {ratio:.3f} for context"))
    else:
        checks.append(("action_conditioning",
                       "PASS" if ratio >= 1.15 else "WARN" if ratio >= 1.05 else "FAIL",
                       f"worst counterfactual/real step-1 MSE ratio {ratio:.3f} "
                       + " ".join(f"{cf}={r:.3f}" for cf, r in ratios.items())
                       + "; <=1 means the model ignores the agent's actions"))

    # Opponent-channel conditioning (v3 gate): perturbing the opponent stream
    # must hurt too, or the WM is still marginalizing over the opponent.
    opp_ratios = {}
    for cf in ("opp_noop", "opp_shuffled"):
        curve = m.get(f"latent/mse_{cf}_step")
        if curve and real_curve and real_curve[0] > 0:
            opp_ratios[cf] = curve[0] / real_curve[0]
    ogi = m.get("probe/opp_gap_issued")
    if ogi is not None:
        base = m.get("probe/mse_true", 0.0) or 1e-8
        orel = ogi / base
        checks.append(("opponent_conditioning",
                       "PASS" if orel >= 0.10 else "WARN" if orel >= 0.03 else "FAIL",
                       f"issued-cell opponent CF gap {ogi:+.4f} ({orel * 100:+.1f}% of "
                       f"true-action MSE {base:.4f})"))
    elif opp_ratios:
        oratio = max(opp_ratios.values())
        checks.append(("opponent_conditioning",
                       "PASS" if oratio >= 1.15 else "WARN" if oratio >= 1.05 else "FAIL",
                       f"worst opponent counterfactual/real step-1 MSE ratio {oratio:.3f} "
                       + " ".join(f"{cf}={r:.3f}" for cf, r in opp_ratios.items())
                       + "; <=1 means the opponent is still an unmodeled confounder"))

    acc = m.get("obs/acc_occ_step", [])
    base = m.get("obs/copylast_frozen_acc_occ", [])
    if acc and base:
        n = min(5, len(acc))
        d = float(np.mean(acc[:n]) - np.mean(base[:n]))
        checks.append(("open_loop_beats_frozen_copylast",
                       "PASS" if d > 0.0 else "WARN" if d > -0.02 else "FAIL",
                       f"occupied-cell acc over first {n} imagined steps: model "
                       f"{np.mean(acc[:n]):.4f} vs frozen-last-frame {np.mean(base[:n]):.4f} "
                       f"(delta {d:+.4f})"))

    auc = m.get("cont/auc", float("nan"))
    p_term = m.get("cont/p_mean_terminal", float("nan"))
    status = "PASS" if (auc == auc and auc >= 0.9 and p_term < 0.5) else \
             "WARN" if (auc == auc and auc >= 0.7) else "FAIL"
    checks.append(("continue_head", status,
                   f"terminal-vs-nonterminal AUC {auc:.4f}, mean p(cont) at terminals "
                   f"{p_term:.4f} (n={m.get('cont/n_terminals', 0)}); imagination can't "
                   f"terminate episodes if this fails"))

    rc = m.get("reward/corr_nonzero", float("nan"))
    checks.append(("reward_head",
                   "PASS" if rc == rc and rc >= 0.5 else "WARN" if rc == rc and rc >= 0.2 else "FAIL",
                   f"corr(pred, true) on nonzero-reward slots {rc:.4f} "
                   f"(all slots {m.get('reward/corr_all', float('nan')):.4f}, along generated "
                   f"latents {m.get('reward/rollout_corr', float('nan')):.4f})"))

    rms = m.get("imagine/rms_step", [])
    nan_frac = m.get("imagine/nan_frac", 1.0)
    if rms:
        drift = rms[-1] / max(rms[0], 1e-8)
        status = "FAIL" if (nan_frac > 0 or not (0.5 <= drift <= 2.0)) else \
                 "PASS" if 0.8 <= drift <= 1.25 else "WARN"
        checks.append(("imagination_stability", status,
                       f"normalized latent RMS drift over horizon {drift:.3f} "
                       f"(start {rms[0]:.3f} -> end {rms[-1]:.3f}), NaN frac {nan_frac:.2e}"))
    elif "imagine/error" in m:
        checks.append(("imagination_stability", "FAIL", f"imagine() raised: {m['imagine/error']}"))

    k1 = m.get("latent/mse_k1", float("nan"))
    kmax_key = max((k for k in m if k.startswith("latent/mse_k") and k[13:].isdigit()),
                   key=lambda s: int(s[13:]), default=None)
    if kmax_key and kmax_key != "latent/mse_k1":
        km = m[kmax_key]
        status = "PASS" if km <= 1.05 * k1 else "WARN" if km <= 1.2 * k1 else "FAIL"
        checks.append(("shortcut_distillation", status,
                       f"many-step ({kmax_key[11:]}) MSE {km:.4f} vs 1-step {k1:.4f}; "
                       f"RL imagines at few steps, so k4 quality is what matters"))
    return checks


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/dreamer_dynamics_v2.pt")
    p.add_argument("--data", default="/data/micro-rts/tokdyn_pretrain_v*__*.h5")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="must match training so the split is truly held out")
    p.add_argument("--context", type=int, default=8, help="imagine_context")
    p.add_argument("--horizon", type=int, default=15, help="imagine_horizon")
    p.add_argument("--flow-steps", type=int, default=4, help="imagine_flow_steps (RL setting)")
    p.add_argument("--samples", type=int, default=4, help="best-of-N open-loop samples")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--scalar-batches", type=int, default=16)
    p.add_argument("--rollout-batches", type=int, default=4)
    p.add_argument("--imagine-batches", type=int, default=4)
    p.add_argument("--terminal-windows", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--render", type=int, default=1, help="strip PNG sets to dump (0 disables)")
    p.add_argument("--out-dir", default="checkpoints/dynamics_eval")
    p.add_argument("--smoke", action="store_true", help="tiny CPU pass over every section")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        args.batch, args.samples = 2, 2
        args.scalar_batches = args.rollout_batches = args.imagine_batches = 1
        args.terminal_windows, args.horizon, args.context = 8, 4, 4
        args.num_workers, args.device = 0, "cpu"
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, obs_shape, step, phase = load_model(args.ckpt, device)
    print(f"[eval] loaded {args.ckpt} (phase={phase}, step={step}) obs_shape={obs_shape} "
          f"latent_scale={float(model.world_model.latent_scale):.4f} device={device}")

    path = resolve_data(args.data)
    seq_len = args.context + args.horizon
    loader = build_mrts_loader(
        path, task="dynamics", seq_len=seq_len, batch_size=args.batch,
        num_workers=args.num_workers, shuffle=True, locking=False,
        val_frac=args.val_frac, split="val", drop_last=False)
    ds = loader.dataset
    print(f"[eval] data={path}  val_windows={len(ds)}  seq_len={seq_len} "
          f"(context {args.context} + horizon {args.horizon})  "
          f"terminal_obs={ds.has_terminal_obs}")
    if not ds.has_terminal_obs:
        print("[eval] WARNING: store has no terminal frames — the continue-head "
              "section will see no cont=0 targets and its verdict is meaningless.")

    metrics = {"ckpt": str(args.ckpt), "step": step, "data": path,
               "context": args.context, "horizon": args.horizon,
               "flow_steps_rl": args.flow_steps, "samples": args.samples}

    print("\n[eval] 1/5 scalar heads (teacher-forced, real latents)...")
    m1, cont_bg = eval_scalar_heads(model, loader, device, args.scalar_batches)
    metrics.update(m1)

    print("[eval] 2/5 terminal transitions (continue + terminal reward)...")
    idx = terminal_window_indices(ds, args.terminal_windows)
    print(f"       {len(idx)} val windows contain a terminal transition")
    if len(idx):
        metrics.update(eval_terminals(model, ds, idx, device, args.batch, cont_bg))

    print("[eval] 3/5 open-loop rollouts + action conditioning "
          f"(k in {{1,{args.flow_steps},{model.world_model.k_max}}}, best-of-{args.samples})...")
    flow_list = sorted({1, args.flow_steps, model.world_model.k_max})
    m3, render_pack = eval_open_loop(model, loader, device, args.rollout_batches,
                                     args.context, args.samples, flow_list, args.flow_steps)
    metrics.update(m3)

    # Issued-cell counterfactual probe (shared with the training val loop) —
    # the aggregate rolled-action ratios above have a built-in SNR ceiling
    # (~6% of latent cells carry an issued action), so this is the primary
    # action-conditioning verdict input; see probes.py.
    from entrypoints.util.probes import counterfactual_action_probe
    probe_acc: dict[str, list] = {}
    with torch.no_grad():
        for bi, b in zip(range(args.rollout_batches), loader):
            b = prep_batch(b, model, device)
            z = model.tokenizer.encode(b["obs"])
            p = counterfactual_action_probe(
                model, z, b["action"], b["opponent_action"], b["is_first"],
                context=args.context, flow_steps=args.flow_steps)
            for k, v in p.items():
                probe_acc.setdefault(k, []).append(v)
    metrics.update({k: float(np.mean(v)) for k, v in probe_acc.items()})

    print("[eval] 4/5 imagination health (predicted masks, actor sampling, "
          f"horizon {args.horizon}, {args.flow_steps} flow steps)...")
    m4, strip = eval_imagination(model, loader, device, args.imagine_batches,
                                 args.context, args.horizon)
    metrics.update(m4)

    print("[eval] 5/5 renders...")
    if args.render and render_pack is not None:
        gt, recon, ctx = render_pack
        for name, (lo, hi) in (("unit_type", CHANNEL_GROUPS["unit_type"]),
                               ("owner", CHANNEL_GROUPS["owner"])):
            render_strip(gt[ctx:], recon, lo, hi, out_dir / f"open_loop_{name}.png")
        if strip is not None:
            render_strip(strip, None, *CHANNEL_GROUPS["unit_type"],
                         out_dir / "imagined_unit_type.png")

    # ---- report -------------------------------------------------------------
    def line(k, fmt="{:.4f}"):
        v = metrics.get(k)
        if v is None:
            return
        s = fmt.format(v) if isinstance(v, float) else str(v)
        print(f"  {k:42s} {s}")

    print("\n================ scalar heads (teacher-forced) ================")
    for k in ("reward/corr_all", "reward/corr_nonzero", "reward/mse",
              "reward/nonzero_frac", "reward/abs_pred_on_zero_slots",
              "reward/true_abs_mean_nonzero", "cont/p_mean_nonterminal"):
        line(k)
    print("---------------- terminal transitions ----------------")
    for k in ("cont/n_terminals", "cont/p_mean_terminal", "cont/terminal_recall_at_0.5",
              "cont/false_terminal_rate_at_0.5", "cont/auc",
              "reward/terminal_pred_mean_win", "reward/terminal_true_mean_win",
              "reward/terminal_pred_mean_loss", "reward/terminal_true_mean_loss",
              "reward/terminal_sign_acc"):
        line(k)

    print("---------------- open-loop rollout (per-step curves) ----------------")
    steps_hdr = " ".join(f"{i+1:>7d}" for i in range(args.horizon))
    print(f"  {'step':42s} {steps_hdr}")
    for k in (f"latent/mse_k1_step", f"latent/mse_k{args.flow_steps}_step",
              f"latent/mse_k{model.world_model.k_max}_step",
              "latent/mse_noop_step", "latent/mse_shuffled_step",
              "latent/mse_opp_noop_step", "latent/mse_opp_shuffled_step",
              "latent/mse_both_shuffled_step",
              "latent/copylast_frozen_mse", "latent/copylast_tf_mse",
              "obs/acc_step", "obs/copylast_frozen_acc", "obs/copylast_tf_acc",
              "obs/acc_occ_step", "obs/copylast_frozen_acc_occ", "obs/copylast_tf_acc_occ",
              "obs/acc_changed_step", "obs/copylast_frozen_acc_changed_step",
              "obs/copylast_tf_acc_changed_step"):
        v = metrics.get(k)
        if v is None:
            continue
        print(f"  {k:42s} " + " ".join(f"{x:7.4f}" for x in v))
    for k in ("latent/mse_k1", f"latent/mse_k{args.flow_steps}",
              f"latent/mse_k{model.world_model.k_max}",
              "latent/mse_noop", "latent/mse_shuffled",
              "latent/mse_opp_noop", "latent/mse_opp_shuffled",
              "latent/mse_both_shuffled", "obs/changed_cell_frac",
              "reward/rollout_corr", "reward/rollout_corr_noop",
              "reward/rollout_corr_shuffled", "reward/rollout_corr_opp_shuffled"):
        line(k)

    print("---------------- imagination (RL settings) ----------------")
    for k in ("imagine/rms_step", "imagine/occupancy_step"):
        v = metrics.get(k)
        if v is not None:
            print(f"  {k:42s} " + " ".join(f"{x:7.4f}" for x in v))
    for k in ("imagine/contradiction_rate", "data/contradiction_rate",
              "data/occupancy", "imagine/reward_mean", "imagine/reward_std",
              "imagine/reward_absmax", "imagine/cont_mean", "imagine/cont_below_0.5_frac",
              "imagine/mask_positive_frac", "data/mask_positive_frac",
              "imagine/nan_frac", "imagine/error"):
        line(k)

    checks = build_verdict(metrics)
    metrics["verdict"] = [{"check": c, "status": s, "detail": d} for c, s, d in checks]
    print("\n================ RL-readiness verdict ================")
    for c, s, d in checks:
        print(f"  [{s:4s}] {c}: {d}")
    n_fail = sum(1 for _, s, _ in checks if s == "FAIL")
    n_warn = sum(1 for _, s, _ in checks if s == "WARN")
    overall = "GO" if n_fail == 0 and n_warn <= 1 else \
              "GO-WITH-CAUTION" if n_fail == 0 else "NO-GO"
    metrics["overall"] = overall
    print(f"\n  OVERALL: {overall}  ({n_fail} FAIL, {n_warn} WARN)")

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"\n[eval] metrics + renders -> {out_dir}")
    return metrics


if __name__ == "__main__":
    main()
