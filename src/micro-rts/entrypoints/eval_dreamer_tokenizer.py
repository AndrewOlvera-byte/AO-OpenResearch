"""Entrypoint: quantitative + qualitative eval of a trained DreamerV4 tokenizer.

The training loop only logs a single scalar MSE/BCE averaged over all 27 obs
channels x 256 cells. Because MicroRTS boards are mostly empty and every channel
group is one-hot, that average is dominated by trivially-easy "empty cell"
channels (owner=none, unit-type=none, hp-bucket=0, ...) and can look deceptively
good even if the tokenizer reconstructs occupied cells (actual units) poorly. This
script breaks the aggregate down:

1. reproduces the training-loop scalar metrics on fresh sampled batches (sanity
   check against the last logged training-loop numbers),
2. per-channel-group recon MSE + categorical argmax accuracy (hp/resources/owner/
   unit-type/action), each split into all-cells vs. occupied-cells-only (occupied
   = unit-type argmax != "none"), which is the number that actually answers "does
   it reconstruct units, not just empty board",
3. predicted action-mask precision/recall (mask is highly imbalanced: most cells
   are not source-selectable), and
4. a handful of PNG side-by-side (ground-truth | reconstruction) board renders for
   eyeballing, written to --out-dir.

Usage (inside the container)::

    python src/micro-rts/entrypoints/eval_dreamer_tokenizer.py \
        --ckpt checkpoints/dreamer_tokenizer.pt \
        --data '/data/micro-rts/tokdyn_pretrain_v1__*.h5' \
        --batches 20 --batch 64 --seq-len 16 --render 6
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PKG = _HERE.parents[1]          # src/micro-rts
_SRC = _HERE.parents[2]          # src
for p in (str(_PKG), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from core.registry import build  # noqa: E402
import models.dreamer  # noqa: E402,F401  (registry side effect)
from collectors.offline_data import build_mrts_loader, to_device  # noqa: E402
from trainers.BaseTrainer import resolve_device  # noqa: E402

# gym_microrts's standard 27-plane GridNet obs layout (owner/unit-type/hp/resources
# all one-hot, "current action" one-hot) -- see environments/microrts_env.py.
CHANNEL_GROUPS = {
    "hp": (0, 5),
    "resources": (5, 10),
    "owner": (10, 13),
    "unit_type": (13, 21),
    "action": (21, 27),
}
NONE_UNIT_TYPE_CH = 13  # "unit_type: none" one-hot channel -> empty-cell indicator

# Distinct RGB per class within a group, for the PNG render (index 0 is always
# "none"/background -> dark grey).
_PALETTE = [
    (30, 30, 30), (220, 50, 47), (38, 139, 210), (133, 153, 0), (211, 54, 130),
    (181, 137, 0), (108, 113, 196), (42, 161, 152), (203, 75, 22),
]


def resolve_data(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f"no dataset matches {pattern!r}")
    return matches[-1]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    obs_shape = tuple(ckpt["obs_shape"])
    action_nvec = ckpt["action_nvec"]
    model_cfg = ckpt["model_cfg"]
    model = build("model", type=model_cfg.get("type", "dreamerv4"),
                  obs_shape=obs_shape, action_nvec=action_nvec, device=str(device),
                  **{k: v for k, v in model_cfg.items() if k != "type"})
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, obs_shape, ckpt.get("step")


@torch.no_grad()
def eval_batch(model, obs, mask):
    """One batch of quantitative metrics. obs (B,T,C,H,W) float one-hot; mask (B,T,HW,MW) bool."""
    tok = model.tokenizer
    z = tok.encode(obs)
    recon = tok.decode(z)
    mask_logits = tok.decode_mask(z)

    out = {}
    out["overall/recon_mse"] = F.mse_loss(recon, obs).item()
    out["overall/mask_bce"] = F.binary_cross_entropy_with_logits(mask_logits, mask.float()).item()
    out["latent/z_abs_mean"] = z.abs().mean().item()
    out["latent/z_sat"] = (z.abs() > 0.99).float().mean().item()

    occ = obs[:, :, NONE_UNIT_TYPE_CH] < 0.5  # (B,T,H,W) True where a unit sits
    n_occ = occ.sum().item()
    n_total = occ.numel()
    out["occupancy/frac_cells_occupied"] = n_occ / max(n_total, 1)

    for name, (lo, hi) in CHANNEL_GROUPS.items():
        g_obs = obs[:, :, lo:hi]                      # (B,T,G,H,W)
        g_rec = recon[:, :, lo:hi]
        out[f"group/{name}/mse_all"] = F.mse_loss(g_rec, g_obs).item()

        tgt = g_obs.argmax(dim=2)                      # (B,T,H,W)
        pred = g_rec.argmax(dim=2)
        correct = (tgt == pred)
        out[f"group/{name}/acc_all"] = correct.float().mean().item()

        occ_correct = correct[occ]
        out[f"group/{name}/acc_occupied"] = (
            occ_correct.float().mean().item() if occ_correct.numel() else float("nan"))
        g_obs_occ = g_obs.permute(0, 1, 3, 4, 2)[occ]   # (n_occ, G)
        g_rec_occ = g_rec.permute(0, 1, 3, 4, 2)[occ]
        out[f"group/{name}/mse_occupied"] = (
            F.mse_loss(g_rec_occ, g_obs_occ).item() if g_rec_occ.numel() else float("nan"))

    # Mask precision/recall (mask is heavily class-imbalanced: most cells are not
    # source-selectable / most components have few legal choices).
    pred_mask = mask_logits > 0.0
    tp = (pred_mask & mask).sum().item()
    fp = (pred_mask & ~mask).sum().item()
    fn = (~pred_mask & mask).sum().item()
    out["mask/precision"] = tp / max(tp + fp, 1)
    out["mask/recall"] = tp / max(tp + fn, 1)
    out["mask/positive_frac"] = mask.float().mean().item()
    return out, recon, mask_logits


def avg_metrics(dicts):
    keys = dicts[0].keys()
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def render_grid(channels_argmax_gt, channels_argmax_pred, group_name, path, cell=16):
    """Side-by-side (ground-truth | reconstruction) PNG for one channel group, one frame."""
    from PIL import Image

    h, w = channels_argmax_gt.shape
    img = Image.new("RGB", (2 * w * cell + cell, h * cell), (255, 255, 255))
    for gi, arr in enumerate((channels_argmax_gt, channels_argmax_pred)):
        ox = gi * (w * cell + cell)
        for r in range(h):
            for c in range(w):
                color = _PALETTE[int(arr[r, c]) % len(_PALETTE)]
                for dy in range(cell - 1):
                    for dx in range(cell - 1):
                        img.putpixel((ox + c * cell + dx, r * cell + dy), color)
    img.save(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/dreamer_tokenizer.pt")
    p.add_argument("--data", default="/data/micro-rts/tokdyn_pretrain_v1__*.h5")
    p.add_argument("--batches", type=int, default=20, help="held-out batches to average over")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--render", type=int, default=6, help="number of (gt|recon) PNG frames to dump")
    p.add_argument("--out-dir", default="checkpoints/tokenizer_eval")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, obs_shape, step = load_model(args.ckpt, device)
    print(f"[eval] loaded {args.ckpt} (trained to step={step}) obs_shape={obs_shape} device={device}")

    path = resolve_data(args.data)
    loader = build_mrts_loader(path, task="tokenizer", seq_len=args.seq_len,
                               batch_size=args.batch, num_workers=args.num_workers,
                               shuffle=True, locking=False)
    print(f"[eval] data={path}  windows={len(loader.dataset)}  batches={args.batches}")

    per_batch, rendered = [], 0
    for i, batch in zip(range(args.batches), loader):
        batch = to_device(batch, device)
        m, recon, mask_logits = eval_batch(model, batch["obs"], batch["mask"])
        per_batch.append(m)

        if rendered < args.render:
            obs, rec = batch["obs"], recon
            b, t = 0, 0  # first frame of first sequence in the batch
            for name, (lo, hi) in CHANNEL_GROUPS.items():
                gt = obs[b, t, lo:hi].argmax(dim=0).cpu().numpy()
                pr = rec[b, t, lo:hi].argmax(dim=0).cpu().numpy()
                render_grid(gt, pr, name, out_dir / f"frame{rendered:02d}_{name}.png")
            rendered += 1

    m = avg_metrics(per_batch)
    print(f"\n[eval] averaged over {args.batches} batches x {args.batch}x{args.seq_len} "
          f"({args.batches * args.batch * args.seq_len} frames)\n")
    print(f"  overall/recon_mse       {m['overall/recon_mse']:.6f}")
    print(f"  overall/mask_bce        {m['overall/mask_bce']:.6f}")
    print(f"  latent/z_abs_mean       {m['latent/z_abs_mean']:.4f}")
    print(f"  latent/z_sat            {m['latent/z_sat']:.4f}")
    print(f"  occupancy/frac_occupied {m['occupancy/frac_cells_occupied']:.4f}")
    print(f"  mask/precision          {m['mask/precision']:.4f}")
    print(f"  mask/recall             {m['mask/recall']:.4f}")
    print(f"  mask/positive_frac      {m['mask/positive_frac']:.6f}")
    print()
    print(f"  {'group':12s} {'mse_all':>10s} {'mse_occ':>10s} {'acc_all':>9s} {'acc_occ':>9s}")
    for name in CHANNEL_GROUPS:
        print(f"  {name:12s} {m[f'group/{name}/mse_all']:10.6f} "
              f"{m[f'group/{name}/mse_occupied']:10.6f} "
              f"{m[f'group/{name}/acc_all']:9.4f} {m[f'group/{name}/acc_occupied']:9.4f}")
    print(f"\n[eval] rendered {rendered} (ground-truth | reconstruction) frames -> {out_dir}")
    return m


if __name__ == "__main__":
    main()
