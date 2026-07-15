"""CLI: inspect & validate a collected MicroRTS dataset (shapes / dataloader).

Four stages, each asserting the on-disk store matches the DreamerV4 data
contract, so it doubles as a debugging tool and a shape sanity check:

1. raw HDF5 structure (datasets, dtypes, chunks, attrs, trajectory index),
2. a mock ``torch`` DataLoader (``IterableDataset`` over
   :class:`H5SequenceDataset`) — validates batch ``[B,T,...]`` shapes/dtypes,
3. semantic invariants (obs one-hot binary, action within nvec, mask binary,
   every recorded action legal-or-degenerate under its stored mask); on
   ``has_terminal_obs`` stores additionally: ``terminal_obs`` nonzero exactly
   at done rows (streamed), terminal frames one-hot and distinct from the
   post-reset obs, win-reward distribution over done rows,
4. model-readiness — builds a small real :class:`DreamerV4` from the file's
   attrs and runs the actual ``world_model_loss`` on a sampled batch; on
   terminal stores also on a **spliced** batch via :class:`MRTSSequenceDataset`
   windows that contain a terminal boundary (the actual v2 training path, so
   cont=0 / win-reward supervision is exercised end-to-end).

Usage (inside the container)::

    python src/micro-rts/collectors/offline_data/inspect_dataset.py \
        /data/micro-rts/debug__*.h5
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

if __package__ in (None, ""):
    _here = Path(__file__).resolve()
    sys.path.insert(0, str(_here.parents[2]))  # src/micro-rts
    sys.path.insert(0, str(_here.parents[3]))  # src


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="dataset .h5 (globs allowed; newest match is used)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--iters", type=int, default=4, help="mock loader iterations")
    p.add_argument("--skip-invariants", action="store_true",
                   help="skip stage 3 (one-hot/legality checks; for synthetic files)")
    p.add_argument("--skip-model", action="store_true",
                   help="skip stage 4 (real DreamerV4 world_model_loss)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    import h5py
    import numpy as np
    import torch

    from collectors.offline_data import H5SequenceDataset

    args = parse_args(argv)
    matches = sorted(glob.glob(args.path))
    if not matches:
        raise SystemExit(f"no file matches {args.path!r}")
    path = matches[-1]
    B, T = args.batch, args.seq_len
    print(f"=== FILE: {path} ===\n")

    # -- 1. raw store --------------------------------------------------------
    print("--- 1. raw HDF5 structure ---")
    with h5py.File(path, "r") as f:
        has_term = bool(f.attrs.get("has_terminal_obs", False))
        fields = ("obs", "action", "opponent_action", "mask", "reward",
                  "raw_rewards", "done", "is_first") \
            + (("terminal_obs",) if has_term else ())
        for k in fields:
            d = f[k]
            print(f"  {k:12s} shape={str(d.shape):22s} dtype={str(d.dtype):8s} chunks={d.chunks}")
        A = dict(f.attrs)
        C = f["obs"].shape[1]
        obs_shape = (C, *(int(x) for x in A["grid_hw"]))
        action_nvec = [int(x) for x in A["action_nvec"]]
        opp = f["traj"]["opponent_id"][:]
        lens = f["traj"]["length"][:]
        maps, opps = list(A["maps"]), list(A["opponents"])
        pols = list(A.get("policies", []))
        print(f"  attrs: num_steps={A['num_steps']} num_trajectories={A['num_trajectories']} "
              f"format_version={A.get('format_version')} has_terminal_obs={has_term}")
        print(f"  maps={maps}  opponents={opps}  policies={pols}")
        pol_id = f["traj"]["policy_id"][:]
        eps = f["traj"]["action_noise"][:]
        pcounts = {(pols[i] if 0 <= i < len(pols) else i): int(c)
                   for i, c in zip(*np.unique(pol_id, return_counts=True))}
        print(f"  per-policy trajs={pcounts}  action_noise values={sorted(set(eps.tolist()))}")
        print(f"  derived obs_shape={obs_shape}  action_nvec={action_nvec}")
        counts = {opps[i]: int(c) for i, c in zip(*np.unique(opp, return_counts=True))}
        print(f"  traj lengths={sorted(set(lens.tolist()))}  per-opponent trajs={counts}")
    print()

    C, H, W = obs_shape
    G, MW = action_nvec[0], 1 + sum(action_nvec[1:])
    N_COMP = len(action_nvec) - 1

    # -- 2. mock torch DataLoader -------------------------------------------
    print("--- 2. mock DataLoader (IterableDataset over H5SequenceDataset) ---")

    class H5Iterable(torch.utils.data.IterableDataset):
        def __init__(self, ds, batch, seq_len, iters):
            self.ds, self.batch, self.seq_len, self.iters = ds, batch, seq_len, iters

        def __iter__(self):
            for _ in range(self.iters):
                yield self.ds.sample(self.batch, self.seq_len)

    ds = H5SequenceDataset(path)
    if not ds.can_sample(T):
        raise SystemExit(f"no trajectory has length >= seq_len={T}")
    # batch_size=None + identity collate: the dataset yields a whole [B,T,...]
    # TensorDict per item, so pass it through untouched (the default collate would
    # try to re-batch the TensorDict and choke on its string keys).
    loader = torch.utils.data.DataLoader(
        H5Iterable(ds, B, T, args.iters), batch_size=None, num_workers=0,
        collate_fn=lambda batch: batch)

    expected = {
        "obs": ((B, T, C, H, W), torch.float32),
        "action": ((B, T, G, N_COMP), torch.int64),
        "opponent_action": ((B, T, G, N_COMP), torch.int64),
        "mask": ((B, T, G, MW), torch.bool),
        "reward": ((B, T), torch.float32),
        "raw_rewards": ((B, T, 6), torch.float32),
        "cont": ((B, T), torch.float32),
        "is_first": ((B, T), torch.bool),
    }
    n = 0
    for batch in loader:
        n += 1
        for k, (shape, dtype) in expected.items():
            assert tuple(batch[k].shape) == shape, f"{k}: {tuple(batch[k].shape)} != {shape}"
            assert batch[k].dtype == dtype, f"{k}: dtype {batch[k].dtype} != {dtype}"
    print(f"  {n} batches drained; all shapes/dtypes OK:")
    for k, (shape, dtype) in expected.items():
        print(f"    {k:12s} {shape}  {dtype}")
    print()

    # -- 3. semantic invariants ---------------------------------------------
    b = ds.sample(B, T)
    if not args.skip_invariants:
        print("--- 3. semantic invariants on a batch ---")
        cell_nvec = torch.as_tensor(action_nvec[1:])
        assert set(torch.unique(b["obs"]).tolist()) <= {0.0, 1.0}, "obs not one-hot binary"
        for name in ("action", "opponent_action"):
            checked = b[name].clone()
            explicit_none = (checked[..., 0] == 0) & (checked[..., 6] == 255)
            checked[..., 6][explicit_none] = 0
            assert (checked >= 0).all() and (checked < cell_nvec).all(), \
                f"{name} out of range"
        assert set(torch.unique(b["mask"].float()).tolist()) <= {0.0, 1.0}, "mask not binary"
        assert ((b["cont"] == 0) | (b["cont"] == 1)).all(), "cont not in {0,1}"
        cm = b["mask"][..., 1:].reshape(-1, MW - 1)
        act = b["action"].reshape(-1, N_COMP)
        legal, off = True, 0
        for i, sz in enumerate(action_nvec[1:]):
            comp_mask = cm[:, off:off + sz]
            has_valid = comp_mask.any(-1)
            chosen_ok = comp_mask.gather(1, act[:, i:i + 1].clamp(max=sz - 1)).squeeze(1).bool()
            legal &= bool((chosen_ok | ~has_valid).all())  # legal, or degenerate (ignored)
            off += sz
        assert legal, "a recorded action is illegal under its stored mask"
        print("  obs one-hot binary + action in nvec range + mask binary + cont in {0,1}: OK")
        print("  every recorded action legal-or-degenerate under its mask: OK")

        if has_term:
            # Terminal-frame invariants on the WHOLE store. terminal_obs is
            # sparse (zeros off the done rows), so stream block-wise to keep RAM
            # bounded — a full v2-sized read would decompress ~obs-sized data.
            with h5py.File(path, "r") as f:
                done = f["done"][:].astype(bool)
                S = done.shape[0]
                nz = np.zeros(S, dtype=bool)
                blk = 8192
                for s0 in range(0, S, blk):
                    t = f["terminal_obs"][s0:s0 + blk]
                    nz[s0:s0 + len(t)] = t.reshape(len(t), -1).any(axis=1)
                assert (nz == done).all(), \
                    "terminal_obs must be nonzero exactly at done rows"
                d = np.flatnonzero(done)
                assert d.size > 0, "has_terminal_obs store contains no done rows"
                term_d = f["terminal_obs"][d]
                # Anti-regression for the jar bug (terminal frame silently replaced
                # by the NEXT episode's first frame): compare against the reset
                # frame, which sits at the next row (is_first) of the same lane
                # block. NOT against obs[d] — that is the departure frame, and a
                # quiet truncation tick may legitimately change nothing.
                is_first = f["is_first"][:].astype(bool)
                nxt = d + 1
                chk = nxt[(nxt < S)]
                chk = chk[is_first[chk]]
                assert chk.size > 0, "no done row has its reset frame in-block"
                term_chk = f["terminal_obs"][chk - 1]
                reset_chk = f["obs"][chk]
                diff = (term_chk != reset_chk).reshape(chk.size, -1).sum(axis=1)
                assert (diff > 0).all(), \
                    "a terminal frame equals the next episode's reset frame (jar bug)"
                # one-hot per plane group [hp 5, res 5, player 3, unit-type C-19, action 6]
                edges = np.cumsum([0, 5, 5, 3, C - 19, 6])
                for g in range(5):
                    s = term_d[:, edges[g]:edges[g + 1]].sum(axis=1)
                    assert (s == 1).all(), f"terminal frames: plane group {g} not one-hot"
                win = f["raw_rewards"][d][:, 0]
                vals, cnts = np.unique(win, return_counts=True)
                dist = {float(v): int(c) for v, c in zip(vals, cnts)}
            print(f"  terminal_obs nonzero exactly at {d.size} done rows "
                  f"({d.size / S:.5f} of {S}): OK")
            print(f"  terminal frames one-hot; {chk.size}/{d.size} checked against their "
                  f"reset frame, all differ (cells min/mean/max "
                  f"{diff.min()}/{diff.mean():.0f}/{diff.max()}): OK")
            print(f"  win-reward at done rows (-1 loss / 0 truncation / +1 win): {dist}")
        print()

    # -- 4. real DreamerV4 world_model_loss ---------------------------------
    if not args.skip_model:
        print("--- 4. model-readiness: real world_model_loss on a batch ---")
        from loss.dreamer import world_model_loss
        from models.dreamer.config import DreamerV4Config
        from models.dreamer.dreamer import DreamerV4

        cfg = DreamerV4Config.from_dict({
            "tokenizer": {"d_latent": 16, "enc_channels": 32},
            "dynamics": {"d_model": 64, "depth": 2, "n_heads": 4},
        })
        model = DreamerV4(obs_shape, action_nvec, cfg=cfg)
        loss, metrics, z = world_model_loss(
            model, b["obs"], b["action"], b["reward"], b["cont"], b["mask"],
            b["is_first"], opponent_action=b["opponent_action"])
        assert torch.isfinite(loss), "non-finite world-model loss"
        assert z.shape[:2] == (B, T), "latent batch/time mismatch"
        print(f"  latents z={tuple(z.shape)} (B,T,n_spatial,d_latent)  loss={float(loss.detach()):.4f}")
        for k, v in metrics.items():
            print(f"    {k:14s} {v:.4f}")

        if has_term:
            # The v2 training path: MRTSSequenceDataset windows that contain a
            # terminal boundary, so the spliced terminal frame and its cont=0 /
            # win-reward targets actually flow through the loss.
            from collectors.offline_data.mrts_dataset import MRTSSequenceDataset

            with h5py.File(path, "r") as f:
                done_rows = np.flatnonzero(f["done"][:])
            # locking=None -> h5py's default flags, matching the tool's other
            # handles on this file (h5py rejects mixed lock flags per file).
            mds = MRTSSequenceDataset(path, seq_len=T, task="all", locking=None)
            ws = mds._win_start  # sorted ascending (trajectories in file order)
            chosen = []
            for r in done_rows:
                # windows [s, s+T) holding row r at an interior offset <= T-2,
                # so the spliced arrival slot r-s+1 stays inside the window.
                lo, hi = np.searchsorted(ws, [r - (T - 2), r + 1])
                chosen.extend(range(lo, hi))
                if len(chosen) >= B:
                    break
            assert chosen, "no dataset window contains a terminal boundary"
            chosen = chosen[:B]
            items = [mds[i] for i in chosen]
            sb = {k: torch.stack([it[k] for it in items]) for k in items[0]}
            mds.close()
            n_term = int((sb["cont"] == 0).sum())
            assert n_term > 0, "spliced batch lost its cont=0 targets"
            assert bool((~sb["mask"].reshape(len(chosen), T, -1).any(-1)).any(1).all()), \
                "a spliced window has no all-illegal (game-over) mask slot"
            loss2, metrics2, _ = world_model_loss(
                model, sb["obs"], sb["action"], sb["reward"], sb["cont"],
                sb["mask"], sb["is_first"], opponent_action=sb["opponent_action"])
            assert torch.isfinite(loss2), "non-finite loss on spliced batch"
            print(f"  spliced batch ({len(chosen)} terminal windows, {n_term} cont=0 targets): "
                  f"loss={float(loss2.detach()):.4f} wm/continue={metrics2['wm/continue']:.4f}")
        print()

    ds.close()
    print("=== ALL VALIDATIONS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
