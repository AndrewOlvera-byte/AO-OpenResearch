"""Compare structured observation-tokenizer checkpoints on fixed held-out rows.

Common metrics certify reconstruction and the frozen latent interface.  A JEPA
checkpoint additionally evaluates its disposable EMA target/predictor, clearly
separated from metrics available to every tokenizer.

Example::

    python src/micro-rts/entrypoints/compare_structured_tokenizers.py \
      --checkpoint jepa=checkpoints/pretrain_medium_jepa/best.pt \
      --checkpoint recon=checkpoints/pretrain_medium_recon/best.pt \
      --data /data/micro-rts/wm_v2_pretrain__20260713-030227__5a273944.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for path in (HERE.parents[1], HERE.parents[2]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from collectors.offline_data import build_mrts_loader, to_device  # noqa: E402
from entrypoints.pretrain_common import amp_ctx, setup_backend  # noqa: E402
from models.dreamer_v2 import (  # noqa: E402
    StructuredTokenizer,
    StructuredTokenizerConfig,
    TemporalJEPAConfig,
    TemporalJEPATokenizerPretrainer,
    structured_reconstruction_loss,
    structured_temporal_jepa_loss,
    structured_tokenizer_state_dict,
)
from trainers.BaseTrainer import resolve_device  # noqa: E402


def _flatten_rows(value: torch.Tensor, trailing: int) -> torch.Tensor:
    return value.reshape(-1, *value.shape[-trailing:])


@torch.no_grad()
def evaluate_tokenizer_batch(tokenizer, batch: dict) -> dict[str, torch.Tensor]:
    """Return common exactness, semantic, and intrinsic transition metrics."""
    state = _flatten_rows(batch["state"], 2)
    glob = _flatten_rows(batch["globals"], 1)
    nxt = _flatten_rows(batch["next_state"], 2)
    nglob = _flatten_rows(batch["next_globals"], 1)
    obs = _flatten_rows(batch["obs"], 3)
    mask = _flatten_rows(batch["mask"], 2)

    decoded, z0 = tokenizer(state, glob)
    reconstruction, semantic = structured_reconstruction_loss(
        tokenizer, decoded, state, glob, obs, mask
    )
    pred_state, pred_globals = tokenizer.discretize(decoded)
    true_state = state.clone()
    true_state[..., 2] = -1
    exact_cell = (pred_state == true_state).all(-1)
    exact_global = (pred_globals == glob).all(-1)

    occupied = true_state[..., 1].bool()
    assigned = true_state[..., 7].bool()
    raster_target = obs.movedim(-3, -1).reshape(
        obs.shape[0], tokenizer.h * tokenizer.w, -1
    ).bool()
    raster_exact = ((decoded["legacy_obs"] >= 0) == raster_target).all(-1)
    mask_exact = ((decoded["mask"] >= 0) == mask.bool()).all(-1)

    z1 = tokenizer.encode(nxt, nglob)
    z0n = F.layer_norm(z0.float(), (z0.shape[-1],))
    z1n = F.layer_norm(z1.float(), (z1.shape[-1],))
    transition_delta = z1n - z0n

    metrics = {
        "reconstruction": reconstruction,
        "exact_cell": exact_cell.float().mean(),
        "exact_frame": exact_cell.all(-1).float().mean(),
        "exact_globals": exact_global.float().mean(),
        "exact_roundtrip": (exact_cell.all(-1) & exact_global).float().mean(),
        "exact_occupied_cell": (
            (exact_cell & occupied).sum().float() / occupied.sum().clamp_min(1)
        ),
        "exact_assigned_cell": (
            (exact_cell & assigned).sum().float() / assigned.sum().clamp_min(1)
        ),
        "exact_raster_cell": raster_exact.float().mean(),
        "exact_raster": raster_exact.all(-1).float().mean(),
        "exact_mask_cell": mask_exact.float().mean(),
        "exact_mask": mask_exact.all(-1).float().mean(),
        "latent_rms": z0.float().square().mean().sqrt(),
        "transition_delta_mse": transition_delta.square().mean(),
        "transition_delta_rms": transition_delta.square().mean().sqrt(),
    }
    for index, name in ((13, "start_tick"), (14, "eta"), (15, "remaining")):
        use = assigned
        metrics[f"exact_{name}"] = (
            ((pred_state[..., index] == true_state[..., index]) & use).sum().float()
            / use.sum().clamp_min(1)
        )

    cf_valid = batch["counterfactual_valid"].reshape(-1).bool()
    if cf_valid.any():
        raw_cf_state = _flatten_rows(batch["counterfactual_next_state"], 2)
        raw_cf_glob = _flatten_rows(batch["counterfactual_next_globals"], 1)
        cf_state = torch.where(cf_valid[:, None, None], raw_cf_state, nxt)
        cf_glob = torch.where(cf_valid[:, None], raw_cf_glob, nglob)
        zcf = tokenizer.encode(cf_state, cf_glob)[cf_valid]
        zcfn = F.layer_norm(zcf.float(), (zcf.shape[-1],))
        effect = zcfn - z1n[cf_valid]
        true_effect = (
            (raw_cf_state[cf_valid] != nxt[cf_valid]).any(-1).any(-1)
            | (raw_cf_glob[cf_valid] != nglob[cf_valid]).any(-1)
        )
        metrics["paired_nonzero_fraction"] = true_effect.float().mean()
        if true_effect.any():
            metrics["paired_effect_rms"] = effect[true_effect].square().mean().sqrt()

    # Preserve the existing field metrics under a compact common namespace.
    for key, value in semantic.items():
        if key.startswith("tok/") and key != "tok/total":
            metrics[key.removeprefix("tok/")] = value
    return {key: value.detach() for key, value in metrics.items()}


def _load_model(checkpoint, ds, device):
    tc = StructuredTokenizerConfig.from_dict(checkpoint["tokenizer_cfg"])
    tc.mask_width = 1 + sum(ds.action_nvec[1:])
    tc.legacy_obs_channels = ds.obs_channels
    tokenizer = StructuredTokenizer(ds.grid_hw, tc).to(device)
    tokenizer.load_state_dict(structured_tokenizer_state_dict(checkpoint))
    tokenizer.eval()
    return tokenizer, tc


def _load_jepa(checkpoint, tc, grid_hw, device):
    if not checkpoint.get("temporal_jepa_cfg"):
        return None
    if not any(name.startswith("predictor.") for name in checkpoint["model"]):
        return None
    jc = TemporalJEPAConfig.from_dict(checkpoint["temporal_jepa_cfg"])
    model = TemporalJEPATokenizerPretrainer(grid_hw, tc, jc).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _parse_checkpoint(value: str):
    label, sep, path = value.partition("=")
    if not sep or not label or not path:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    return label, path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", action="append", type=_parse_checkpoint, required=True
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", default=None, help="optional result JSON path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    setup_backend(device)
    loader = build_mrts_loader(
        args.data,
        task="structured_tokenizer_jepa",
        seq_len=1,
        batch_size=args.batch,
        num_workers=0,
        locking=False,
        val_frac=args.val_frac,
        split="val",
        drop_last=False,
        shuffle=False,
        fixed_chunk_batches=args.batches,
        fixed_chunk_seed=args.seed,
    )
    results = {}
    for label, path in args.checkpoint:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        tokenizer, tc = _load_model(checkpoint, loader.dataset, device)
        jepa = _load_jepa(checkpoint, tc, loader.dataset.grid_hw, device)
        sums, count = {}, 0
        for batch in loader:
            batch = to_device(batch, device)
            with amp_ctx(device, device.type == "cuda"):
                metrics = evaluate_tokenizer_batch(tokenizer, batch)
                if jepa is not None:
                    _, native = structured_temporal_jepa_loss(jepa, batch)
                    metrics.update(
                        {
                            f"native_{key.removeprefix('tok/jepa_')}": value
                            for key, value in native.items()
                            if key.startswith("tok/jepa_")
                        }
                    )
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            count += 1
        values = {key: value / count for key, value in sums.items()}
        values["checkpoint_step"] = int(checkpoint.get("step", 0))
        results[label] = values

    columns = (
        "reconstruction",
        "exact_cell",
        "exact_frame",
        "exact_globals",
        "exact_roundtrip",
        "exact_raster",
        "exact_mask_cell",
        "exact_start_tick",
        "exact_eta",
        "exact_remaining",
        "native_factual",
        "native_counterfactual",
        "native_effect",
        "native_effect_cosine",
        "native_effect_norm_ratio_aggregate",
    )
    print("model step " + " ".join(columns))
    for label, values in results.items():
        row = [label, str(values["checkpoint_step"])]
        row.extend(
            "-" if key not in values else f"{values[key]:.6f}" for key in columns
        )
        print(" ".join(row))
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"saved {target}")
    return results


if __name__ == "__main__":
    main()
