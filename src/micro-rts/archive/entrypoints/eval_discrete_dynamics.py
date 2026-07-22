"""Evaluate teacher-forced and true autoregressive discrete MicroRTS dynamics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for path in (HERE.parents[2], HERE.parents[3]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from collectors.offline_data import build_mrts_loader, to_device  # noqa: E402
from entrypoints.structured_v2_common import device_from, resolve_latest  # noqa: E402
from models.dreamer_v2 import (  # noqa: E402
    DiscreteDynamicsConfig,
    DiscreteStructuredWorldModel,
    DiscreteTokenizerConfig,
    discrete_causal_paired_loss,
)


def _f1(pred, target):
    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    return 2 * tp / (2 * tp + fp + fn).clamp_min(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--autoregressive-batches", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = device_from(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    tc = DiscreteTokenizerConfig.from_dict(payload["discrete_tokenizer_cfg"])
    dc = DiscreteDynamicsConfig.from_dict(payload["discrete_dynamics_cfg"])
    model = DiscreteStructuredWorldModel(tuple(payload["grid_hw"]), tc, dc).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    loader = build_mrts_loader(
        resolve_latest(args.data),
        task="structured_dynamics_eval",
        seq_len=1,
        batch_size=args.batch,
        num_workers=0,
        locking=False,
        shuffle=False,
        drop_last=False,
    )
    teacher_sums, teacher_count = {}, 0
    ar = {
        "code_acc": 0.0,
        "exact_cell": 0.0,
        "exact_frame": 0.0,
        "exact_globals": 0.0,
        "changed_cell_f1": 0.0,
    }
    ar_count = 0
    with torch.no_grad():
        for index, raw in enumerate(loader):
            if index >= args.batches:
                break
            batch = to_device(raw, device)
            _, metrics = discrete_causal_paired_loss(model, batch)
            for key, value in metrics.items():
                teacher_sums[key] = teacher_sums.get(key, 0.0) + float(value)
            teacher_count += 1
            if index >= args.autoregressive_batches:
                continue
            state = batch["state"].reshape(-1, *batch["state"].shape[-2:])
            glob = batch["globals"].reshape(-1, batch["globals"].shape[-1])
            nxt = batch["next_state"].reshape(-1, *batch["next_state"].shape[-2:])
            nglob = batch["next_globals"].reshape(-1, batch["next_globals"].shape[-1])
            action = batch["action"].reshape(-1, *batch["action"].shape[-2:])
            opponent = batch["opponent_action"].reshape(-1, *batch["opponent_action"].shape[-2:])
            current_codes = model.tokenizer.encode_codes(state, glob)
            target_codes = model.tokenizer.encode_codes(nxt, nglob)
            events, valid, _ = model.action_events(state, action, opponent)
            generated = model.dynamics.generate(
                current_codes,
                events,
                valid,
                router_state_tokens=model.router_tokens(current_codes),
            )
            pred_state, pred_globals = model.tokenizer.discretize(
                model.tokenizer.decode_codes(generated)
            )
            current = state.clone(); current[..., 2] = -1
            target = nxt.clone(); target[..., 2] = -1
            exact_cell = (pred_state == target).all(-1)
            ar["code_acc"] += float((generated == target_codes).float().mean())
            ar["exact_cell"] += float(exact_cell.float().mean())
            ar["exact_frame"] += float(exact_cell.all(-1).float().mean())
            ar["exact_globals"] += float((pred_globals == nglob).all(-1).float().mean())
            ar["changed_cell_f1"] += float(
                _f1((pred_state != current).any(-1), (target != current).any(-1))
            )
            ar_count += 1

    teacher = {key: value / max(teacher_count, 1) for key, value in teacher_sums.items()}
    print(
        "[discrete-eval teacher] "
        f"loss={teacher['dynamics/total']:.6f} "
        f"code={teacher['dynamics/code_acc']:.6f} "
        f"changed_code={teacher['dynamics/changed_code_acc']:.6f} "
        f"cell={teacher['dynamics/exact_cell_teacher_forced']:.6f} "
        f"frame={teacher['dynamics/exact_frame_teacher_forced']:.6f} "
        f"changed_f1={teacher['dynamics/changed_cell_f1_teacher_forced']:.6f}"
    )
    if ar_count:
        print(
            "[discrete-eval autoregressive] "
            + " ".join(f"{key}={value / ar_count:.6f}" for key, value in ar.items())
        )


if __name__ == "__main__":
    main()
