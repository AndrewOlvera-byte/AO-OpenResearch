# World-action v2 collection runbook

The collection launcher produces three physically separate augmented datasets:

```text
data/micro-rts/world_action_v2/
  train.h5
  validation.h5
  evaluation.h5
  raw/
    train.h5
    validation.h5
    evaluation.h5
  manifests/
    audit.json
```

Each `traj` record is one complete, contiguous game episode. Partial tails at a
policy/map block boundary are counted in `dropped_partial_rows` and are not
written. Episode records include `episode_id`, `collection_seed`, `map_id`,
`opponent_id`, `policy_id`, `action_noise`, `seat`, `length`, and
`terminal_outcome`.

The final files contain the factual structured/raster transition, both issued
actions, exact held-opponent-action counterfactual transition and raster, plus
factual and counterfactual ego observations and visibility masks.

## Infrastructure and smoke

```bash
cd infra
docker compose build research
docker compose up -d research
cd ..

SMOKE=1 NUM_ENVS=2 POLICY_DEVICE=cpu MAX_EPISODE_STEPS=32 \
ROOT=/data/micro-rts/world_action_v2_smoke \
./scripts/collect_world_action_v2.sh
```

Smoke mode uses one map and masked-random policies. It cannot launch the full
collection accidentally.

## Full collection

```bash
./scripts/collect_world_action_v2.sh
```

The defaults use disjoint seeds `11001`, `22001`, and `33001`, a 40k/5k/5k
steps-per-lane-per-map allocation (80/10/10), four 16x16 maps, both seats, the
policy/noise ladder, and a 20% counterfactual request rate. Evaluation includes
the matched scripted opponents and evaluation-only `coacAI`.

The launcher refuses to overwrite an existing raw collection. Choose a new
`ROOT` for another immutable lineage.

## Loader views

`MRTSSequenceDataset` and `build_mrts_loader` accept:

- `window_view="all"` for training;
- `window_view="initial_prefix"` for fixed episode starts;
- `window_view="uniform"` for deterministic timeline coverage; and
- `window_view="completion"` for terminal-anchored windows.

Use `shuffle=False` for validation/evaluation. Physical split labels
`validation` and `evaluation` are accepted directly; do not apply `val_frac`
again to separate split files.
