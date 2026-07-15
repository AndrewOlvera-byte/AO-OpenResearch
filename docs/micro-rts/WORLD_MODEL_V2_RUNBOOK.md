# MicroRTS world model v2 runbook

This is the operational companion to
[`WORLD_MODEL_V2_REPRESENTATION.md`](WORLD_MODEL_V2_REPRESENTATION.md). It covers
complete-state collection, the structured tokenizer, causal flow/shortcut
dynamics, audits, and checkpoint promotion.

## 1. Apply the engine patch

```bash
docker exec ao-research bash /workspace/infra/microrts-jar-patch/apply_patch.sh
```

The patch exports complete current/terminal state and cloned counterfactual
transitions. Reapply it after rebuilding or replacing the installed environment.

## 2. Collect the v2 corpus

Old v3/v3.1 files cannot train v2 because timers, global resources, targets, and
exact arrival state were never stored. Recollection is required. The collector
defaults to complete state and a 15% cloned-engine counterfactual branch rate.

The recommended larger four-map collection is packaged as:

```bash
./scripts/collect_wm_v2.sh
```

It collects 3.84M real transitions plus approximately 768k paired branches by
default. The explicit command below remains the smaller single-map alternative.

```bash
docker exec -i ao-research bash -lc 'cd /workspace && \
  python src/micro-rts/collectors/offline_data/collect_mrts_data.py \
    --name wm_v2_pretrain \
    --num-envs 24 \
    --policy-device cuda \
    --counterfactual-frac 0.15 \
    --plan mode=selfplay,policy=checkpoints/base_rlFS_expert_masked_league/best.pt,eps=0.05,steps=9000 \
    --plan mode=selfplay,policy=checkpoints/base_rlFS_expert_masked_league/best.pt,eps=0.15,steps=12000 \
    --plan mode=selfplay,policy=checkpoints/base_rlFS_expert_masked_league/best.pt,eps=0.30,steps=6000 \
    --plan mode=selfplay,policy=masked_random,steps=3000 \
    --plan mode=bot,policy=checkpoints/base_rlFS_expert_masked_league/best.pt,steps=12000,seats=mix \
    --plan mode=bot,policy=checkpoints/base_rlFS_expert_masked_league/best.pt,eps=0.15,steps=9000,seats=mix \
    --plan mode=bot,policy=checkpoints/base_rlFS_expert_masked_league/best.pt,eps=0.30,steps=3000,seats=mix \
    --plan mode=bot,policy=masked_random,steps=6000'
```

Each selected counterfactual row samples an alternative legal self action from
the same departure state, holds the realized opponent action fixed, clones the
Java `GameState`, and records the alternative one-tick arrival. Live games are
not modified.

## 3. Audit before training

```bash
docker exec -i ao-research bash -lc 'cd /workspace && \
  python src/micro-rts/entrypoints/audit_structured_dataset.py \
    --data "/data/micro-rts/wm_v2_pretrain__*.h5"'
```

For the default deterministic unit table, unexplained contradiction rate must be
zero. Also inspect counterfactual action/effect rates, maximum entity count,
timer ranges, and terminal departure/arrival ticks. The collected corpus peaks
at 80 occupied units, so the configured capacity is 128. Raw JVM unit IDs are
canonicalized out of Markov equivalence while remaining available in HDF5.

Corpora collected before the explicit-`TYPE_NONE` action patch need a one-time
action-only migration before dynamics training. It does not alter states or
tokenizer inputs. Preview and then apply it with:

```bash
docker exec -i ao-research bash -lc 'cd /workspace && \
  python src/micro-rts/entrypoints/migrate_structured_none_actions.py \
    --data "/data/micro-rts/wm_v2_pretrain__*.h5"'

docker exec -i ao-research bash -lc 'cd /workspace && \
  python src/micro-rts/entrypoints/migrate_structured_none_actions.py \
    --data "/data/micro-rts/wm_v2_pretrain__*.h5" --write'
```

Future collection requires reapplying the patched JAR; it writes the marker at
collection time and needs no migration.

## 4. Train the compressed tokenizer

```bash
docker exec -i ao-research bash -lc 'cd /workspace && \
  python src/micro-rts/entrypoints/train_dreamer_tokenizer.py \
    --exp micro-rts/pretrain_structured_tokenizer_v2'
```

The default latent frame is 64 learned 8x8 spatial tokens, up to 128 unpooled
unit/assignment tokens, and three globals: 195 tokens of width 128. Acceptance
depends on occupied unit type/owner, assignment type/target/remaining ETA,
player resources, and legality-mask metrics. Legacy-plane accuracy is secondary.

## 5. Train the oracle compression ceiling

```bash
docker exec -i ao-research bash -lc 'cd /workspace && \
  python src/micro-rts/entrypoints/train_dreamer_tokenizer.py \
    --exp micro-rts/pretrain_structured_tokenizer_v2_oracle'
```

The oracle retains all 256 spatial tokens plus 128 entity and three global
tokens (387 total). It measures
whether failures belong to 8x8 compression; it is not the intended final model.

## 6. Pretrain the multi-event action tokenizer

```bash
docker exec -i ao-research bash -lc 'cd /workspace && \
  python src/micro-rts/entrypoints/train_action_tokenizer.py \
    --exp micro-rts/pretrain_structured_action_tokenizer_v2'
```

This keeps one 512-D token for each of the 32 sparse event slots. It does not
pool or compress either the 195 state tokens or the action set. The frozen state
tokenizer supplies before/after latents; exact event decoding, forward latent
delta prediction, inverse action inference, and cloned-branch effect prediction
jointly train the action interface. The exported checkpoint contains the event
encoder and learned slot positions consumed by dynamics, while the SSL heads are
discarded downstream.

## 7. Train causal flow/shortcut dynamics

```bash
docker exec -i ao-research bash -lc 'cd /workspace && \
  python src/micro-rts/entrypoints/train_dreamer_dynamics.py \
    --exp micro-rts/pretrain_structured_dynamics_v2'
```

Each sample is a conventional causal token sequence:

```text
[complete departure latent,
 sparse self/opponent source->target action events,
 flow signal,
 shortcut step,
 noised arrival latent]
```

Transitions are packed independently (`seq_len: 1`) to enforce the Markov
contract. Autoregressive rollout repeatedly applies this causal transition. A
history sequence is deferred until partial observability, where it is required.

The loss separates empirical flow x-prediction, guaranteed pure-prior prediction
on 25% of examples, field-wise structured grounding, and dyadic shortcut
consistency through the zero-initialized skip head. The tokenizer is frozen and
flow uses stored per-channel latent statistics.

For the paper-faithful Dreamer-4 objective A/B experiment, run:

```bash
docker exec -i ao-research bash -lc 'cd /workspace && \
  python src/micro-rts/entrypoints/train_dreamer_dynamics.py \
    --exp micro-rts/pretrain_structured_dynamics_v2_dreamer4'
```

That experiment partitions batch rows between empirical and bootstrap training,
uses the `0.9 * tau + 0.1` empirical ramp and `(1 - tau)^2` bootstrap rescaling,
and removes the custom forced-prior and decoder-grounding terms. Its reported
training values are 100-step means, and validation reuses fixed HDF5-local
batches and random draws so checkpoints are directly comparable.

For the R1 action-pretraining ablation, run the matched scratch control and then
the frozen-pretrained treatment. Both use the direct causal-paired objective,
the same 50M transition core, 32 action tokens, and a 160k-step budget:

```bash
docker exec -i ao-research bash -lc 'cd /workspace && \
  python src/micro-rts/entrypoints/train_dreamer_dynamics.py \
    --exp micro-rts/pretrain_structured_dynamics_v2_causal_paired_action_scratch'

docker exec -i ao-research bash -lc 'cd /workspace && \
  python src/micro-rts/entrypoints/train_dreamer_dynamics.py \
    --exp micro-rts/pretrain_structured_dynamics_v2_causal_paired_action_pretrained'
```

Do not compare the pretrained treatment only to the legacy summed encoder: the
scratch factorized run separates architectural benefit from SSL initialization.
Both configs package the successful v1/v2/v3 learning-rate envelope into one
resumable staged scheduler (`1e-4 -> 1e-5`, smooth restart toward `3e-5`, then a
final `1e-5 -> 2.5e-6` tail), avoiding manual checkpoint handoffs.
Compare fixed-validation paired-CF MSE, effect cosine/norm, shuffled-action gaps,
changed-cell/effect F1, and steps or wall-clock time to each gate.

## 8. Evaluate

```bash
docker exec -i ao-research bash -lc 'cd /workspace && \
  python src/micro-rts/entrypoints/eval_structured_dynamics.py \
    --checkpoint checkpoints/structured_dynamics_v2.pt \
    --data "/data/micro-rts/wm_v2_pretrain__*.h5" \
    --flow-steps 4'
```

- `latent_mse` must beat `copy_mse`, especially on event rows.
- `present_acc` and occupied `type_acc` expose disappearing/mutating units.
- `changed_f1` is the primary sparse-mechanics metric.
- `exact_cell` requires every canonical field at a cell to match.
- Positive `self_cf_gap` means shuffled self actions worsen prediction.

The next evaluator extension should consume stored paired counterfactuals and add
autonomous 10/50/250-tick rollouts.

## 9. Implementation map

- Engine export/branching: `infra/microrts-jar-patch/JNIGridnetVecClient.java`
- Python environment: `environments/microrts_env.py`
- HDF5 v4 writer/collector: `collectors/offline_data/`
- Schema/action events: `models/dreamer_v2/schema.py`
- Action tokenizer SSL: `models/dreamer_v2/action_tokenizer.py`
- Hybrid tokenizer: `models/dreamer_v2/tokenizer.py`
- Causal flow/skip dynamics: `models/dreamer_v2/dynamics.py`
- Training/evaluation/audit: `entrypoints/*structured*`
- Tests: `tests/test_structured_world_model_v2.py`

## 10. Promotion order

1. Schema audit passes with zero contradictions.
2. Oracle tokenizer reconstructs complete fields.
3. Compressed tokenizer approaches the oracle on transition-relevant fields.
4. One-step dynamics beats copy-last on events and paired interventions.
5. Flow sampling remains valid through long action timers.
6. Connect the latent to control.
7. Introduce fog-of-war belief and opponent action prediction.

Do not reuse v4.5 dynamics with this representation. The PPO checkpoint remains
useful for collection; the old visual tokenizer remains useful for legacy
experiments, but neither initializes the new structured tokenizer cleanly.
