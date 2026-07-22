# Micro-RTS experiment configs

Configs are organized first by training phase and then by model/objective setup.
Pass the path below `src/configs/exp` to `--exp`, without the `.yaml` suffix.
Every config selects orchestration explicitly with `trainer.type`; `model.type`
is reserved for architecture selection.

## Layout

- `tokenizer/dreamerv4`: DreamerV4 tokenizer experiments.
- `tokenizer/structured_v2`: structured state and action tokenizers.
- `tokenizer/discrete_v3`: hard product-code state tokenizer and categorical
  action-JEPA pretraining.
- `dynamics/dreamerv4`: DreamerV4 dynamics experiments.
- `dynamics/structured_v2/core`: original structured dynamics objective.
- `dynamics/structured_v2/dreamer4`: structured Dreamer4 objective.
- `dynamics/structured_v2/causal_paired`: causal-paired architecture and action-interface ablations.
- `dynamics/structured_v2/trust_region`: current frozen-router residual world-model setup.
- `dynamics/discrete_v3`: plain causal next-code transformer with no flow matching.
- `rl/ppo`: model-free PPO baselines.
- `rl/dreamerv4`: DreamerV4 RL experiments.
- `rl/structured_v2`: structured world-model RL experiments.
- `rl/smoke`: integration smoke configs.

## Current working world model

The active stable continuation is:

```text
micro-rts/dynamics/structured_v2/trust_region/pretrain_structured_dynamics_v2_causal_paired_action_residual_trust_region_160k_stable_tail
```

It loads the best checkpoint from the initial full-budget attempt as model
weights only, uses a fresh low-LR optimizer, keeps the pretrained action router
frozen, and trains under the residual trust region.

Launch it inside the research container with:

```bash
python src/micro-rts/entrypoints/train_dreamer_dynamics.py \
  --exp micro-rts/dynamics/structured_v2/trust_region/pretrain_structured_dynamics_v2_causal_paired_action_residual_trust_region_160k_stable_tail
```

The three configs in `dynamics/structured_v2/trust_region` intentionally remain
together: the successful 40k baseline, the high-LR 160k attempt, and its stable
weight-only continuation.

## Parallel discrete MicroRTS path

The discrete-v3 path is additive. It does not modify or replace the continuous
Dreamer-4/structured-v2 model family. Its three checkpointed phases are:

```text
train_discrete_tokenizer.py
  -> train_discrete_action_tokenizer.py
  -> train_discrete_dynamics.py
```

The tokenizer screen holds the 8x8 grouped spatial topology fixed and compares
only product-code capacity. Both runs pin the same corpus and use the same seed,
batch size, optimizer, schedule, field losses, trajectory split, and fixed
validation sampling:

| Matrix | Config suffix | Spatial slots | Product code | Base codes |
|---|---|---:|---:|---:|
| A | `tokenizer_v3` | 8x8 grouped | 4x512 | 272 |
| B | `tokenizer_v3_compact_shallow` | 8x8 grouped | 2x1024 | 136 |

The A/B comparison isolates product depth/capacity at fixed topology. Advance
only candidates with zero OOVs and near-perfect held-out exact frame, global,
raster, and complete-round-trip metrics. Action-JEPA and dynamics are downstream
promotion stages, not part of this first screen.

The action phase keeps one exact sparse tuple per issued event and transfers its
complete categorical next-code router. The dynamics phase trains the sequence
`[current codes, action events, BOS_NEXT, next codes]` with teacher-forced
cross-entropy, balanced changed/unchanged code losses, and a paired
counterfactual preference margin. It has no target noise, flow signal, shortcut
step, latent normalization, or numerical integration.

Before dynamics constructs its optimizer, it audits the transferred action
prior on paired rows from the live corpus. The default run requires at least one
paired effect code, at least 0.5 bidirectional factual/counterfactual preference,
and zero action overflow. The audit is printed, logged at step zero, and stored
in the dynamics checkpoint as `step_zero_geometry`; a checkpoint that misses a
gate stops before training. `--smoke` intentionally omits the pretrained action
router and verifies execution only, so smoke checkpoints are not representation
certificates.
