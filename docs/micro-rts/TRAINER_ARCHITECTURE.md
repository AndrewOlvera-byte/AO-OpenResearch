# MicroRTS trainer architecture

Training orchestration is selected independently from model architecture:

```yaml
trainer:
  type: structured_v2_dynamics

model:
  type: structured_v2
```

Importing `registry_imports` registers the complete MicroRTS model, loss, and
trainer surface. Unknown or duplicate registrations fail immediately. Training
entrypoints do not contain optimization loops; they parse CLI overrides, load
the experiment, build `trainer.type`, and call `train()` or `smoke_test()`.

## Pretraining trainers

- DreamerV4: `dreamerv4_tokenizer`, `dreamerv4_dynamics`
- Structured-v2: `structured_v2_tokenizer`,
  `structured_v2_action_tokenizer`, `structured_v2_dynamics`
- Discrete-v3: `discrete_v3_tokenizer`,
  `discrete_v3_action_tokenizer`, `discrete_v3_dynamics`
- Incomplete information: ego/self/opponent tokenizers, opponent intent,
  belief dynamics, and joint-flow dynamics
- Causal world-action v1: predictive belief encoder and factorized dynamics

Use `train_tokenizer.py`, `train_action_tokenizer.py`, `train_dynamics.py`,
`train_encoder.py`, or `train_opponent_latent.py` according to the module being
trained. Historical architecture-specific command names remain thin wrappers.

## RL trainers

PPO, DreamerV4, structured Dreamer, and incomplete-belief Dreamer also resolve
through `trainer.type`. Their established training loops and checkpoint formats
are unchanged.
