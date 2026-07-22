# Archived (superseded) code

Legacy DreamerV4 and discrete-v3 **leaf entrypoints and tests**, preserved for
reference but out of the active tree. Not collected by pytest (`testpaths`
targets `src/micro-rts/tests`) and not imported by any live code.

- `entrypoints/` — DreamerV4 / discrete-v3 train + eval scripts.
- `tests/` — DreamerV4 / discrete-v3 world-model tests.

**Note on the models:** `models/dreamer/` (DreamerV4 world model) and the
`discrete_*` modules inside `models/dreamer_v2/` were intentionally **left in
place**, not archived: the preserved structured_v2 entrypoints
(`train_dreamer_tokenizer.py`, `train_dreamer_dynamics.py`) still import
`models.dreamer`, and `models/dreamer_v2/__init__.py` imports the discrete
modules. Cleanly separating them requires refactoring the kept structured_v2
path first.

The active architecture is registry + `PretrainTrainer` subclasses (see
`registry_imports.py`, `trainers/`, `entrypoints/pretrain.py`).
