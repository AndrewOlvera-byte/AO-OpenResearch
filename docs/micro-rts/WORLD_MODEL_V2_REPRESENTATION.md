# MicroRTS world model v2: complete structured representation

Status: implementation contract, 2026-07-12.

This design replaces the v1 world model's 27-plane-only transition interface.
It does not replace the working PPO observation path. Its purpose is to learn a
high-fidelity, full-information MicroRTS mechanics model before introducing fog
of war, opponent belief, or imagination RL.

## 1. Design decisions

The agreed baseline uses:

- a privileged, Markov-complete engine snapshot for world-model collection;
- a hybrid representation: dense spatial cells plus global state, with active
  assignments stored on their source units;
- 16x16 input cell tokens and learned 8x8 compressed spatial latent tokens;
- exact field-wise reconstruction heads, including a legacy-observation head;
- sparse source-to-target action event tokens for both players;
- a plain causal transformer across the interleaved state/action sequence;
- Dreamer-4-style flow matching and the shortcut/skip head as first-class
  dynamics objectives;
- structured next-state and state-delta heads as grounding losses;
- a deterministic transition baseline alongside flow sampling, not instead of
  it.

The causal transformer is deliberately less specialized than the existing
alternating spatial/temporal implementation. A conventional causal sequence
makes token order, masks, action alignment, and attention probes easy to inspect.
Efficiency can be recovered after the representation passes mechanics gates.

## 2. Separation of concerns

The system has three representations with different contracts:

1. **Policy observation:** the existing 27-plane tensor and GridNet mask. PPO
   continues to use this unchanged.
2. **Canonical engine state:** lossless structured integers exported by the JAR.
   This is the truth used for collection, contradiction tests, and field losses.
3. **Learned world state:** compressed continuous spatial tokens. These are what
   flow matching predicts and what a later imagination policy will consume.

The old design implicitly treated (1) as if it were (2). V2 makes the boundary
explicit.

## 3. Canonical engine snapshot schema

Every transition stores both departure and arrival snapshots. The departure is
`state_t`; the arrival is `next_state_t`, including the real terminal state even
when Gym-MicroRTS auto-resets internally.

### 3.1 Cell table

`state` has shape `[H*W, 16]`, row-major. Empty cells retain terrain and use zero
or sentinel values for unit fields.

| index | field | encoding |
|---:|---|---|
| 0 | terrain | 0 free, 1 wall |
| 1 | unit present | 0/1 |
| 2 | unit ID | nonnegative engine ID, -1 empty |
| 3 | owner role | 0 neutral, 1 perspective/self, 2 opponent, -1 empty |
| 4 | unit type | engine `UnitType.ID`, -1 empty |
| 5 | hit points | integer, 0 empty |
| 6 | carried resources | integer |
| 7 | active assignment present | 0/1 |
| 8 | active action type | 0--5, -1 absent |
| 9 | direction | engine direction, -1 absent/not directional |
| 10 | target x | absolute board coordinate, -1 absent |
| 11 | target y | absolute board coordinate, -1 absent |
| 12 | produced unit type | `UnitType.ID`, -1 absent |
| 13 | assignment start tick | engine tick, -1 absent |
| 14 | action ETA | total duration in ticks, 0 absent |
| 15 | remaining ETA | `max(0, start + ETA - game_time)`, 0 absent |

Attack actions use their explicit attack location. Directional move, harvest,
return, and produce actions also receive an absolute target computed from source
plus direction. This makes the effect endpoint directly available without
discarding the original direction.

Unit IDs are retained in the canonical record and raw action-event record for
diagnostics only. They are JVM-global allocation counters, so they are excluded
from learned inputs, reconstruction targets, exact-state metrics, and Markov
equivalence. Role plus source coordinate is a complete unit/action binding on a
single-occupancy grid.

### 3.2 Global table

`globals` has shape `[8]`:

| index | field |
|---:|---|
| 0 | game tick |
| 1 | perspective/self player resources |
| 2 | opponent player resources |
| 3 | self resources reserved by active assignments |
| 4 | opponent resources reserved by active assignments |
| 5 | number of reserved board positions |
| 6 | winner in perspective coordinates: -1 none/draw, 1 self, 2 opponent |
| 7 | game-over flag |

Reserved board positions are also recoverable from assignment targets. The count
and per-player reserved resources are exported because action legality depends on
them and they are useful schema-completeness probes.

### 3.3 Perspective convention

Each lane is canonicalized around its Python-controlled player:

- role 1 is always self;
- role 2 is always opponent;
- global resources and winner use the same ordering;
- absolute board coordinates are not rotated or reflected.

Self-play produces two perspective records from the same engine state. This is
intentional data augmentation and preserves the current paired-lane contract.

## 4. Collected transition schema

HDF5 format v4 adds:

```text
state             [S, H*W, 16] int32
next_state        [S, H*W, 16] int32
globals           [S, 8]       int32
next_globals      [S, 8]       int32
action            [S, H*W, 7]  uint8
opponent_action   [S, H*W, 7]  uint8
obs               [S, 27,H,W]  uint8
mask              [S, H*W,79]  uint8
reward, raw_rewards, done, is_first
```

The legacy observation remains for PPO compatibility, tokenizer diagnostics,
and visual rollout rendering. Exact arrival snapshots eliminate the fragile
implicit "next row" target and make terminal transitions ordinary samples.

The first dataset audit groups rows by `(state, globals, effective joint action)`
and requires one unique `(next_state, next_globals)` in the deterministic unit
table. Any contradiction is a collection/schema bug or genuine engine
randomness, not a modeling problem.

## 5. Structured tokenizer

### 5.1 Field embedding

Each of 256 cell records becomes a vector by concatenating or summing bounded
field embeddings:

- categorical embeddings: terrain, present, owner, unit type, assignment
  present, action type, direction, produced type;
- normalized numeric projections: HP, carried resources, assignment age, ETA,
  remaining ETA;
- 2-D sinusoidal absolute coordinates;
- target-relative displacement `(dx, dy)` for active assignments;
- a cell-token type embedding.

Global fields are embedded into two player tokens plus one game token. Numerical
fields use fixed meaningful scales and learned linear projections; no single
global RMS is responsible for all semantics.

### 5.2 Compression

The oracle mode retains all 256 cell tokens. The default compressed mode uses 64
learned 8x8 queries. Each query cross-attends to its corresponding 2x2 cell region
and then participates in global self-attention. This preserves locality while
allowing information exchange. It is learned compression, not stride-convolution
aliasing.

Up to 128 occupied-unit/entity tokens are packed row-major and projected without
spatial pooling. They retain the complete gameplay unit/assignment vector and
source coordinate, excluding the diagnostic raw unit ID. A learned padding token
fills unused slots. Compression is
therefore applied to the board lattice, not to the unit records themselves.

Global tokens remain uncompressed. Consequently one default latent frame is:

```text
[64 spatial latent tokens, 128 entity tokens, 3 global latent tokens]
```

The implementation supports `downsample=1` as the oracle ceiling and
`downsample=2` as the v2 default. A 4x4 latent is a later measured ablation.

### 5.3 Decoder and tokenizer losses

The decoder expands compressed tokens to per-cell features and predicts each
canonical field with its natural loss:

- cross entropy for categorical fields;
- categorical bins or normalized regression for bounded integer fields;
- masked assignment losses only where assignments exist;
- global resource/time/reservation losses;
- an auxiliary legacy 27-plane reconstruction head;
- an auxiliary GridNet legality-mask head.

Tokenizer acceptance is based on exact structured fields and rare-event fields,
not aggregate binary-plane reconstruction. Timer, target, unit type, owner,
occupied-cell, and global-resource metrics are reported separately.

## 6. Action event tokenizer

The collected GridNet raster is converted to a sparse list of issued action
events. A token is emitted only for an effective acting source cell:

```text
(role, unit_id, source_x, source_y, action_type,
 destination_x, destination_y, direction, produced_type, attack_offset)
```

Inactive conditional components are zeroed before conversion. Role uses separate
embeddings for self and opponent. Source and destination receive 2-D positional
embeddings. The raw event retains `unit_id` for diagnostics, but the action
encoder ignores it. The acting unit is bound by `(role, source_x, source_y)` in
the complete departure state; single occupancy makes this unique and stable.

An explicitly issued engine `TYPE_NONE` is a real action because it creates a
10-tick busy assignment. The raw raster stores sentinel 255 in its otherwise
inactive attack-offset component to distinguish it from no issued action. The
sparse converter emits this as an action event with `action_type=0`; the sentinel
itself is not embedded.

The action sequence includes:

- one learned `NO_ISSUED_ACTIONS` token when no unit acts;
- sparse self events;
- sparse opponent events;
- one learned joint-action summary token.

Active assignments are **state**, not new action input. The model therefore sees
the distinction between a unit continuing a 200-tick production assignment and a
new production command issued this tick.

## 7. Causal dynamics sequence

The first implementation packs each Markov transition independently:

```text
[state_t spatial/global tokens,
 joint-action_t event tokens,
 flow-signal_t, shortcut-step_t,
 target/noised-state_t+1 tokens]
```

The causal attention mask allows a target-state token to read all departure
state and joint-action tokens, plus earlier target tokens if autoregressive
within-frame decoding is enabled. It cannot read clean future-state tokens.
Token-type, frame-index, role, and spatial-position embeddings make every route
inspectable.

Packing transitions independently is intentional: a complete state must satisfy
the one-step Markov contract without using history as a repair channel. Open-loop
rollout repeatedly feeds the generated latent through the same causal transition.
An interleaved multi-transition context is a later extension for incomplete
information, where history is genuinely required.

For the first implementation, each transition is packed to a fixed maximum
action-event count with an attention padding mask. The transformer is a standard
pre-norm causal encoder stack. Attention-map probes can answer whether a changed
destination attends to its action event and source unit.

## 8. Flow matching and skip prediction

Flow matching remains part of v2. Clean structured tokenizer latents are
normalized per channel. For arrival latent `z1` and Gaussian source `z0`:

```text
z_tau = (1 - tau) z0 + tau z1
v*    = z1 - z0
```

The transformer predicts clean `z1` (x-prediction), from which the velocity is
derived during Euler integration. Unlike v1, the model receives no clean target
information when evaluating the prior condition, and pure-prior samples receive
an explicit, substantial loss allocation.

The shortcut/skip head predicts the result of a larger integration step from a
smaller-step teacher target. It remains zero-initialized and is trained only
after empirical one-step grounding begins to pass. Metrics separate:

- empirical flow loss;
- pure-prior next-state loss;
- skip/self-consistency loss;
- structured decoded-next-state loss;
- open-loop exact/event metrics.

Flow prediction and structured field prediction share the causal trunk. The
structured head prevents a low-MSE off-manifold latent from being counted as a
correct game state.

## 9. Implementation phases

### Phase A: schema and collection

1. Patch the vector JNI client to export current and terminal full snapshots.
2. Surface snapshots through `MicroRTSVecEnv` behind `full_state=True`.
3. Add HDF5 v4 fields and exact departure/arrival collection.
4. Add schema range, perspective-symmetry, terminal, and Markov contradiction
   tests.

### Phase B: tokenizer

1. Implement field embeddings and oracle/downsample-2 tokenization.
2. Implement structured and legacy/mask decoders.
3. Train on individual v4 snapshots.
4. Gate on timer/target/resource/exact occupied-cell reconstruction.

### Phase C: dynamics

1. Convert raster joint actions to sparse event tokens.
2. Implement the standard causal transformer sequence.
3. Train empirical flow, explicit pure-prior, structured grounding, and delayed
   skip losses.
4. Evaluate paired action interventions and 1/10/50/250-step rollouts.

### Phase D: compression and control

1. Compare oracle 16x16 tokens with compressed 8x8 tokens.
2. Add categorical/VQ latents as a controlled tokenizer variant.
3. Connect the stable latent to the existing actor/critic interfaces.
4. Resume imagination RL only after the mechanics gates pass.

## 10. Acceptance gates

Before dynamics training:

- zero unexplained duplicate-input contradictions;
- lossless terminal arrival snapshots;
- perspective-role symmetry for self-play pairs;
- action event source and destination match engine assignments.

Before compression is accepted:

- near-perfect categorical reconstruction on occupied/assignment cells;
- remaining-ETA and player-resource errors small enough to preserve legality and
  completion timing;
- no material degradation from oracle tokens on one-step mechanics probes.

Before imagination RL:

- generated state is valid under field constraints;
- changed-cell/event F1 beats copy-last decisively;
- paired self and opponent action interventions change the correct endpoints;
- open-loop state does not contract or lose occupancy;
- action completion remains calibrated through 250-tick production horizons.

## 11. Future incomplete-information decomposition

V2 is deliberately privileged. The later system composes:

```text
partial observation history
    -> recurrent belief over canonical v2 state
    -> opponent policy over v2 action events
    -> frozen/finetuned v2 mechanics transition
    -> imagined belief trajectories and control
```

The mechanics representation therefore becomes the supervised target for the
belief model rather than being redesigned when fog of war is introduced.
