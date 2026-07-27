# World-Action Model: Reusing the Proven Structured-v2 Objective

## Status and decision

This document is the handoff for the next incomplete-information MicroRTS
world-model iteration.

The central decision is to stop redesigning the successful mechanics objective.
The complete-information `structured_v2` stable-tail model remains the reference
for how to learn dynamics. The incomplete-information model should preserve that
direct, action-sensitive residual objective and change only the information
interface:

1. replace the complete structured-state latent with a predictive belief latent
   inferred from ego observations and history; and
2. replace the privileged current/future opponent-action input with opponent
   intent tokens inferred from the same observable history.

The downstream actor must receive a compact recurrent state that contains the
representations needed for planning under fog. Representation learning and
dynamics learning are staged and frozen at their boundaries so that dynamics
cannot continually move the actor's state space.

For the final experiment lineage, collect a new corpus and retrain the complete
representation chain on its train split before training dynamics and RL. Reusing
the existing checkpoints is valid for a short engineering smoke test, but not
for the final train/validation/evaluation result.

## Research question

Can the proven structured-v2 world-model objective retain its action sensitivity
and rollout quality when its privileged state and opponent-action inputs are
replaced by two inferred token sets?

The two inferred sets have distinct jobs:

- **belief tokens** represent the observable-history posterior over the current
  game state; and
- **opponent-intent tokens** represent a predictive distribution over likely
  opponent behavior.

The intended transition is

```text
predicted_belief_(t+1)
    = belief_t
    + residual_dynamics(
          belief_t,
          issued_self_action_t,
          inferred_opponent_intent_t
      )
```

This is not a transformer that receives the true opponent action during
deployment. Privileged opponent actions and full state are training targets
only.

## Why return to structured-v2

The promoted complete-information model is:

```text
checkpoints/
  pretrain_structured_dynamics_v2_causal_paired_action_
  residual_trust_region_160k_stable_tail/best.pt
```

Its successful properties should be treated as the mechanics baseline:

- direct one-step latent transition rather than iterative flow integration;
- explicit self-action conditioning;
- paired cloned-engine factual/counterfactual transitions;
- residual prediction around a strong persistence/copy prior;
- a copy margin that requires the model to improve over persistence;
- changed-region weighting;
- counterfactual effect magnitude and direction supervision;
- factual/counterfactual preference or ranking;
- semantic event grounding;
- reward and continuation grounding for control;
- zero/residual initialization and a correction trust region;
- short, fast rollout inference suitable for Dreamer.

The earlier factorized/flow incomplete-information dynamics path added too much
movement and too much repeated computation. It made iteration slow without
showing that the additional machinery improved the relevant causal probes. The
new design deliberately returns to a single direct transition.

## Architecture

```text
ego raster o_t + visibility v_t + previous self action a_(t-1)
                              |
                  shared low-level tokenizers
                              |
             +----------------+----------------+
             |                                 |
     predictive-belief encoder         opponent-intent encoder
             |                                 |
       belief_t: 28 tokens              opp_z,t: plan tokens
             |                                 |
             +---------------+-----------------+
                             |
                  issued self action a_t
                             |
             structured-v2-style direct residual
                    dynamics transformer
                             |
                  predicted belief_(t+1)
                             |
          +------------------+-------------------+
          |                  |                   |
      event heads       reward/continue       Dreamer state
```

### Predictive-belief encoder

The belief encoder consumes only information available to the ego player:

- fogged/ego raster observations;
- an explicit visibility mask;
- previous issued self actions;
- reset boundaries; and
- a causal history window.

The current representation layout is 28 tokens:

| Branch | Tokens | Intended content |
|---|---:|---|
| Self | 8 | Own units, assignments, resources, pending effects |
| Opponent | 8 | Belief about hidden/visible opponent state |
| Static | 4 | Map and slowly changing context |
| Interaction | 8 | Contested regions and joint causal effects |

These names are an architectural prior, not proof that every token has the
desired semantics. Promotion probes must verify what is recoverable from each
branch.

### Opponent-intent encoder

The opponent-intent path is separate because it has a different supervised
role. During pretraining, a privileged opponent-plan tokenizer encodes actual
future opponent actions. An intent prior then learns

```text
p(opp_z,t | ego observations <= t, self actions < t).
```

At deployment it sees no privileged state or opponent action. It produces
multiple possible plan modes and their probabilities. Initial dynamics training
may use deterministic top-1 tokens for reproducibility. Dreamer should later
evaluate sampled or marginalized intent modes so that hidden-opponent
uncertainty is not collapsed permanently to one guess.

### Self-action tokens

The issued ego action is known and remains an explicit input. It should be
routed as sparse action-event tokens rather than compressed immediately into
one weak pooled vector. The model must be able to associate source cell, action
type, direction/target, produced unit type, and relevant validity with the
belief-token corrections they cause.

### Direct dynamics

For one transition, the dynamics transformer operates on token sets, not on
four scalar variables per timestep:

```text
queries/state:
    28 current belief tokens

conditioning memory:
    sparse self-action event tokens
    inferred opponent-intent plan tokens

output:
    28 next-belief tokens
```

History is processed by the frozen belief and intent encoders. Dreamer
imagination carries the 28-token belief forward recurrently and should not
rerun the full raw-observation history transformer at every imagined step.

## The two substitutions from complete information

The clean comparison to structured-v2 is:

| Complete-information structured-v2 | Incomplete-information model |
|---|---|
| Structured state tokenizer output | Predictive-belief tokens from ego history |
| True opponent action/action-event tokens | Inferred opponent-intent plan tokens |
| Known issued self action | Same known issued self action |
| Direct residual dynamics | Same objective family |
| Structured semantic targets | Privileged training targets only |

Thus the new model is not intended to invent another mechanics learner. It asks
whether inferred belief and intent representations can replace privileged
information while leaving the successful transition objective recognizable.

## Frozen boundaries and representation drift

During dynamics pretraining, freeze:

- the ego observation tokenizer;
- the self-action tokenizer;
- the privileged opponent-plan tokenizer;
- the predictive-belief encoder; and
- the observable-history opponent-intent encoder.

Only the direct dynamics transformer and its prediction/readout heads train.

This is important because the raster provides sparse information about hidden
state. If the representation changes online while the dynamics model and actor
are adapting, latent targets, reward heads, and policy inputs all move together.
That makes failures hard to identify and can destabilize Dreamer. A later
end-to-end fine-tuning experiment is allowed only as a separate ablation with
slow target encoders and explicit drift probes.

## Why a new collection is required

The old corpus contains exact cloned counterfactual structured arrivals but
does not contain the exact counterfactual Gym raster emitted by the engine. The
patched collector now exports:

- factual full raster and structured state;
- factual ego action and opponent action;
- exact counterfactual self action;
- held-fixed counterfactual opponent action;
- exact counterfactual full arrival state/globals;
- exact counterfactual full arrival raster; and
- counterfactual validity.

Fog augmentation derives:

- `ego_obs`;
- `ego_visibility`;
- `counterfactual_ego_obs`; and
- `counterfactual_ego_visibility`.

This permits the frozen belief teacher to construct the exact posterior target
for an alternative self action:

```text
E_belief(factual history, counterfactual self action,
         counterfactual ego arrival).
```

The new field does not mathematically require retraining the tokenizer because
the raster schema is unchanged. Nevertheless, the final result should retrain
the tokenizer and encoders because the new dataset also introduces a corrected
episode-level split and evaluation protocol. Every learned component must see
train episodes only.

## Corpus design

### Store complete episodes

Collection segments are an I/O detail. The unit of dataset identity and
splitting must be a complete game episode:

```text
episode_id
collection_seed
map_id
opponent_id
ego_policy_id
ego_policy_checkpoint
action_noise
seat
start_row
length
terminal_outcome
```

Do not split rows or overlapping windows randomly. All rows and all
counterfactual branches from one episode belong to exactly one of train,
validation, or evaluation.

Autorestarts must become new episode records. A sequence loader must never
carry history across a reset merely because both episodes were written in one
collection segment.

### Separate train, validation, and evaluation collections

Prefer physically separate immutable manifests/shards:

```text
data/micro-rts/world_action_v2/
    train/...
    validation/...
    evaluation/...
    manifests/
        train.jsonl
        validation.jsonl
        evaluation.jsonl
```

Use disjoint collection seeds. For the strongest generalization claim, reserve
some opponent-policy seeds or variants for evaluation while retaining a
matched in-distribution evaluation stratum.

A reasonable initial episode allocation is:

- 80% train;
- 10% validation;
- 10% evaluation.

The exact percentages matter less than having enough complete games in every
map/opponent/seat/outcome stratum.

### Balance collection dimensions

Each split manifest should report counts and transition totals by:

- map;
- scripted opponent or self-play policy;
- ego seat;
- win/loss/draw;
- game-length bucket;
- policy checkpoint;
- exploration/noise level;
- counterfactual-valid fraction;
- counterfactual state-change fraction; and
- visibility/opponent-contact phase.

Training should include policy-quality and exploration diversity:

- strong PPO/self-play trajectories for realistic planning states;
- controlled epsilon/noise for action coverage;
- both seats;
- all target maps and opponent styles;
- early, middle, and terminal game states;
- some null/inconsequential interventions; and
- enough state-changing counterfactual interventions to train action geometry.

Do not accept a corpus in which most counterfactual alternatives are legal but
causally identical no-ops.

### Validation must represent whole-game prediction

Validation should not be a random set of middle-game rows. Use fixed episode
manifests and report at least four views:

1. **Initial-prefix validation**  
   Begin at the true initial observation and roll the history/belief forward.
   This measures belief construction without an artificially informative
   warm-start.

2. **Uniform transition validation**  
   Deterministically sample transitions across the complete episode timeline.
   This remains comparable to ordinary one-step validation.

3. **Completion-anchored validation**  
   Include windows ending at terminal arrivals, wins/losses, and late economic
   or combat changes. This prevents long games and terminal rows from
   disappearing under row sampling.

4. **Full-episode open-loop/chunked rollout evaluation**  
   Start from initial belief and evaluate recurrent predictions through the
   game, resetting only at true episode boundaries. Report error versus rollout
   horizon and phase.

Validation batches must be fixed by manifest and seed. Model selection must not
depend on whichever HDF chunks happen to be cached.

## Storage and loading plan

### Sharded storage

Use moderately sized HDF5 shards rather than one monolithic file. Shards should
contain complete episodes where practical and expose a global episode manifest.
This provides:

- easier recollection/replacement of a failed shard;
- parallel worker reads without one global HDF lock;
- bounded corruption impact;
- simpler train/validation separation;
- better chunk locality; and
- resumable fog augmentation and auditing.

The precise shard size should be benchmarked, but several GB per shard is a
reasonable starting point. Compression should remain low enough that GPU
training is not decompression-bound.

### Episode-aware indices

Build indices for:

- complete episodes;
- initial-prefix windows;
- terminal/completion windows;
- uniform windows;
- paired-counterfactual rows; and
- episode phase/length buckets.

Every sampled window must carry a correct `is_first` mask and must remain within
one episode.

### Training shuffle

Avoid both extremes:

- fully random row reads destroy HDF locality; and
- reading one trajectory or shard for too long creates highly correlated
  batches.

Use hierarchical shuffling:

1. shuffle shard order per epoch;
2. shuffle episode blocks within a bounded shard/cache working set;
3. sample windows within those episodes;
4. assemble batches across multiple episodes and, where possible, multiple
   maps/opponents;
5. rotate to another chunk working set.

Bucket by sequence length only where required for efficient padding. Do not let
length bucketing collapse a batch to one opponent or one game phase.

The loader should expose explicit batch mixtures rather than hoping random
sampling produces them. For example:

```text
ordinary factual windows             40%
paired state-changing CF windows     30%
initial-prefix windows                15%
completion/terminal windows           15%
```

These numbers are starting points and should be adjusted from corpus audits and
throughput measurements.

### Validation loader

Validation/evaluation loading should:

- use immutable index files;
- disable stochastic shuffling;
- use fixed ordering or fixed seeded permutations;
- cover complete stratified episode sets;
- report the number of unique episodes, not only batch count;
- retain initial and terminal rows;
- cache only as a performance optimization, never as the definition of the
  evaluated sample; and
- produce identical results after restart.

## Full training dependency chain

For the clean final lineage, train every learned component on the new train
split and select it on the new validation split:

```text
new exact-counterfactual corpus
        |
        +--> structured/full-state teacher tokenizer
        |
        +--> ego observation tokenizer
        |
        +--> self-action tokenizer
        |
        +--> privileged opponent-plan tokenizer
        |          |
        |          +--> observable-history opponent-intent prior
        |
        +--> predictive-belief encoder
                   |
                   +--> frozen direct structured-v2 dynamics
                                |
                                +--> frozen-world Dreamer actor/critic
```

Recommended order:

1. Collect and audit train, validation, and evaluation episodes.
2. Apply and audit factual and counterfactual fog projections.
3. Train/revalidate the privileged structured-state tokenizer.
4. Train the ego tokenizer.
5. Train the self-action tokenizer.
6. Train the privileged opponent-plan tokenizer.
7. Train the opponent-intent prior from observable history.
8. Train the predictive-belief encoder.
9. Freeze every representation module.
10. Train direct structured-v2-style dynamics with `opp_z`.
11. Train the matched no-`opp_z` dynamics control.
12. Run representation and rollout promotion probes.
13. Train separate downstream Dreamer actors from each promoted world model.

The ego tokenizer and the two history encoders may share low-level tokenizers,
but the final checkpoints and their training-data manifest hashes must be
recorded explicitly.

## Dynamics objective

The direct objective should include:

### Factual latent prediction

```text
MSE(predicted belief_(t+1), frozen target belief_(t+1))
```

### Persistence/copy gate

The learned transition must outperform copying `belief_t`. Log:

- factual MSE;
- copy MSE;
- copy gain; and
- fraction of samples beating copy.

### Exact counterfactual belief target

For a cloned alternative action, reconstruct its factual-history prefix and
substitute only the alternative self action and exact counterfactual ego
arrival. Use a sparse/common anchor per batch initially so this adds one frozen
teacher pass instead of one pass for every transition.

This is an auxiliary loss at first, not the dominant objective.

### Semantic factual/counterfactual targets

Retain dense structured semantic supervision for:

- changed cells/patches;
- unit/resource/action events;
- factual and counterfactual branches;
- effect-vector magnitude;
- effect-vector direction; and
- factual/counterfactual pairing preference.

### Control heads

Train reward, return/value-supporting, and continuation heads from the predicted
next belief, not from privileged arrival state.

### Stability

Retain:

- residual/zero initialization;
- bounded cosine gradients near zero effect;
- norm calibration;
- correction trust region;
- BF16 AMP;
- flash/scaled-dot-product attention where supported;
- one direct dynamics call for concatenated factual/CF branches; and
- gradient/RMS monitoring per token branch.

## Ablation matrix

The primary final comparison is:

| System | Observation | World model conditioning | Downstream control |
|---|---|---|---|
| PPO | Ego policy input as configured | None | PPO |
| Structured-v2 reference | Complete structured state | Self + true opponent action | Dreamer |
| Incomplete + `opp_z` | Ego/fog history | Belief + self action + inferred opponent intent | Dreamer |
| Incomplete control | Ego/fog history | Belief + self action + null intent tokens | Dreamer |

The conditioned and control dynamics must have:

- identical data;
- identical frozen belief encoder;
- identical architecture and parameterization where possible;
- identical optimizer, schedule, batch composition, and training steps;
- identical Dreamer training; and
- the same intent-computation path if matching wall-clock compute is important,
  with its output replaced by learned null tokens in the control.

## Promotion probes

Do not promote on validation loss alone.

### Representation

- belief-token RMS and per-branch variance;
- hidden full-state linear probes;
- visible versus hidden opponent decoding;
- opponent future-action/plan decoding from `opp_z`;
- intent mode entropy and calibration;
- shuffled-history degradation;
- shuffled-self-action degradation;
- shuffled-`opp_z` degradation;
- sample-specific versus static opponent information;
- map/opponent leakage checks; and
- train/validation/evaluation gaps by episode stratum.

### One-step causal dynamics

- factual belief MSE;
- copy gain;
- exact CF belief MSE;
- semantic factual/CF loss;
- effect cosine;
- effect norm ratio;
- paired preference accuracy/gap;
- action shuffle degradation;
- intent shuffle/null degradation; and
- changed versus unchanged-region error.

### Rollouts

- error at horizons 1, 2, 4, 8, and longer diagnostic horizons;
- initial-prefix full-game curves;
- completion-anchored performance;
- self-state versus opponent-state drift;
- reward and continuation calibration;
- sampled-intent best-of-N and expected performance;
- stability under the actor's action distribution; and
- inference steps/second and VRAM.

### Downstream

- win rate by map/opponent/seat;
- sample efficiency;
- actor performance with true, inferred, shuffled, and null `opp_z` where
  scientifically valid;
- actor sensitivity to intent modes;
- imagination throughput; and
- comparison against PPO and complete-information structured-v2 Dreamer.

## Existing implementation and required work

Already present:

- exact counterfactual observation export in the patched Java client;
- counterfactual observation storage;
- factual/counterfactual fog augmentation fields;
- an episode-sequence dataset abstraction;
- the frozen belief and intent modules;
- direct residual incomplete-information dynamics;
- conditioned and null-intent configs; and
- a sparse exact-counterfactual-belief loss path.

Before collection:

1. Rebuild Docker and the research image after the Docker storage reset.
2. Reapply and verify the MicroRTS jar patch.
3. Add/verify explicit episode IDs and terminal metadata in the writer.
4. Implement sharded manifests and physically separate split destinations.
5. Add deterministic collection-seed and split-manifest hashes.
6. Add collection audits for every balance dimension above.
7. Add fixed initial-prefix, uniform, completion, and full-episode evaluation
   loaders.
8. Benchmark HDF shard size, compression, workers, prefetch, and chunk-aware
   sampling on the RTX 5070 Ti system.

Before full training:

1. Run a small end-to-end collection smoke.
2. Verify exact factual and counterfactual raster/state alignment.
3. Verify no episode crosses split or sequence boundaries.
4. Verify terminal arrivals survive autoreset.
5. Verify validation is deterministic after process restart.
6. Verify the new loader keeps the GPU fed.
7. Train short representation and dynamics screens before committing to full
   budgets.

## Suggested execution stages

### Stage A: infrastructure smoke

- several complete games per map/opponent/seat;
- counterfactual fraction enabled;
- write multiple shards;
- run fog augmentation and full audit;
- iterate every loader/evaluation view; and
- perform a 20-step train smoke for every stage.

### Stage B: pilot corpus

- approximately 10--20% of the intended final episode count;
- full split protocol;
- retrain shortened tokenizer/encoder models;
- train conditioned and null-intent dynamics for 10--20k steps;
- evaluate action/intent shuffle degradation and rollout curves.

Advance only if the direct model:

- beats copy;
- responds materially to self actions;
- uses `opp_z` sample-specifically;
- improves paired counterfactual geometry;
- remains stable over short rollouts; and
- is fast enough for iteration and Dreamer.

### Stage C: final corpus and representation lineage

- collect the complete immutable split;
- freeze manifests;
- train all tokenizer and representation stages from scratch;
- store config, git SHA, dataset-manifest hash, and best/final metrics in every
  checkpoint; and
- prohibit evaluation-set use for checkpoint selection.

### Stage D: dynamics ablation

- train `opp_z` and null-`opp_z` dynamics from scratch;
- select on validation episodes;
- evaluate once on the held-out evaluation episodes; and
- retain identical training budgets.

### Stage E: downstream RL

- freeze the promoted world model;
- train matched Dreamer actors;
- evaluate against PPO and the complete-information structured-v2 reference;
- only afterward consider joint fine-tuning or a shared history trunk.

## Non-goals for this iteration

- No iterative flow-matching dynamics.
- No online updating of the belief or intent encoders during the primary
  dynamics/RL ablation.
- No architecture consolidation before proving that `opp_z` helps.
- No random row-level train/validation split.
- No validation composed only of convenient middle-game windows.
- No use of privileged opponent action or full state during deployment.
- No claim that branch names guarantee semantics without probes.

## Final view

The mechanics problem and the incomplete-information representation problem
should be separated.

Structured-v2 already provides the strongest demonstrated mechanics learner in
this repository. The next experiment should preserve that objective and test
whether two frozen observable-history representations—a predictive belief and
an opponent-intent plan—can replace complete state and privileged opponent
actions. A new episode-complete corpus, exact counterfactual observations, and a
strict episode-level train/validation/evaluation protocol are necessary to make
that result interpretable.

If the conditioned system wins over the matched null-intent control, the next
engineering step is to consolidate duplicated history computation into a
shared temporal trunk with separate belief and intent queries. If it does not,
the immediate bottleneck is representation content or conditioning—not the
structured-v2 mechanics objective.
