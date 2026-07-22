# Incomplete-Information Flow World Model for MicroRTS

Status: initial implementation scaffold validated on 2026-07-19

This document revises the incomplete-information world-model proposal around three transformer modules:

1. a roughly 15M-parameter ego-observation and self-action tokenizer stack;
2. a roughly 30M-parameter causal belief/opponent-intent transformer, including its training decoders;
3. a roughly 100M-parameter flow-matching dynamics transformer based on the current best dynamics architecture.

The central idea is to retain the successful full-state representation and exact categorical decoder as privileged teachers while learning a deployable model whose inputs are only fog-of-war observations and the ego player's own action history. The recurrent state of an RSSM is replaced by a causal transformer over local history. Flow matching represents the irreducible distribution over hidden state and opponent behavior instead of forcing that uncertainty into a deterministic average.

The target deployment factorization is

\[
e_t = E_{ego}(o^{ego}_t, m_t),
\]

\[
(b_t, c_t) = H_\phi(e_{\leq t}, a^{self}_{<t}, d_{\leq t}),
\]

\[
z^{opp}_t \sim q_\psi(z \mid c_t),
\]

\[
b_{t+1} \sim p_\theta
  \left(b_{t+1} \mid b_t, a^{self}_t, z^{opp}_t\right).
\]

Here, `o_ego` is the 27-channel fog observation, `m` is the explicit visibility mask, `d` contains boundary and optional public event tokens, `b_t` is a history-conditioned belief state, and `z_opp` is a sampled opponent-intent/short-plan latent. Neither `e_t` nor the raw fog observation is assumed to be Markov.

## Existing assets to preserve

The proposal is deliberately compatible with the strongest existing stack:

- Base dynamics configuration: `src/configs/exp/micro-rts/paper/dynamics/pretrain_flow_dynamics_medium_jepa_exact_overshoot_200k.yaml`
- Full-state tokenizer and exact grouped categorical decoder
- Existing self/opponent action tokenizer and spatial action router
- Existing flow-matching residual dynamics formulation
- Existing JEPA target/predictor machinery and short rollout overshooting
- Augmented dataset: `data/micro-rts/wm_v2_pretrain__20260713-030227__5a273944__fog_v1.h5`
  - privileged full observation;
  - both players' actions;
  - `ego_obs`: `[N, 27, 16, 16]`, `uint8`;
  - `ego_visibility`: `[N, 1, 16, 16]`, `bool`.

The frozen full-state modules define a stable semantic coordinate system. New local-information modules should learn to predict distributions in that coordinate system rather than inventing an unrelated latent space immediately.

## Information and action timing

The model must have an explicit decision-time contract.

Before action selection at time `t`, the history transformer may consume:

- ego observations and visibility masks through `t`;
- ego actions through `t - 1`;
- rewards, terminal flags, elapsed-time features, and other genuinely observable events through `t`.

It must not consume either player's action at `t`. It produces `b_t` and the distribution over `z_opp,t`. Once the agent chooses `a_self,t`, the dynamics model consumes `(b_t, a_self,t, z_opp,t)` to generate the next belief/world hypothesis.

This distinction matters both scientifically and operationally. Allowing the current ego action into the pre-decision intent encoder can leak policy-specific information about hidden state, especially because the existing trajectories were generated under full information.

## Module A: ego and self-action tokenizers, approximately 15M parameters

### Role

The ego tokenizer converts one incomplete observation into stable local tokens. It should represent what is currently visible and what is explicitly unknown; it should not be asked to infer all hidden units from one frame. Temporal inference belongs in the history transformer.

A separate self-action tokenizer converts the ego player's known structured action into spatial action/event tokens. It keeps the same exact structured-action contract used by the successful full-information system. This is a new deployable tokenizer, although it should be initialized from or distilled against the existing action tokenizer wherever shapes permit. It is not the opponent-intent model: it represents a concrete known action, while `z_opp` represents a distribution over unobserved opponent plans.

### Ego-observation interface

Input:

- the same 27 grouped observation channels used by the existing tokenizer;
- one separate binary visibility plane;
- optional fixed player-perspective and map-coordinate embeddings.

Output:

- 64 spatial tokens at width 320, matching the current 2x downsampled spatial geometry;
- 4 to 8 summary/register tokens describing visibility, economy, and global local-observation context;
- an explicit visibility embedding attached to every spatial token.

The ego tokenizer should not emit 128 apparently concrete hidden-entity tokens. Doing so would conflate observation encoding with belief inference. The history transformer will create the full-compatible belief/entity tokens.

### Self-action interface

Input:

- the ego action's source/unit mask;
- action type and all exact categorical parameters;
- the current visible spatial/entity tokens needed to route the action;
- a fixed `self` role embedding.

Output:

- spatially routed action tokens compatible with the dynamics transformer;
- compact issued-action event tokens for the causal history transformer;
- exact action-decoder logits used during tokenizer pretraining.

The action decoder should reconstruct source, type, direction/target, production type, and other parameters using the existing masks and categorical conventions. The same latent action schema can receive decoded hypothetical opponent actions later, with an `opponent` role embedding, but privileged opponent actions are never inputs to the deployable history encoder.

### Candidate scale

Treat 15M as the target for the complete deployable representation front end: ego encoder plus self-action encoder/router. Allocate most capacity to the ego transformer and retain a smaller structured action transformer/router. A practical ego starting point is width 320, 8 attention heads, MLP ratio 4, and approximately 5 to 6 transformer blocks; exact depth must be reduced as necessary to leave room for the action tokenizer. The final configuration must be selected by the repository's parameter-count tool rather than by the estimate in this document.

### Pretraining objectives

Use the same observation-to-latent contract that has worked for the full-state tokenizer:

1. **Exact grouped reconstruction.** Decode the 27 ego-observation groups with the existing categorical semantics. Decode visibility separately with a binary loss.
2. **Visible-region teacher alignment.** At visible cells, align projected ego tokens with stop-gradient tokens from the frozen full-state tokenizer.
3. **Masked spatial JEPA.** Mask visible patches or entities in the online branch and predict their EMA target embeddings with a disposable predictor.
4. **Light temporal consistency.** Adjacent ego frames may be used as a secondary JEPA target, but long-range hidden-state inference should remain the job of Module B.

Train the self-action tokenizer alongside or immediately after this with:

1. exact structured action reconstruction;
2. action-to-consequence JEPA against the frozen full-state transition target;
3. distillation/alignment to the existing successful action-token geometry;
4. source and legality masking identical to the current action path.

The reconstruction decoder is retained for diagnostics and may be reused for observation-space consistency losses. The JEPA predictor is training-only.

### Why reconstruction and JEPA both belong here

Reconstruction preserves the exact feature meanings needed by downstream decoders. JEPA prevents the representation from becoming merely a lossless local code and encourages useful semantic structure. The visible-region teacher target ties the new tokenizer to the already successful full-state latent basis without pretending that hidden cells have observed labels.

## Module B: causal belief and opponent-intent transformer, approximately 30M parameters

### Transformer state instead of an RSSM state

An RSSM summarizes history with a recurrent deterministic state. Here, a causal transformer performs the same integration over a bounded local history:

\[
c_t = H_\phi([e_{t-K+1}, a^{self}_{t-K+1}, \ldots, e_t]).
\]

The transformer uses block-causal masking: tokens inside one time step may communicate, while a time step may attend only to the present and past. At inference, a KV cache makes the update incremental rather than recomputing the entire window.

Initial context should be 32 or 64 environment steps, with context length treated as an ablation. Episode-boundary tokens are mandatory, and cached state must be reset at `is_first`.

### Proposed architecture and outputs

A useful initial scale is width 512, 8 attention heads, MLP ratio 4, and roughly 8 transformer blocks. Eight standard blocks are approximately 25M parameters; input projections, latent queries, flow heads, and training decoders should place the complete module near 30M.

At each time step it outputs:

- `b_t`: 195 belief tokens compatible with the privileged full-state latent geometry;
  - 64 spatial tokens;
  - 128 entity/belief query tokens;
  - 3 global tokens;
- `c_t`: one or more compact history/context tokens;
- parameters/conditioning tokens for the opponent-intent flow;
- optional uncertainty or existence logits for hidden entity slots.

The 195-token geometry is a compatibility interface, not a claim that the model knows one exact full state. Samples conditioned on `c_t` represent possible full-state hypotheses.

### `z_opp` should be a distribution over short opponent plans

A single deterministic vector trained to regress the realized opponent action will collapse ambiguous futures. Instead, define a privileged training target from an opponent action/event window:

\[
y^{opp}_t = T_{opp}
  \left(s_t, a^{opp}_{t:t+H}, s_{t+1:t+H}\right),
\]

where `T_opp` is a frozen or EMA target encoder. It may reuse the existing action tokenizer and full-state tokenizer. The deployable intent flow is conditioned only on `c_t`:

\[
z_0 \sim \mathcal{N}(0,I), \qquad
\frac{d z_\tau}{d\tau} = v_\psi(z_\tau, \tau \mid c_t),
\qquad z^{opp}_t = z_1.
\]

Conditional flow matching trains `v_psi` to transport noise to the privileged plan-latent distribution. A small number of intent tokens, such as 4 to 8 tokens at width 512, is preferable to one vector because plans contain multiple units, targets, and time scales.

At deployment, privileged data and the target encoder disappear. The local history produces the conditioning context, and sampling the intent flow produces one coherent opponent hypothesis. Multiple samples expose strategic uncertainty to imagination.

### Opponent action targets must be event-aware

MicroRTS actions are sparse and asynchronous. Per-tick raw action prediction can be dominated by inactive/no-op entries and long-duration assignments. The training target should include both:

- the dense per-tick action/assignment representation required by the existing dynamics interface;
- a compact event sequence containing newly issued actions, source unit/cell, action type, parameters or destination, and time to the next event.

Recommended prediction horizons are `0, 1, 2, 4, 8, 16`, plus next-event prediction inside the context window. All losses must mask nonexistent units and distinguish "no new command" from "unknown because unobserved."

### Training decoders and heads

The roughly 30M budget includes these training-time heads/decoders:

1. **Ego reconstruction decoder** for current and optionally next ego observation plus visibility.
2. **Privileged state decoder** that maps belief samples through a small adapter into the frozen full-state exact decoder.
3. **Opponent action decoder** for current and future action distributions.
4. **Opponent event decoder** for next-event time, source, action type, and action parameters.
5. **Temporal JEPA predictor** for future frozen full-state and action target embeddings.

The full-state tokenizer/decoder and target action encoder are frozen teachers and are not counted in the 30M deployable/training-head budget. Every parameter report should separately show deployable parameters, disposable training parameters, and frozen teacher parameters.

### Losses

The Module B loss can be organized as

\[
\mathcal{L}_B =
\lambda_{flow}\mathcal{L}_{intent\_flow}
+ \lambda_{ego}\mathcal{L}_{ego\_recon}
+ \lambda_{state}\mathcal{L}_{priv\_state}
+ \lambda_{act}\mathcal{L}_{opp\_actions}
+ \lambda_{event}\mathcal{L}_{opp\_events}
+ \lambda_{jepa}\mathcal{L}_{temporal\_JEPA}
+ \lambda_{vis}\mathcal{L}_{visible\_consistency}.
\]

The terms have different purposes:

- `intent_flow` prevents averaging over incompatible opponent plans;
- `ego_recon` protects the local observation contract;
- `priv_state` teaches hidden-state inference from privileged labels;
- `opp_actions` and `opp_events` force intent to be behaviorally meaningful;
- `temporal_JEPA` makes the context predictive beyond exact reconstruction;
- `visible_consistency` ensures every sampled hypothesis agrees with current evidence.

For hidden privileged state, deterministic MSE should not be the only objective. Use categorical decoding likelihood for discrete state and flow-based samples for ambiguous latent content. Penalize a hypothesis strongly when it contradicts visible cells, but allow multiple explanations behind the fog.

### Preventing `z_opp` from becoming unused

Powerful dynamics transformers can ignore a weak conditioning latent. Use all of the following:

- decode opponent future actions directly from `z_opp`, not only from `c_t`;
- randomly drop or bottleneck direct history-to-dynamics conditioning during training;
- include a `z_opp` shuffle test in every validation run;
- contrast matched intent/history pairs against mismatched pairs;
- measure conditional rollout degradation when `z_opp` is zeroed, shuffled, or sampled from another episode;
- keep `z_opp` small enough to be interpretable but large enough for multi-unit plans.

The aim is not to maximize mutual information with every privileged variable. The aim is to retain information that changes the distribution of opponent actions and future states.

## Module C: 100M flow-matching belief dynamics transformer

### What this dynamics model predicts

Under partial observability, the raw observation is not a Markov state. The dynamics model should operate on the history-conditioned belief state and a sampled opponent plan:

\[
p_\theta(b_{t+1} \mid b_t, a^{self}_t, z^{opp}_t).
\]

More concretely, training targets can be frozen privileged next-state latents, while conditioning comes from local information. A flow sample is therefore a coherent next full-world hypothesis. Its exact decoder output must agree with the actual next full state during supervised training and with the next ego observation on visible cells.

This is not merely `p(o_ego,t+1 | o_ego,t)`. It is a generative filter plus mechanics model: history constructs a belief, `z_opp` chooses one plausible opponent future, and flow matching advances the corresponding world hypothesis.

### Reuse the proven dynamics formula

Start from the current medium JEPA/exact/overshooting configuration:

- width 512;
- 28 transformer blocks;
- 8 attention heads;
- MLP ratio 4;
- causal paired tokenization;
- residual flow prediction;
- exact categorical decoding;
- short rollout overshooting.

This is already close to the desired 100M scale. Preserve the architecture and as much checkpoint-compatible structure as possible. Add projections for the belief and intent tokens rather than redesigning the transformer core immediately.

At a single transition, the conditioning sequence should contain:

- current belief spatial/entity/global tokens `b_t`;
- encoded selected ego action `a_self,t` using the new self-action tokenizer/router;
- sampled opponent intent tokens `z_opp,t`;
- optionally, decoded or softly routed opponent action tokens derived from `z_opp,t`;
- flow time/noise tokens and episode timing features.

The prediction target is the next privileged-compatible latent. The frozen exact decoder supplies categorical state loss. A deterministic fog projection of a decoded world sample supplies an ego-observation consistency loss.

### Opponent intent enters mechanics through actions first

The first implementation deliberately exposes `z_opp` to mechanics only by decoding opponent action-event tokens and passing those through the existing action encoder/router. This preserves the successful mechanics interface and prevents the model from learning a direct intent-to-next-state shortcut that bypasses legal actions. Direct intent cross-attention remains a later ablation, to be admitted only if the decoded-action path has measurable conditional effect and direct conditioning improves multi-step prediction without violating action-mediated mechanics.

### JEPA inside dynamics training

Retain the existing JEPA target formulation, but change the conditioning side to local belief and sampled intent. Predict frozen or EMA privileged future latents at multiple horizons. Keep the exact decoder loss as the semantic anchor.

The desired combination is:

- reconstruction/exact categorical decoding for state fidelity;
- conditional flow matching for multimodal next-state hypotheses;
- JEPA future-latent prediction for long-horizon predictive structure;
- short overshooting for rollout stability.

No one term substitutes for the others.

## Recommended staged training

### Stage 0: lock and audit teachers

- Select the exact full-state tokenizer, decoder, action encoder, and dynamics checkpoint.
- Freeze them and record hashes, parameter counts, and normalization/category metadata.
- Verify the augmented fog dataset preserves the original full observations and actions bit-for-bit.

### Stage 1: pretrain the ego tokenizer

- Train exact ego reconstruction plus visible-region teacher alignment.
- Add masked spatial JEPA once reconstruction is stable.
- Freeze the accepted encoder for the first Module B experiments.
- Retain its reconstruction decoder for audits.

### Stage 2: pretrain belief and opponent intent

- Train the causal history transformer using local inputs only.
- Use privileged full state and future opponent actions only in stop-gradient targets and training decoders.
- Train reconstruction and future-action/event heads jointly with intent flow matching and temporal JEPA.
- Validate strict causal masking with intentional future-data corruption tests.

At the end of this stage, the deployable pair `(E_ego, H_phi + intent flow)` must infer calibrated belief samples and opponent action distributions without any privileged input.

### Stage 3: adapt the 100M dynamics model

- Initialize from the strongest existing dynamics checkpoint where tensor shapes permit.
- Freeze all tokenizers, the belief/intent module, and privileged teachers initially.
- Train only new adapters and newly introduced conditioning paths first.
- Then train the 100M dynamics transformer at a conservative learning rate.
- Finally, optionally unfreeze the history/intent module at a much lower learning rate so dynamics gradients can improve predictive sufficiency.

The privileged full-state tokenizer, exact decoder, and target encoders remain frozen throughout. Do not freeze the 30M history/intent module forever by definition: its initial pretraining establishes the contract, while a controlled final joint phase determines whether task-aligned improvement is real or merely latent drift.

### Stage 4: Dreamer-style imagination

For each imagined step:

1. obtain the current history-conditioned belief;
2. sample one or more `z_opp` hypotheses;
3. choose an ego action from the actor;
4. flow-sample the next world/belief latent;
5. decode reward, continuation, and optional state diagnostics;
6. append the imagined observation/action event to the cached history.

Actor and critic should consume belief/context tokens and may receive uncertainty summaries over multiple intent samples. They should not consume privileged teacher latents unavailable at deployment.

## What is frozen, reused, and discarded

| Component | Initial pretraining | Dynamics adaptation | Deployment/imagination |
|---|---:|---:|---:|
| Full-state tokenizer | Frozen teacher | Frozen teacher | Optional diagnostic only |
| Full-state exact decoder | Frozen teacher/decoder | Frozen loss and audit decoder | Optional sampled-state decoder |
| Existing action tokenizer/router | Frozen teacher/initialization source | Frozen teacher | Discarded after compatibility is verified |
| New self-action tokenizer/router | Trained | Frozen, then optional low-LR tuning | Used for ego and decoded opponent action events |
| Ego tokenizer encoder | Trained | Frozen, then optional low-LR tuning | Used |
| Ego reconstruction decoder | Trained | Diagnostic/loss | Optional; not required by policy |
| Causal history transformer | Trained | Frozen, then optional low-LR tuning | Used with KV cache |
| Opponent intent flow | Trained | Frozen, then optional low-LR tuning | Used and sampled |
| Privileged opponent target encoder | Frozen/EMA target | Frozen target | Discarded |
| Future-action/event decoders | Trained | Auxiliary loss/diagnostic | Optional; action path may be retained |
| JEPA predictors | Trained | Training only | Discarded |
| 100M dynamics transformer | Not required | Trained/adapted | Used |

## Parameter accounting target

The numbers below are targets, not assertions until exact construction is counted.

| Module | Target | Counting rule |
|---|---:|---|
| Ego + self-action tokenizer stack | ~15M | Deployable encoders/routers; report observation/action training decoders separately and together |
| Belief/intent module | ~30M | History transformer, intent flow, and requested training decoders/heads |
| Dynamics model | ~100M | Flow transformer plus required conditioning adapters |
| Frozen privileged teachers | Report only | Never hide these from total training-memory reports, but do not call them deployable parameters |

Every experiment should log three totals: trainable parameters, total resident parameters during training, and deployable inference parameters.

## Evaluation and acceptance gates

### Ego tokenizer

- exact per-group reconstruction accuracy and NLL;
- visibility precision/recall and exact-mask accuracy;
- visible-region agreement with the full-state teacher;
- reconstruction stratified by unit type, resource count, assignment, and map position.

### Belief state

- visible-cell contradiction rate, ideally zero after categorical decoding;
- hidden unit occupancy/type/resource/assignment NLL;
- calibration and coverage of sampled hidden states;
- last-seen unit tracking as a function of time since visibility;
- full-state decoder accuracy separated into visible and hidden regions.

### Opponent intent

- current and future opponent action NLL, top-k accuracy, and event F1;
- source, action type, target/destination, production type, and next-event-time metrics;
- calibration and sample diversity;
- best-of-N coverage without using best-of-N as the only score;
- performance versus action-frequency, last-action, behavior-cloning, and no-intent baselines.

### Dynamics

- one-step exact categorical state metrics;
- 4/8/16-step decoded rollout metrics;
- visible observation consistency after deterministic fog projection;
- stochastic coverage and calibration across multiple flow samples;
- `z_opp` zero, shuffle, resample, and oracle-target ablations;
- mechanics tests in which only the sampled opponent action path changes;
- episode-boundary and KV-cache reset tests.

## Required ablations

At minimum, run:

1. reconstruction only versus reconstruction plus JEPA;
2. no privileged full-state teacher versus privileged teacher;
3. deterministic intent regression versus conditional intent flow;
4. immediate opponent action only versus multi-horizon action/event targets;
5. no `z_opp`, shuffled `z_opp`, and correctly matched `z_opp`;
6. decoded opponent-action conditioning only versus direct intent only versus both;
7. frozen history/intent module versus final low-learning-rate joint tuning;
8. context lengths 16, 32, and 64;
9. history with and without ego action inputs;
10. existing full-information dynamics initialized versus dynamics trained from scratch.

## Dataset limitation that must remain visible

The fog observations are a faithful deterministic projection of the recorded full states, so recollection is not required for representation and dynamics pretraining. However, the recorded players selected their actions with full-state information. Therefore:

- ego actions may reveal hidden information through the behavior policy;
- opponent behavior is not behavior generated under fog-of-war uncertainty;
- intent metrics measure prediction of these recorded full-information opponents from incomplete observations;
- policy performance under canonical fog still requires careful online evaluation and may eventually justify recollecting policy data.

This does not invalidate the augmented dataset. It makes it especially valuable for privileged-teacher pretraining, while requiring explicit action-history leakage ablations and careful claims about canonical partially observable gameplay.

## Initial recommended experiment

The cleanest first experiment is deliberately staged:

1. train the ~15M ego/self-action tokenizer stack with exact reconstruction, visible teacher alignment, spatial JEPA, exact action reconstruction, and action-consequence JEPA;
2. freeze it and train the ~30M history/intent transformer with a 32-step context, 8 intent tokens, intent flow matching, privileged full-state decoding, and opponent event targets at horizons `0, 1, 2, 4, 8, 16`;
3. verify that matched `z_opp` substantially outperforms shuffled `z_opp` on opponent actions and hidden-state rollouts;
4. initialize the ~100M dynamics from the current best flow checkpoint, train new conditioning adapters, then adapt the full transformer;
5. establish decoded opponent-action conditioning first, then compare direct intent conditioning and the combination as guarded ablations.

This sequence answers the highest-risk question before expensive dynamics training: whether local history can produce a compact, sampleable intent latent that contains information the dynamics model actually uses.

## Design decisions to keep stable in the first implementation

- A transformer with a KV cache replaces the RSSM recurrence.
- The ego tokenizer represents observations; the history transformer represents belief.
- `z_opp` is sampled and multi-horizon, not a deterministic label embedding.
- Privileged modules are teachers and decoders, never deploy-time inputs.
- Exact reconstruction anchors semantics; JEPA shapes predictive structure.
- Flow matching handles both opponent-plan ambiguity and next-world ambiguity.
- The current proven 100M dynamics architecture is adapted rather than discarded.
- Current-time ego action is excluded from the pre-decision intent encoder and included in the subsequent dynamics transition.

## Implemented v1 scaffold

The first code path is additive and leaves structured world model v2 unchanged.

- `models/incomplete_info/ego_tokenizer.py`: 64 fog-aware spatial tokens plus eight readout registers. A register bank replaces a single `<latent>` token because it provides capacity for distinct economy, visibility, unit-tracking, and strategic summaries without changing the continuous-token interface.
- `models/incomplete_info/action_tokenizer.py`: deployable self-action event tokens derived only from the local observation and known ego action.
- `models/incomplete_info/opponent_tokenizer.py`: privileged eight-token plan target built from exact opponent events at horizons `0, 1, 2, 4, 8, 16`.
- `models/incomplete_info/history_flow.py`: block-causal history integration and rectified flow over one joint sample containing 195 hidden-world tokens and eight opponent-plan tokens.
- `models/incomplete_info/losses.py`: exact local reconstruction, visibility BCE, masked EMA-JEPA, teacher alignment, exact action/event reconstruction, action-effect prediction, future-state JEPA, joint conditional flow matching, and exact categorical state grounding.
- `models/incomplete_info/model.py`: freezes observation/action/plan teachers and the proven mechanics checkpoint while training causal history and joint flow.

At the medium defaults, exact parameter counts are 6.49M for the deployable ego tokenizer, 3.92M for the self-action tokenizer, 14.03M for the privileged opponent target tokenizer, and 32.36M trainable parameters for history plus joint flow and the future-state JEPA head. The ego pretraining wrapper is 13.08M including its EMA copy and disposable predictor. Frozen teacher/mechanics parameters are reported separately.

The decision-time input contract is enforced by shifting ego action-event tokens by one step before history attention. Dataset windows are split at every internal `is_first`, and the `observation_mode` config toggle selects either `ego` or the explicit `oracle_full` ablation while keeping the model-facing keys stable.

The v1 joint objective is

\[
\mathcal{L}_{joint} =
\mathcal{L}_{FM}(s^{hidden}, z^{opp} \mid h^{ego})
+ 0.25\mathcal{L}_{exact\ state}
+ 0.50\mathcal{L}_{opp\ events}
+ 0.25\mathcal{L}_{future\ JEPA}.
\]

The flow target is joint rather than two independent heads so hidden-world hypotheses and opponent plans remain correlated. State and plan coordinates are normalized separately. Flow samples are grounded through the frozen exact state decoder, while sampled plan tokens are grounded through the frozen opponent event decoder. The frozen 100M mechanics model is not part of this loss: its next-state transition will consume decoded ego/opponent action events in the rollout adapter, preserving action-mediated causality.

### Entrypoints and starting configurations

The four stages are independently runnable:

```bash
python src/micro-rts/entrypoints/pretrain_incomplete_obs_tokenizer.py \
  --exp micro-rts/paper/incomplete_info/probe_ego_tokenizer
python src/micro-rts/entrypoints/pretrain_incomplete_action_tokenizer.py \
  --exp micro-rts/paper/incomplete_info/probe_self_action_tokenizer
python src/micro-rts/entrypoints/pretrain_opponent_plan_tokenizer.py \
  --exp micro-rts/paper/incomplete_info/probe_opponent_plan_tokenizer
python src/micro-rts/entrypoints/pretrain_incomplete_flow_dynamics.py \
  --exp micro-rts/paper/incomplete_info/probe_joint_flow_dynamics
```

Matching `pretrain_*_medium.yaml` configurations provide the first scaled run. Every entrypoint supports `--set data.observation_mode=oracle_full` for the information-completeness upper bound and `--smoke` for a two-step integration run.

### Learning ladder for opponent latent modeling

1. Accept the ego tokenizer only if JEPA improves future/full-teacher probes without degrading visible grouped reconstruction.
2. Accept the opponent target tokenizer only if eight plan tokens decode multi-horizon events and max-horizon future state better than immediate-action-only and no-plan baselines.
3. Train joint flow and measure event NLL, hidden-state categorical NLL, calibration, and best-of-N coverage under `ego` versus `oracle_full` inputs.
4. Add mandatory zero/shuffle/cross-episode plan interventions. A useful intent latent must change decoded opponent events and action-mediated frozen-mechanics rollouts while leaving unrelated mechanics stable.
5. Implement the rollout adapter: sample joint belief/plan, decode the immediate opponent event, combine it with the selected ego event, and call frozen deterministic mechanics. Do not add a direct plan-to-mechanics path in this stage.
6. Expose multiple coherent plan/world samples and uncertainty summaries to the Dreamer actor. Compare one sample, ensemble summaries, and risk-sensitive aggregation before allowing actor gradients into the history/flow module.
7. Only after these gates, evaluate low-learning-rate joint tuning and the guarded direct-intent conditioning ablation.
