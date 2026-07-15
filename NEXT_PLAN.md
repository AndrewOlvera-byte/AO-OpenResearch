# NEXT_PLAN — World Model v3: joint-action dynamics + diverse pretrain corpus

**Date:** 2026-07-10. **Prereq reading:** this doc, then `src/micro-rts/models/dreamer/world_model.py`,
`src/micro-rts/collectors/offline_data/` (esp. `policies.py`, `HDF5Writer.py`, `collect_mrts_data.py`),
`infra/microrts-jar-patch/`.

## Why (diagnosis of the failed hybrid RL run, wandb `oqwvpf7g`)

The hybrid DreamerV4 RL run was degenerate: **eval win_rate = 0.000 at all 22 evals** over 2.15M env
steps while *imagined* return climbed 0.15 → 1.04 and policy entropy rose to ~5.5. The world model was
excellent teacher-forced (recon 0.001, mask_acc 0.999, reward_corr 0.96) but useless generatively:
**open-loop MSE ≈ 3× worse than a copy-last-frame baseline** (`dyn/openloop_mse` ~0.08 vs
`dyn/copylast_mse` ~0.026), `z_motion` ~0.05. The actor maximized reward inside action-insensitive mush.

Two root causes, both **data identifiability**, not architecture bugs:

1. **Actions carry no information in expert-only data.** The v2 corpus is scripted-bot play, so
   `a_t ≈ π_bot(s_t)` — the action tokens add ~zero mutual information beyond the state context, and the
   transformer learns to ignore them. Nothing in the flow objective forces action-dependence. This is the
   model-based analog of causal confusion: WMs trained on narrow/expert data capture the sampling policy's
   state→state correlations instead of the causal action→state mechanism (FOCUS, Springer FCS 2024,
   https://link.springer.com/article/10.1007/s11704-024-3946-y; "Offline vs. Online Learning in MBRL:
   Lessons for Data Collection", arXiv:2509.05735).
2. **The opponent is an unmodeled latent confounder.** MicroRTS is simultaneous-move: true dynamics are
   `z' = f(z, a_self, a_opp)`, but the WM conditions only on `a_self`. It can only learn the *marginal*
   over the data's opponents; few-step flow sampling then averages the multimodal opponent response into
   near-static latents. (Note: no fog of war ⇒ the *state* is fully observed; only the opponent's
   simultaneous action is hidden. This is Markov-game stochasticity, not poker-style hidden information.)

Recent multi-agent world-model work confirms the fix direction: model the other agent's actions/latents
explicitly rather than marginalizing ("Dreaming of Others", arXiv:2605.31361; CoDreamer,
arXiv:2406.13600; diffusion-inspired multi-agent world modeling, arXiv:2505.20922; Simultaneous
AlphaZero for simultaneous-move Markov games, arXiv:2512.12486). MuZero-family systems handle opponents
by putting their moves *in the model*; we do the same for simultaneous moves via joint-action
conditioning.

**Scope decision: do NOT touch the tokenizer.** It works (recon 0.001, mask_acc 0.999) and observations
are unchanged. "Rebuild the obs" means the *dynamics transition inputs* gain the opponent action; the
tokenizer/latent space stays frozen so `latent_scale` and all v2 checkpoints remain comparable.

## Workstream A — Jar patch: expose the scripted bot's action (blocker for everything else)

Today the JNI only surfaces a gridnet action for the Python-controlled player; the native `ai2` never
exposes what it chose (`collectors/offline_data/policies.py` docstring documents this). Patch it exactly
like the terminal_obs patch:

- Edit `infra/microrts-jar-patch/JNIGridnetVecClient.java`. In the step path, `ai2.getAction(...)`
  returns a `PlayerAction` (list of unit→`UnitAction` assignments). Encode it into the same
  `(H*W, 7)` gridnet layout Python actions use (component order `[action_type, move_dir, harvest_dir,
  return_dir, produce_dir, produce_type, attack_offset]`; cells with no acting unit = zeros/NOOP). The
  jar already contains the gridnet⇄PlayerAction decode for player 1 — mirror it in reverse. Expose as a
  new field (e.g. `opponentAction`, `int[][][]`) alongside `terminalObservation`.
- `bash infra/microrts-jar-patch/apply_patch.sh` inside the research container (idempotent, keeps
  `.orig`, `--release 8`). Extend the script's final `javap` smoke check to assert the new field.
- Surface it in `environments/microrts_env.py` as `info["opponent_action"]` (same pattern as
  `terminal_obs`). **Alignment check:** the exposed action must be the one that produced the *returned*
  obs (i.e. same tick as the learner's submitted action), not off by one. Write a test: run a
  deterministic bot (workerRushAI) 2-player game, replay both recorded action streams through the engine
  from the same seed/state, assert observation trajectories match.

Self-play needs no patch: both players are Python-controlled and both actions are already in hand
(`dream_collector.py` selfplay mode) — they just aren't *recorded*.

## Workstream B — Data schema: record both actions + provenance

Extend the offline HDF5 pipeline (`HDF5Writer.py`, `collector.py`, `dataset.py`, `mrts_dataset.py`):

- New per-step dataset `opponent_action` `(rows, H*W, 7)` i8/i16, same layout as `action`.
- New per-trajectory provenance: `policy_id` (player-1 controller) alongside the existing
  `opponent_id`, plus an `action_noise` float attr (ε used, see Workstream C). Keep `map_id`.
  Provenance filters let ablations slice the corpus (e.g. train with/without random traces).
- Bump a format-version attr; `dataset.py` should hard-error on old files rather than silently yielding
  zero opponent actions.

## Workstream C — Collection matrix: vast, *coherent* transition diversity

The corpus must make both action channels informative. Diversity of **policies**, not just states —
mixed-quality + noise-injected data is what makes dynamics identifiable (arXiv:2509.05735; causal
action-influence data augmentation, arXiv:2405.18917). Player-1 controllers to run (extend
`policies.py` / `collect_mrts_data.py`):

1. **Trained PPO gridnet checkpoints** (`base_rlFS_expert_masked_league` runs) — coherent, winning play.
2. **ε-greedy wrappers around (1)**: with prob ε per cell, replace the policy's choice with a masked
   random legal action. Sweep ε ∈ {0.05, 0.15, 0.3}. This is the workhorse: coherent games whose
   actions are NOT a deterministic function of state ⇒ action channel becomes identifiable.
3. **MaskedRandomPolicy** (exists) — full random legal traces, ~10–15% of the corpus. Broad state/action
   coverage, including the "flailing beginner" region the RL actor starts in.
4. **Weak/mid checkpoints** (early-training PPO snapshots) — intermediate skill bands.

Opponents (`ai2`): randomBiasedAI, workerRushAI, lightRushAI, coacAI **minus one held out entirely from
pretraining** (keep coacAI eval-only, as now) — plus **Python-vs-Python self-play blocks** (both actions
recorded natively, and these are the only trajectories where *both* channels come from non-scripted
policies). Target rough mix: ~40% ε-noised strong play, ~20% clean strong play, ~15% weak/mid, ~15%
self-play, ~10% pure random. Collect on the same map as v2 first (`basesWorkers16x16`); more maps only
after the probe (Workstream E) passes.

## Workstream D — WM v3: joint-action conditioning

In `models/dreamer/world_model.py`:

- `GridActionEncoder` extension: embed both players' per-cell actions on the H×W grid and **sum with a
  learned player-role embedding** before the conv trunk (cheapest; keeps `n_action` token count
  unchanged), or concatenate channels pre-trunk. Prefer the sum-with-role-embedding first.
- `shift_actions` / `no_action` logic applies to both channels (opponent action at slot t = opponent
  action that produced frame t; `is_first` masks both with the learned first-embed).
- **Opponent-action dropout, p≈0.15**: replace the opponent channel with a learned `unknown_opp`
  embedding at training time. This trains the marginal *and* the conditional in one model — at
  imagination time you can either supply a real opponent policy or fall back to `unknown_opp`; it is
  also the forward-compatibility hook for fog-of-war later.
- Optional but cheap identifiability booster: an **inverse-dynamics auxiliary head** (predict player-1's
  action components from `(z_t, z_{t+1})` register/spatial outputs, masked CE, small coefficient).
- Config: new `pretrain_dreamerv4_dynamics_v3.yaml`; tokenizer block byte-identical to v2 (frozen,
  reuse the v2 tokenizer checkpoint + `latent_scale`).

## Workstream E — Gates: prove action-causality before any RL

Add to the dynamics eval (`eval_dreamer_dynamics.py`) and to `analyze_every` during training:

1. **Counterfactual action probe** (the single most important number): open-loop rollout MSE with
   (a) true actions, (b) shuffled/random player-1 actions, (c) shuffled opponent actions,
   (d) both shuffled. Log the gaps. *Gate: (b)−(a) and (c)−(a) must be clearly > 0 and growing with
   horizon.* Until then, imagination training is provably pointless — do not start RL.
2. **Open-loop MSE vs copy-last** — must beat copy-last by horizon 15 (v2 never did).
3. Existing reward_corr / mask_acc / continue metrics, unchanged.

Also fix the RL-side observability gap while in there: log real-env collected episode return and
win/loss/timeout counts from the collector every console interval (the failed run was blind between
50-iter evals).

## Sequencing for the next thread

1. **A** (jar patch + env surfacing + alignment test) — everything downstream depends on it.
2. **B** (schema) and **D** (model) in parallel — both small, test-driven (see
   `tests/`; keep the "tests as ground truth" discipline, esp. the action-alignment test).
3. **E** probes wired into eval *before* collection finishes, so the first v3 train run reports them.
4. **C** collection (long-running, launch in container background once A+B land).
5. Retrain dynamics v3 → run `eval_dreamer_dynamics.py` → judge by the Workstream E gates.
6. Only then revisit RL — and per the prior analysis, run the **`online` mode baseline first** to
   validate the actor/critic/league/eval stack independent of the WM, then hybrid with the imagination
   opponent drawn from league snapshots (two-sided dreams: our actor picks `a_self`, a frozen snapshot
   picks `a_opp` on the shared latent space — the WM never predicts the opponent at all).

## Research grounding (short list)

- Expert/narrow data breaks world models; diverse+noised collection fixes identifiability:
  arXiv:2509.05735 (offline vs online MBRL data collection); FOCUS, Frontiers of Computer Science 2024
  (causal structured world models); arXiv:2405.18917 (causal action-influence counterfactual
  augmentation).
- Model the other agent explicitly instead of marginalizing: "Dreaming of Others" arXiv:2605.31361;
  CoDreamer arXiv:2406.13600; diffusion-inspired multi-agent world modeling arXiv:2505.20922.
- Simultaneous-move adversarial planning with learned models: Simultaneous AlphaZero arXiv:2512.12486;
  Stochastic MuZero (chance codes for unobserved stochasticity — our `unknown_opp` dropout is the
  amortized analog); learned-model look-ahead in imperfect-info games arXiv:2510.05048 (relevant once
  fog of war is enabled; not needed for the no-fog game).
- Dreamer 4 recipe (shortcut forcing, frozen tokenizer phase) stays the backbone — the v3 change is
  *what conditions the denoiser*, not the objective.

## Implementation status (2026-07-10) — A–E LANDED, ready to collect + retrain

- **A done.** Jar patch v2 (`infra/microrts-jar-patch/`, applied in-container):
  `opponentAction` (H*W,7) exposed per tick from `JNIGridnetClient.pa2` (exact
  inverse of `UnitAction.fromActionArray`), plus `playerIds`/`setPlayerIds` so
  the Python policy can sit in EITHER seat (scripted bot as player 1). Env:
  `EnvConfig(opponent_action=True, player_ids=...)`. Found+fixed an upstream
  bug: `getMatrixObservation(1)` tags neutral resources as "own" for seat 1 —
  corrected Python-side (`_fix_seat1_owner_`). Alignment PROVEN by replay test
  (`tests/test_opponent_action.py`): recorded bot actions replayed through a
  self-play env reproduce the obs trajectory tick-for-tick.
- **B done.** HDF5 format_version=3: `opponent_action` dataset, `traj/policy_id`
  + `traj/action_noise` provenance, `policies` legend; both readers hard-error
  on pre-v3 files; loaders emit `opponent_action` for dynamics/all tasks.
- **C done.** `EpsilonGreedyPolicy` (per-cell masked-legal ε swap);
  `collect_mrts_data.py` is now a collection-matrix CLI (`--plan` blocks or
  `--preset-v3` for the target mix incl. seat-mix + self-play blocks where the
  partner lane's action is the opponent channel). coacAI held out by default.
- **D done.** `GridActionEncoder` embeds both channels with SEPARATE tables
  (a shared-table sum is order-invariant — that's the role distinction),
  `unknown_opp` embedding + `dynamics.opp_dropout: 0.15` (train-time, loss-side),
  threaded through denoise/contextualize/sample_next/open_loop/imagine (incl.
  `opponent_policy` hook for two-sided dreams). Configs:
  `pretrain_dreamerv4_{tokenizer,dynamics}_v3.yaml` (tokenizer retrained for the
  diverse corpus; blocks byte-identical between the two files).
- **E done.** `entrypoints/probes.py::counterfactual_action_probe` (true vs
  shuffled self/opp/both, per-horizon gap growth) wired into the dynamics val
  probe (CF-PROBE console line), `eval_dreamer_dynamics.py` (new opp_noop /
  opp_shuffled / both_shuffled variants + `opponent_conditioning` verdict), and
  `DreamerRLTrainer.analyze_dynamics` (self-channel only; buffer has no opp
  stream). Collector now logs real-env episode return + W/L/T every console
  interval (`pop_episode_stats`).
- 174 tests pass. Next: run the v3 collection (see the CLI docstring command),
  retrain tokenizer v3 then dynamics v3, judge by the E gates.

**Novelty note:** joint-action shortcut-forcing/diffusion world model for a simultaneous-move
adversarial RTS, with league-snapshot opponents inside imagination, has no published equivalent we
found (nearest neighbors above are cooperative SMAC or alternating-move board games). The v2 failure
(expert-only pretraining ⇒ action-ignoring WM ⇒ imagination-hacked actor) is itself a clean, reportable
negative result — keep the `oqwvpf7g` run data.
