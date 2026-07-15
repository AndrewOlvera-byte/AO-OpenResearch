# MicroRTS — Electronic Notebook

Running log + reference for the gym-microrts environment in this monorepo.

## Stack / setup

- **Image:** `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` — ships a Blackwell-capable
  PyTorch (CUDA 12.8, sm_120) so there is no torch re-install churn on the RTX 5070 Ti.
- **Java:** OpenJDK 17 — gym-microrts drives a Java MicroRTS engine through JPype.
- **Versions (pinned, see `pyproject.toml`):** `gym_microrts==0.3.2`, `gym==0.21.0`,
  `numpy==1.23.5`, `JPype1`.

### Gotchas learned

- `gym_microrts` 0.6.0 exists but requires **Python <3.10**; the base image is 3.11,
  so 0.3.2 (last that works here) is pinned.
- 0.3.2 still uses the removed `np.int` alias → **numpy must be <1.24** (1.23.5).
- gym 0.21's `setup.py` is rejected by modern setuptools (`extras_require` marker
  error). The Dockerfile pins `setuptools==65.5.0` / `wheel==0.38.4` and installs with
  `--no-build-isolation`.
- **JPype starts one JVM per process and cannot restart it.** Only one env may exist
  per process; `env.close()` shuts the JVM down for good. Tests share a single
  session-scoped env (see `tests/conftest.py`).

### Commands

```bash
docker compose -f infra/docker-compose.yaml build
docker compose -f infra/docker-compose.yaml up -d
docker exec -i ao-research bash -c 'cd /workspace && python -m pytest'
```

## Environment spec (`MicroRTSGridModeVecEnv`, 16x16 `basesWorkers16x16.xml`)

Vectorized env: `num_bot_envs` parallel games, each vs. a player-2 bot from
`ai2s`. Player 1 is controlled by the agent.

### Observation space

`Box(0, 1, shape=(H, W, 27))` — for 16x16, `(16, 16, 27)`. Each cell is a
concatenation of **one-hot** feature planes (`num_planes = [5, 5, 3, 8, 6]`):

| Planes | Count | Meaning |
|--------|-------|---------|
| Hit points | 5 | 0, 1, 2, 3, ≥4 |
| Resources | 5 | 0, 1, 2, 3, ≥4 |
| Owner | 3 | none, player 1, player 2 |
| Unit type | 8 | none, resource, base, barracks, worker, light, heavy, ranged |
| Current action | 6 | none, move, harvest, return, produce, attack |

Batched reset/step returns `(num_envs, 16, 16, 27)`.

### Action space

`MultiDiscrete` per **unit action**, `nvec = [256, 6, 4, 4, 4, 4, 7, 49]`:

| Index | Size | Meaning |
|-------|------|---------|
| 0 | 256 | Source unit location (flattened `H*W = 16*16`) |
| 1 | 6 | Action type: NOOP, move, harvest, return, produce, attack |
| 2 | 4 | Move direction (N, E, S, W) |
| 3 | 4 | Harvest direction |
| 4 | 4 | Return direction |
| 5 | 4 | Produce direction |
| 6 | 7 | Produce unit type |
| 7 | 49 | Attack target, relative 7x7 window |

Actions are passed to the Java engine as **`int[][][]` of shape
`(num_envs, num_units, 8)`** — one 8-dim vector per commanded unit. A valid no-op
batch is `np.zeros((num_envs, 1, 8), dtype=np.int32)`. A 2-D array will fail JPype
overload resolution.

> Note: 0.3.2 exposes no `get_action_mask()`; invalid actions are ignored by the
> engine rather than masked. (Action masking arrives in later releases.)

### Reward

`step` returns a scalar = `raw_rewards @ reward_weight`. The 6 raw components:

| Index | Reward function | Default weight used here |
|-------|-----------------|--------------------------|
| 0 | WinLoss | 10.0 |
| 1 | ResourceGather | 1.0 |
| 2 | ProduceWorker | 1.0 |
| 3 | ProduceBuilding | 0.2 |
| 4 | Attack | 1.0 |
| 5 | ProduceCombatUnit | 4.0 |

`info[i]["raw_rewards"]` holds the unweighted 6-vector per env.

### Built-in experts (`gym_microrts.microrts_ai`)

`coacAI`, `droplet`, `guidedRojoA3N`, `izanagi`, `lightRushAI`, `mixedBot`,
`naiveMCTSAI`, `passiveAI`, `randomAI`, `randomBiasedAI`, `rojo`, `tiamat`,
`workerRushAI`. Each is a factory `ai(unit_type_table)` used as a player-2 opponent.

## Test suite

`src/micro-rts/tests/` — thin base-integration only:

- `test_env.py` — obs/action space shapes, reset, no-op step, random-action step.
- `test_experts.py` — every expert resolves, loads as an opponent, and advances its
  game (all non-passive experts perturb the board over 20 steps).
- `test_buffer.py` — RolloutBuffer storage + GAE (hand-checked) + minibatch coverage.
- `test_collection.py` — MicroRTSVecEnv (bot + self-play) + RandomPolicy + Collector.
- `test_multiprocess.py` — MultiprocessPool spaces/step/aggregation + Collector on pool.

Status: **26 passed**. Suite runs under `--forked` (one JVM per process). Use the
`--forked` *flag* in addopts — the `pytest.mark.forked` *marker* hangs on JVM teardown.

> Operational note: don't run many concurrent/killed forked-JVM pytest sessions — a
> wedged JVM leaks semaphores/shm and later forked tests hang (even Docker may fail to
> stop the container). If forked tests start hanging, recreate the container
> (`docker compose -f infra/docker-compose.yaml down && up -d`) — a fresh container is
> always green. Run one suite at a time.

## RL collection infra (`environments/` + `collectors/`)

Data-plane built on `tensordict`. Interfaces in `environments/base.py`:
`VecEnv` (async `async_reset`/`send`/`recv` + sync `reset`/`step`) and `Policy`
(`step(obs, mask) -> {action, logprob, value}`). Pieces:

- `MicroRTSVecEnv` (`environments/microrts_env.py`) — adapter over gym_microrts.
  `EnvConfig(mode="bot", bots=(...))` for scripted experts (cycled across envs) or
  `mode="selfplay"` (agent nets both player slots). Obs NHWC→NCHW float; action codec
  tensor→`int[][][]`; episode auto-reset is internal to the Java engine.
- `RandomPolicy` (`models/random_policy.py`) — pass-through test model.
- `RolloutBuffer` (`collectors/buffer.py`) — preallocated `[T,N]` TensorDict + GAE.
- `Collector` (`collectors/collector.py`) — synchronous rollout loop.
- `MultiprocessPool` (`collectors/vecpool.py`) — W worker processes (own JVM each),
  shared-memory TensorDict, async send/recv. Same `VecEnv` API.

### Throughput findings (RandomPolicy, randomBiasedAI, 16x16) — IMPORTANT

| Config | steps/s |
|--------|---------|
| Serial 1 JVM, 24 envs | ~6,000 |
| Serial 1 JVM, 64 envs | ~14,000 |
| Serial 1 JVM, 128 envs | ~17,600 |
| Serial 1 JVM, 256 envs | ~17,800 |
| MultiprocessPool 4 JVMs, 24 envs | ~600 (**10× slower**) |

**The in-process env IS the fast path.** gym_microrts steps all N games in one JNI
call, and **JPype releases the GIL during that call**, so one process already
parallelizes across cores (saturating ~18k sps around 128 envs). The multiprocess
pool's per-step pipe barrier + obs copy is pure overhead at this scale and only makes
sense to (a) run independent experiment shards or (b) exceed one JVM's ceiling — not
for raw throughput here.

**Implication for Phase 3 (async overlap):** prefer a *background thread* that calls
`env.step` (GIL is free during the Java step) concurrently with the GPU policy update,
over multiprocessing. Scale collection first via `num_envs` on a single JVM.

## Phase 3: config-toggleable training loop (`collectors/runner.py`)

`Runner(RunConfig)` drives collection + PPO update with an **`overlap` toggle**:

- `overlap=False` — synchronous: collect rollout, then update. On-policy, no lag.
- `overlap=True` — double-buffered (APPO-style): a background thread collects the next
  rollout on a frozen **actor** snapshot while the main thread runs the GPU update on the
  **learner**; two `RolloutBuffer`s ping-pong. Works because `env.step` frees the GIL, so
  CPU/Java collection overlaps GPU training. Cost: bounded 1-iteration policy lag. The
  actor/learner split (refresh actor only while the collector is idle) keeps it race-free.

Pieces added: `CNNPolicy` (`models/cnn_policy.py`, real conv net), `ppo_loss`
(`loss/ppo.py`), `PPOTrainer` (`trainers/PPOTrainer.py`). `Collector.collect(buffer)`
now takes an optional explicit buffer for ping-pong.

### Throughput sweep (CNN policy + PPO update, RTX 5070 Ti, `scripts/sweep.py`)

| num_envs | horizon | overlap | steps/s |
|---|---|---|---|
| 64  | 128 | off | 5,799 |
| 64  | 128 | on  | 5,079 |
| 128 | 128 | off | 7,338 |
| 128 | 128 | on  | 6,513 |
| 256 | 128 | off | 9,316 |
| **256** | **128** | **on** | **10,136** ← best |
| 256 | 256 | off | 5,728 |
| 256 | 256 | on  | 5,631 |
| 512 | 128 | — | too slow (update-bound) |

**Optimal: serial, `num_envs=256`, `horizon=128`, `overlap=True` → ~10k steps/s.**

Findings:
- Throughput scales with `num_envs` at fixed horizon (5.8k→7.3k→9.3k); the env is cheap,
  so more parallel games amortize the per-iter Python/GPU overhead.
- **Bigger rollouts hurt** (256×256 < 256×128): more samples per update makes the GPU
  PPO step + giant float32 obs buffers dominate. Keep `horizon` modest, scale `num_envs`.
- **Overlap only pays off at scale** — it's *slower* at 64/128 envs (snapshot copy +
  thread + GPU contention exceed the benefit when the update is cheap) and wins ~+9% at
  256 envs where the update is large enough to hide collection behind.
- The full RL loop (~10k sps) sits below the env-only ceiling (~18k): with a real policy
  the system is **GPU/update- and obs-memory-bound, not env-bound**. To push closer to
  env-bound: store obs as int8 planes and one-hot on GPU (4× less memory/bandwidth),
  shrink the net, or cut epochs/minibatches — not more env parallelism.


What's actually broken: not the action encoder, not attention — identifiability

  The architecture is fine on paper. GridActionEncoder (world_model.py:63) is arguably better than stock    
  Dreamer conditioning: the 7 per-cell action components are embedded, summed on the H×W grid,
  conv-downsampled to H/4×W/4 tokens spatially aligned with the latent tokens, with full within-frame       
  attention mixing. Actions physically can influence the prediction.

  The problem is that nothing in the training objective ever forced them to. Your pretrain corpus is        
  expert-bot play, where the action is a near-deterministic function of the state: a_t ≈ π_bot(s_t). That   
  means the action tokens carry approximately zero mutual information beyond what the spatial context       
  already contains — the transformer can predict the next frame just as well from state alone, so gradient  
  descent takes the shortcut and the attention heads learn to ignore the action tokens. This is the
  model-based twin of causal confusion in imitation learning: the model captures the spurious "state        
  predicts next state" relation induced by the sampling policy rather than the causal action → next state   
  mechanism (see FOCUS: offline MBRL with causal structured world models
  (https://link.springer.com/article/10.1007/s11704-024-3946-y) and Offline vs. Online Learning in
  Model-based RL: lessons for data collection (https://arxiv.org/pdf/2509.05735) — narrow expert data is a  
  known WM killer). So: not an attention bug to patch — attention did exactly what the objective permitted. 
  It's a data identifiability problem, and no architecture tweak fixes it while the data stays expert-only. 

  Your own metrics tell exactly this story. Everything state-conditional is excellent (recon 0.001, mask_acc
  0.999, reward_corr 0.96 — all teacher-forced), while everything generative is dead: open-loop MSE ~3×     
  worse than copy-last-frame, z_motion ≈ 0.05. And hybrid finetuning didn't rescue it, because at 4 WM      
  updates/iter at lr 3e-5, the high-entropy replay that could teach action-conditioning barely moved the    
  weights.

  Where your "information incomplete" intuition is right — and where it needs sharpening

  Worth separating three things you're bundling, because they have different fixes:

  1. It's not poker-style hidden information (yet). Default gym-microrts with no fog of war is fully
  observable state — the model sees both players' units. What's hidden is the opponent's action. Since      
  MicroRTS is simultaneous-move, the true dynamics are z_{t+1} = f(z_t, a_self, a_opp), and your WM
  conditions only on a_self. The opponent is an unmodeled latent confounder; the model can only learn the   
  marginal over whatever opponents were in the data. That's your run-2 "rollout MSE floor from unobserved   
  opponent actions" finding, formalized.
  2. The league makes it worse. That marginal is only coherent if the opponent distribution is stationary — 
  but league self-play deliberately makes it nonstationary. The WM is chasing a moving marginal it can't    
  even see. Your instinct that "both players' data needs collected and represented to the model" is, I      
  think, exactly the right conclusion.
  3. The raster nature makes hacking cheap. MicroRTS frames are mostly static (a few units move per tick),  
  so copy-last is a strong predictor, dense-shaped reward is predictable from board state without any causal
  action understanding, and a 4-step flow sample deterministically averages over the multimodal
  opponent-response distribution into static mush. The actor then discovers that in this mush, imagined     
  dense reward drifts upward regardless of what it does — imagined return 0.15 → 1.04, entropy → max, real  
  wins = 0. That's the degeneracy you saw, and yes, it's model exploitation, though note it's dynamics      
  hacking more than eval hacking — your eval already includes coacAI, which is held out from collection, so 
  the eval design is less in-distribution than you feared. The policy didn't overfit the eval; it never     
  learned anything real at all.

  Really interesting takeaway driven through fable, the identifiability all bounds down to the actual transition dynamics not being observable to the model, information incompleteness magnifies this if enabled. The full transition dynamics are required in order to properly model the world effectively under general rollouts and covering a broad distribution of world transition states.

## Opponent modeling roadmap (v4.4 implemented; fog-of-war step saved for later)

Context: v4.3 solved self-action conditioning (issued-cell CF gap +48.5% of
true-action MSE) but the opponent channel stayed ignored (-0.3%). v4.4
(implemented 2026-07-11) is steps 1+2 of the three-step opponent plan:

1. **Opponent-policy head (in tree)** — `dynamics.opp_head`: a per-cell BC head
   on the dynamics transformer trunk predicting the opponent's action AT each
   frame from the jar-patch's engine-executed labels
   (`loss.dreamer.opponent_bc_loss`; CE at opponent source cells only,
   conditional components gated by the true type, terminal-splice slots
   masked, sigma-weighted). Privileged supervision: labels are needed only at
   training time (LIAM / "Dreaming of Others" recipe). Reading the TRUNK is
   the point — the BC gradient forces the dynamics hidden state to encode
   opponent intent.
2. **Dream opponent (in tree)** — `DreamerV4.imagine` samples the head
   (clean trunk pass, source-masked via the decoded frame) whenever no
   explicit `opponent_policy` is passed, feeding it through the same opp
   injection channel as recorded actions — MBOM level-0: an explicit acting
   opponent inside dreams instead of the exploitable `unknown_opp` marginal.
   Costs ~+25% imagination compute (one extra clean pass per imagined step).

### Step 3 — fog of war / information-incomplete (SAVED, do when fog lands)

The key design property of steps 1+2 is that the **interface does not change
under partial observability** — only the head's *inputs* do:

- Under fog, the ego obs no longer contains enemy units. The opponent-policy
  head keeps training on the same privileged executed-action labels (available
  from the engine during training/self-play collection, never at deployment),
  but its features must now come from a **belief module**: the planned separate
  opponent encoder — a recurrent/transformer summary of the ego's observation
  history (scouting glimpses, last-seen positions) — replaces/augments the
  trunk features feeding the head. That encoder is exactly LIAM's
  encoder-decoder shape: ego trajectory in, opponent action/state out, trained
  with privileged targets, deployed on ego inputs alone.
- **Only then** add decoding to privileged opponent STATE (hidden enemy unit
  planes) as a second auxiliary target next to actions: with full obs it is
  redundant (the latent can read enemy units off the frame); under fog it is
  the belief-state loss that grounds the encoder. Predict both "where are
  their units" (masked planes) and "what will they do" (actions).
- The source-cell masking used at sampling time must switch from the decoded
  frame's enemy-idle planes (ground truth under full obs) to the belief
  module's predicted enemy planes — same code path, predicted input.
- Keep `opp_dropout` > 0 through the transition: `unknown_opp` is the trained
  fallback for frames where the belief is useless (game start, total scouting
  blackout).
- Explicitly NOT planned: public-belief-state machinery (DeepNash/R-NaD
  style) — overkill for this pipeline; revisit only if belief-BC dreams
  visibly diverge from real opponent behavior at league scale.

References: LIAM (Papoudakis et al. 2021), "Dreaming of Others"
(arXiv:2605.31361), MBOM (arXiv:2108.01843), COMBO (ICLR 2025), asymmetric/
privileged POMDP RL (arXiv:2412.00985, arXiv:2105.11674).

## v4.4 findings (60k run + eval + deep probes, 2026-07-11/12)

First checkpoint to clear `eval_dreamer_dynamics` with **0 FAIL**
(GO-WITH-CAUTION; v4.3 was NO-GO, v2 before it degenerate). Artifacts:
`checkpoints/dreamer_dynamics_v4_4.pt`, eval in
`checkpoints/dynamics_eval_v4_4`, one-off probes in
`outputs/probe_v4_4_deep.py`, wandb `ujhra09x`.

| metric | v4.3 | v4.4 |
|---|---|---|
| self CF gap (issued cells, % of true MSE) | +48.5% | **+56.9%** |
| opp CF gap (issued cells) | -0.3% (FAIL) | **+6.2%** (WARN) |
| open-loop latent MSE k1 / k4 | 0.273 / 0.290 | **0.236 / 0.252** |
| reward corr (nonzero slots) | 0.90 | 0.83 (regressed) |
| continue AUC | 0.993 | 0.994 |
| imagined reward mean +/- sd (random actor) | 1.99 +/- 1.58 | 4.37 +/- 0.27 |

### The opponent question is CLOSED as a data problem (measured, not argued)

1. **Masked-input BC probe (decisive):** the opponent-BC head's type accuracy
   is 0.910 with the true opponent input stream and **0.899 with the entire
   stream masked to `unknown_opp`** — the trunk reconstructs the opponent
   almost perfectly from board state alone. In the v3 corpus the opponent
   channel carries ~1 accuracy point of unique information, so gradient
   descent is CORRECT to barely use it. Same causal-confusion mechanism as
   the v3 self-action failure (see "identifiability" section above), now
   isolated to the opponent channel.
2. **Why self worked and opp didn't, despite symmetric architecture:** both
   channels share the grid-sum -> conv trunk -> per-cell injection (pos-enc /
   attention ruled out; collector frame alignment verified correct,
   collector.py row t = obs_t + same-tick opp action). The asymmetry is the
   collection matrix: the SELF channel got exogenous variation (eps
   0.05/0.15/0.30 blocks + masked_random block + sampling entropy) while the
   opponent is a deterministic scripted bot in ~85% of the corpus (only the
   15% selfplay block at eps=0.05 has stochastic opponent actions).
3. **Pipeline coherence (open-loop, real self actions):** MSE ordering
   true 0.203 < head-sampled +2.0% < unknown +4.6% < shuffled +6.5%.
   Conditional beats marginal modestly, wrong opponent hurts more than none,
   and the head's own samples nearly recover true-stream quality — the
   dream-opponent path works end to end.

### Watch items for RL (why eval gating is the contract)

- **Rosy dreams:** imagined reward 4.37 with sd only 0.27 — a low-variance
  optimistic reward surface is exploitable by the actor. Trust only
  `eval/win_rate`; imagined return rising while win_rate stalls = dynamics
  hacking, stop.
- **Reward head regressed** (0.90 -> 0.83 corr; terminal magnitudes shrunk,
  sign acc still 1.0) — most likely `opp_bc_coef=1.0` competing for trunk /
  register capacity. Next dynamics train: `opp_bc_coef 0.3-0.5`.
- Latent RMS drifts to 0.79 over the 15-step horizon (borderline WARN).

## Post-v4.4 RL audit and repaired setup (2026-07-12)

The first `rl_dreamerv4_hybrid_v4_4` run was stopped conceptually after 9.5M
steps: 94 evals had zero wins, imagined returns reached 130-160, and live
open-loop latent MSE was 2.2x copy-last. The tokenizer remained healthy on held
out v3 data (occupied-cell categorical accuracy 98.8-99.9%, mask precision /
recall 92.3/91.7%, no latent saturation), so the evidence points first to
dynamics exploitation rather than proven representation loss.

The audit found an RL-only regression: hybrid finetuning called
`shortcut_forcing_loss` without v4's occupied/changed-cell weights and replayed
no opponent actions. Thus online training returned to background-dominated
uniform MSE and trained only the moving `unknown_opp` marginal while the shared
trunk drifted underneath the opponent BC head.

The repaired `rl_dreamerv4_hybrid_v4_4.yaml` now:

- stores same-tick executed opponent actions for both patched-jar bots and
  Python self-play, with an explicit validity bit;
- uses the full foreground-weighted `dynamics_loss` online and continues
  opponent BC at coefficient 0.3;
- mixes a 0.5-weight v3.1 offline anchor batch into every WM update;
- logs normalized open-loop/copy-last MSE plus generated-mask precision/recall;
- pauses actor/critic updates unless open-loop beats copy-last, issued-cell
  self-action sensitivity is present, and mask precision/recall exceed 0.8.

`rl_dreamerv4_online_representation_v4.yaml` is the required isolation run: it
freezes tokenizer+world-model and trains the actor/critic only on real replay.
It answers whether the current /4, 16x32 latent supports control independently
of imagined dynamics. Do not redesign/retrain the tokenizer unless this real-RL
baseline is also flat; if it is, the next controlled change is a /2 spatial
tokenizer (with a new tokenizer and dynamics checkpoint), not a larger dynamics
transformer on the same latent.

### Next steps

1. **Run the representation baseline** — this requires no retraining and gives
   the cleanest answer about the frozen tokenizer:

       python src/micro-rts/entrypoints/train_dreamer_rl.py \
           --exp micro-rts/rl_dreamerv4_online_representation_v4

2. **Retrain dynamics v4.5 on the already-collected v3.1 corpus** using the
   frozen v4 tokenizer and `opp_bc_coef=0.3`:

       python src/micro-rts/entrypoints/train_dreamer_dynamics.py \
           --exp micro-rts/pretrain_dreamerv4_dynamics_v4_5

3. **After v4.5 passes the dynamics evaluator**, run the repaired hybrid with a
   fresh output name. The health gate keeps actor/critic paused until live
   replay passes:

       python src/micro-rts/entrypoints/train_dreamer_rl.py \
           --exp micro-rts/rl_dreamerv4_hybrid_v4_4 \
           --set run.name=rl_dreamerv4_hybrid_v4_5 \
           --set training.dreamer.init_from=checkpoints/dreamer_dynamics_v4_5.pt

### v3.1 corpus recipe (already collected)

The opponent-identifiability fix is ~50% selfplay with an eps ladder (the
opponent channel is the partner lane's Python policy, the only place opponent
noise is injectable — scripted bots act inside the JVM), plus a fully-random
selfplay block and bot blocks as anchors. Same held-out coacAI rule. The
collection command was:

       docker exec -i ao-research bash -c 'cd /workspace && \
         python src/micro-rts/collectors/offline_data/collect_mrts_data.py \
           --name tokdyn_pretrain_v3_1 --num-envs 24 --policy-device cuda \
           --plan mode=selfplay,policy=checkpoints/base_rlFS_expert_masked_league/best.pt,eps=0.05,steps=9000 \
           --plan mode=selfplay,policy=checkpoints/base_rlFS_expert_masked_league/best.pt,eps=0.15,steps=12000 \
           --plan mode=selfplay,policy=checkpoints/base_rlFS_expert_masked_league/best.pt,eps=0.30,steps=6000 \
           --plan mode=selfplay,policy=masked_random,steps=3000 \
           --plan mode=bot,policy=checkpoints/base_rlFS_expert_masked_league/best.pt,steps=12000,seats=mix \
           --plan mode=bot,policy=checkpoints/base_rlFS_expert_masked_league/best.pt,eps=0.15,steps=9000,seats=mix \
           --plan mode=bot,policy=checkpoints/base_rlFS_expert_masked_league/best.pt,eps=0.30,steps=3000 \
           --plan mode=bot,policy=masked_random,steps=6000'

(60k steps/lane x 24 lanes = 1.44M transitions, same budget as v3; 50%
selfplay vs v3's 15%. Optional collector extension for even more opponent
entropy: heterogeneous selfplay pairs — strong vs masked_random/mid seats —
currently unsupported, one policy drives both lanes.)

## Dynamics v2 research synthesis: fix the state before scaling the model (2026-07-12)

### Executive conclusion

The current failure is not primarily evidence that the transformer is too
small, nor that PPO/RL is inadequate. The first-order problem is that the
world model is trained as if the 27-plane Gym-MicroRTS observation were a
Markov state, but it is not. The engine is information-complete; the raster
interface supplied to the model is not. The second-order problem is that a
sparse, deterministic, symbolic transition is compressed to a 4x4 continuous
latent and trained with a same-frame Gaussian denoising objective. That
combination rewards copying/averaging much more strongly than learning exact,
action-routed changes.

This distinction changes the plan:

> First build an oracle, factored, full-engine-state transition model that can
> learn MicroRTS mechanics nearly exactly. Only after that baseline works should
> compression, stochastic latents, imagination RL, or hidden-opponent belief be
> layered back in.

The v4 tokenizer remains a good observation/policy encoder. It is not the
right place to establish whether the game mechanics are learnable, so do not
retrain it first. Dynamics v2 should initially bypass it.

### Decisive local finding: the current input aliases different engine states

The installed MicroRTS JAR was inspected directly (`javap` plus decompiled
`GameState.getMatrixObservation`). A `UnitActionAssignment` contains the unit,
the complete `UnitAction`, and its assignment start time. The complete action
contains its type, direction or attack target, produced unit type, and ETA.
The 27-plane observation exposes only five raw values per cell:

- hit points;
- carried resources on the unit (not the player's global resource stock);
- owner;
- unit type;
- current unit-action **type**.

It omits at least the action start/elapsed/remaining time, action direction or
target, produced unit type, both players' global resource stocks, and engine
resource reservations/conflict state. This matches the public
[Gym-MicroRTS observation description](https://pypi.org/project/gym-microrts/0.3.2/),
which describes 27 planes but only a categorical current-action plane, while
the submitted GridNet action contains several action parameters.

The omission is consequential because MicroRTS actions are asynchronous. In
the installed unit table, movement takes roughly 8--12 ticks, attacks 5 ticks,
harvest 20 ticks, worker production 50 ticks, combat-unit production 80--120
ticks, and base/barracks production 200--250 ticks. A busy unit receives no new
submitted action while the observation repeats the same current-action type.
Two visually identical frames with the same submitted joint action can
therefore differ only in a hidden action timer and transition differently on
the next tick.

This is not hypothetical. An exact scan of all 1,439,998 v3.1 transitions
compared consecutive occurrences of identical 27-plane state plus identical
effective legal self action plus identical opponent action:

| corpus test | count |
|---|---:|
| identical adjacent model inputs | 127,654 |
| those followed by contradictory next outputs | 33,888 |
| contradiction rate conditional on identical input | **26.5468%** |

Thus the function currently requested from the network does not exist:

`next_raster = f(current_raster, submitted_self_action, submitted_opp_action)`

is one-to-many because the left-out engine state determines completions,
spawns, moves, damage, legality, and resource changes. "No fog of war" means
all units are visible; it does not mean the matrix observation serializes the
full simulator state. Full observability at the engine API and Markov
sufficiency of a learned-model input are separate properties.

This also explains the observed signature: teacher-forced occupied/scalar
metrics can look strong while open-loop latents contract, static copying is a
hard baseline to beat, and action counterfactual gaps remain weak. Under MSE,
the optimal prediction for aliased completion time is a conditional average.
In latent space that average is often an off-manifold, lower-norm board.

### What the present architecture is asking the network to do

The v4/v4.5 path has several reasonable components, but their composition is a
poor inductive bias for this simulator:

1. Each 16x16x27 frame is independently compressed by two stride-2 convolutions
   to 16 continuous 32-D tokens (a 4x4 lattice). One latent cell must encode a
   4x4 region in a game where one-cell movement, adjacency, range, blocking,
   spawn position, and unit identity are the dynamics.
2. Each of seven action components is embedded at the **source** cell. The
   components are summed, the self and opponent maps are summed, and a CNN
   downsamples the result to 4x4. The destination/attack/spawn cell is never an
   explicit endpoint of the action. The model must rediscover routing after
   lossy spatial pooling.
3. Random 16-tick windows reset transformer memory in the middle of actions
   whose durations reach 250 ticks. The missing timer cannot generally be
   inferred from such a window, and the first frame has no burn-in at all.
4. Shortcut forcing corrupts the target frame itself and predicts that same
   clean latent. At most signal levels, the easiest solution is denoising the
   supplied current target, not using history and action to generate it. With
   the current 16-level ramp, the pure-prior `sigma=0` case contributes only
   about 1.2% of total ramp weight.
5. Between one quarter and one half of rows use self-consistency targets rather
   than empirical next-state flow. That is useful only after the grounded
   transition is already good; here it reinforces the model's own smooth
   attractor.
6. The dynamics target is a Euclidean continuous latent even though most state
   fields and changes are categorical. A unit is at cell A or B; an averaged
   latent between the two is not a valid intermediate game state.

None of these alone proves Dreamer 4 cannot work. Together with non-Markov
inputs, however, they make more depth, width, BC loss, or training time a poor
next intervention.

### Comparison with successful world-model recipes

| Work | Choice that matters | Difference here | MicroRTS implication |
|---|---|---|---|
| [DreamerV3](https://arxiv.org/abs/2301.04104) | Recurrent deterministic state plus categorical stochastic state; prior/posterior and balanced KL make the learned state history-dependent and Markovian. | The transformer is reset on short arbitrary windows and has no persistent belief over omitted timers. | Either expose complete state, or treat the raster as a POMDP and use persistent recurrence with long burn-in. |
| [Dreamer 4](https://arxiv.org/abs/2509.24527) | Causal temporal tokenizer, masked-autoencoder training, efficient interactive transformer, shortcut forcing, and running-RMS balancing across losses. | The tokenizer is frame-local; losses use fixed coefficients; target-frame denoising dominates the rare pure-prior condition; MicroRTS controls have delayed asynchronous effects. | Dreamer 4's objective should not be transplanted independently of its tokenizer/training recipe, and its signal schedule needs a dynamics-specific prior term here. |
| [IRIS](https://arxiv.org/abs/2209.00588) / [STORM](https://arxiv.org/abs/2310.09615) | Discrete visual tokens and autoregressive/categorical dynamics make valid modes explicit. | Continuous MSE encourages interpolation between mutually exclusive cell states. | Prefer categorical factored targets for an already symbolic grid. |
| [Delta-IRIS](https://arxiv.org/abs/2406.19320) | Encodes stochastic **deltas** while continuous tokens summarize current state; reports strong Crafter results and direct world-model evaluation. | The current model spends most capacity reconstructing unchanged board content and predicts the whole next latent. | Predict changed fields/events explicitly; keep unchanged state as a deterministic skip connection. |
| [EMERALD](https://openreview.net/forum?id=zNUOZcAUxz) | Explicit spatial latent state, categorical VAE, parallel MaskGIT decoding; spatial plus temporal state outperforms temporal-only state in Crafter. | Our 4x4 spatial state exists, but each token covers 16 semantically distinct board cells. | Preserve spatial state at game-cell resolution; the correct token count follows simulator topology, not image compression convention. |
| [Binding Actions to Objects](https://arxiv.org/abs/2204.13022) | Explicit action-to-object attention improves structured world models, including object grid worlds. | Actions are source-raster features pooled before the model identifies the affected object/cell. | Bind each action to its acting unit and explicit destination/target; keep roles distinguishable until interaction. |
| [Parallel Observation Prediction / REM](https://arxiv.org/abs/2402.05643) | Discrete token world model; ablations show observation-token budget and spatial tokenization matter. | A fixed 16-token budget was accepted because reconstruction was high. | Sweep per-cell, 8x8, and 4x4 under the same transition objective; reconstruction accuracy is not transition sufficiency. |
| [MBOM](https://arxiv.org/abs/2108.01843) and multi-agent opponent models | Opponent policy is modeled because its behavior changes the transition distribution seen by the ego agent. | Mechanics learning, access to the opponent's executed action, and predicting the opponent policy are currently partially entangled. | First learn `p(s' | s, a_self, a_opp)` with privileged joint action; later learn `p(a_opp | history)` under fog and compose the two. |

Crafter is a useful stress test but not a direct architectural template. It is
rendered imagery with meaningful perceptual uncertainty. MicroRTS already
provides compact symbolic state and, in the default engine configuration,
largely deterministic mechanics. A visual generative bottleneck is optional
overhead until an exact structured baseline succeeds.

### Recommended v2A: structured spatial deterministic world model

This is the shortest path to a trustworthy mechanics baseline.

**State schema.** Export a lossless engine snapshot for training and model
input:

- one token per 16x16 board cell with categorical owner/type and bounded
  numerical HP/carried resources;
- for each active assignment: type, direction or target coordinate, produced
  unit type, elapsed and/or remaining ETA;
- global tokens for game tick, both players' resource stocks, and any resource
  usage/reservations required by legality;
- stable unit ID or an equivalent object association if identity across
  movement cannot be recovered unambiguously;
- RNG/conflict-resolution state only if a nondeterministic unit table is used.

The first acceptance test for this schema is mechanical: rerun the duplicate
input test and require the contradiction rate to be zero (apart from explicitly
modeled randomness).

**Action schema.** Represent each issued/active action as an event or edge:

`(player role, source/unit id, action type, destination/target, produced type)`.

Cross-attend event tokens to both source and target cell/object tokens, or
scatter separate action features into source and effect rasters. Do not sum
self and opponent streams before interaction. Simultaneous actions can then be
resolved by an interaction layer rather than hidden inside pooled embeddings.

**Backbone.** Start with a modest residual U-Net, axial transformer, or local
spatial transformer at 16x16 resolution plus a few global tokens. Capacity is
not the research question yet. Preserve a direct unchanged-state skip path and
predict deltas at each cell/object.

**Heads and losses.** Decode the engine fields, not tokenizer latents:

- categorical cross-entropy for owner, unit type, assignment type/direction,
  spawn/death, and existence;
- bounded regression or categorical bins for HP, carried/global resources,
  and remaining ETA;
- a changed-cell/event gate trained with focal or class-balanced loss;
- separate destination/target and action-completion auxiliary heads;
- optional legality/mask prediction from the complete state as a diagnostic.

Compute state loss predominantly on changed cells/events while retaining a
small unchanged consistency term. Train one-step first. Add multi-step open
loop with scheduled model inputs only after one-step event prediction is near
exact. There is no reason to introduce Gaussian flow for the default
deterministic engine baseline.

### Recommended v2B: object/event graph as the scalable research model

Once v2A proves the data and targets, an object-centric version is likely the
cleaner long-term architecture for larger maps, unit identity, and opponent
belief:

- one token per unit/resource plus global player tokens;
- fields `(id, owner, type, x, y, hp, carried, active assignment, ETA)`;
- relative-position/local-neighborhood attention or graph edges;
- action events bound directly to actor and target;
- delta heads for location, HP, assignment, resource transfer, existence,
  spawn, and death.

This factorization makes the later incomplete-information problem explicit:
the mechanics model remains `p(s' | s, joint action)`, while a recurrent belief
model estimates hidden object state and the opponent policy estimates its
action. It also avoids forcing a unit that moves one square to disappear from
one image patch and reappear in another.

### If retaining the current Dreamer-4 path: required v2C corrections

This should be an ablation after v2A, not the new baseline:

1. Supply the full Markov engine fields or a recurrent state with at least
   250-tick burn-in and continuity across training chunks.
2. Increase spatial resolution to 16x16 first (8x8 as a compression ablation),
   and use per-channel normalization/whitening rather than only one global
   latent RMS.
3. Replace source-only pooled actions with source-target event tokens and keep
   self/opponent identity separate.
4. Add an explicit, heavily sampled `sigma=0` empirical next-state loss. Make
   prior generation a first-class objective, not ~1.2% of a denoising ramp.
5. Match train/inference context corruption and integration conditions; add
   K=4 open-loop loss because K=4 is the deployed sampler.
6. Normalize competing dynamics/reward/continue/BC losses by running RMS as in
   the full Dreamer-4 recipe.
7. Delay self-consistency/bootstrap rows until the empirical one-step and
   open-loop event gates pass.

### Data recipe: teach interventions, not only correlations

More trajectories from the same policies will not by itself identify action
effects. The simulator provides a stronger tool than passive video datasets:
clone an engine state and branch it under alternative legal joint actions.
For selected states, store several counterfactual next states spanning:

- no-op versus each legal self action;
- alternate destination/target/produce parameters;
- alternate opponent actions while self action is held fixed;
- simultaneous collisions, contested harvests, attacks, spawns, and resource
  constraints.

Stratify ordinary replay around action issue, completion, movement, damage,
spawn/death, and resource-transfer events. Downsample long unchanged runs or
model them semi-Markovly as `(next event, duration)`. Passive expert/random
mixtures still matter for state coverage, but branched transitions are the
clean causal test of action conditioning.

### Experiment ladder and gates

Run all representation/objective comparisons on identical train/validation
episodes and paired seeds:

| ID | change | question / gate |
|---|---|---|
| E0 | augment raster with full assignments, timers, globals | Does identical `(state, joint action)` always have one next state? Require zero unexplained contradictions. |
| E1 | v2A raw factored one-step model | Can the network learn mechanics without tokenizer/flow? Require near-perfect unchanged accuracy **and** high changed-cell/event F1. |
| E2 | 27 planes vs full state | Quantifies the irreducible cost of missing state. |
| E3 | 4x4 vs 8x8 vs 16x16 tokens | Is spatial compression losing transition-relevant topology? |
| E4 | whole-latent MSE vs categorical delta heads | Does a structured objective eliminate contraction/averaging? |
| E5 | source-only raster vs source-target action events | Does explicit routing improve paired counterfactual accuracy? |
| E6 | random 16 windows vs persistent state / 250+ burn-in | How much of the hidden timer can recurrence recover if the raster is retained? |
| E7 | deterministic predictor vs categorical stochastic vs flow | Is any residual uncertainty real after complete state is supplied? |

Every model should report: one-step field accuracy, changed-cell precision/
recall/F1, action-completion timing error, spawn/death F1, resource error,
counterfactual action sensitivity on paired branches, exact-state match rate,
and 10/50/250-step open-loop validity. Compare against copy-last separately on
event and non-event frames; aggregate MSE is not an acceptance gate.

### What not to infer or do next

- Do not infer Markov completeness from the absence of fog of war.
- Do not retrain the existing tokenizer merely because its latent dynamics
  contract; its reconstruction metrics already answer a different question.
- Do not scale `d_model`, layer count, or dataset size before E0/E1.
- Do not raise opponent-BC weight to force mechanics action conditioning; BC
  predicts a policy and does not prove that the transition uses the action.
- Do not accept teacher-forced denoising metrics as evidence of an autonomous
  simulator.
- Do not introduce opponent-belief uncertainty until privileged full-state,
  privileged-joint-action mechanics are stable. Otherwise epistemic opponent
  uncertainty and accidental state aliasing are impossible to separate.

### Decision moving forward

The immediate milestone is **not RL** and is **not tokenizer v5**. It is an
oracle MicroRTS mechanics model with a provably Markov schema and structured
delta targets. If v2A cannot reach near-exact one-step event prediction, debug
the export/action alignment/losses. If it can, compare compression and
generative objectives against it one variable at a time. Only after open-loop
mechanics survive long production timers should imagination RL resume.

The eventual information-incomplete program then has a clean decomposition:

1. complete-state mechanics model;
2. recurrent belief model over hidden state;
3. opponent policy/intent model over hidden actions;
4. composition and calibration under self-play;
5. imagination RL.

That decomposition preserves the research goal while ensuring that failure in
the later stages is genuinely about opponent inference rather than an
unobserved barracks timer in the supposedly complete baseline.

## World-model v2 representation decision (2026-07-12)

The representation questions above are resolved in
[`WORLD_MODEL_V2_REPRESENTATION.md`](WORLD_MODEL_V2_REPRESENTATION.md). The
implementation target is a privileged hybrid state: complete per-cell unit and
active-assignment fields, global resources/reservations/time, sparse joint action
events bound to source units and effect targets, and learned 16x16 -> 8x8 spatial
compression. PPO keeps the existing 27-plane input.

The initial dynamics backbone will be a conventional causal transformer so token
ordering, masks, and action attention remain easy to audit. Flow matching and the
shortcut/skip head remain first-class; they are now grounded by exact structured
next-state heads and an explicit high-weight pure-prior objective. The complete
state exporter and HDF5 v4 transition schema come before tokenizer/dynamics
training, and the old corpus is not sufficient for this model because it does not
contain assignment timers or player resource state.

Implementation and operational commands are recorded in
[`WORLD_MODEL_V2_RUNBOOK.md`](WORLD_MODEL_V2_RUNBOOK.md). The working vertical
slice includes batched JAR export, exact terminal arrivals, configurable 15%
cloned-engine action branches, HDF5 v4 loading, compressed and oracle tokenizer
configs, sparse action events, causal flow/pure-prior/skip training, a dataset
audit, and a first structured evaluator. Container smoke runs completed through
collect -> tokenizer -> dynamics -> evaluation before full recollection.

## Structured-v2 action-conditioning postmortem and paired experiment (2026-07-14)

The completed Dreamer-4-objective run reached a low validation loss but failed the
mechanics criterion. Its autonomous next-latent MSE was `0.804` versus `0.112` for
copy-last, and paired action-effect F1 was `0.019`. On affected counterfactual
rows, the frozen tokenizer retained the intervention (`0.0782` normalized latent
effect MSE), but the dynamics did not reproduce its geometry:

| query | predicted/target effect cosine | predicted/target norm |
|---|---:|---:|
| `tau=0`, one step | `0.005` | `0.033` |
| four-step flow sample | `0.011` | `0.257` |

This localizes the main failure to the dynamics objective rather than erased
state information. The empirical Dreamer-4 term samples tau uniformly and
multiplies it by `0.9*tau + 0.1`; consequently the target-free `tau=0` query has
only about 1.2% of the weighted empirical mass. High-tau rows can achieve a very
small denoising loss by reading the nearly clean arrival. Paired cloned-engine
branches were loaded for validation but not for training, so nothing required
the prediction difference under two actions to match the true intervention.
Uniform latent MSE further diluted the signal over 128 entity slots (roughly 18
occupied in the inspected examples), 64 spatial tokens, and only about 1.8
changed cells per action.

The next controlled experiment is
`micro-rts/pretrain_structured_dynamics_v2_causal_paired`. It keeps the same
tokenizer, transformer, data split, and 60k-step budget, but changes the training
contract:

- train exactly the deployed one-step `tau=0, d=1` query;
- predict factual and cloned counterfactual arrivals from common noise;
- align `prediction(counterfactual) - prediction(factual)` with the frozen
  tokenizer's true paired effect;
- exclude packed entity padding, boost occupied tokens, and strongly boost
  tokens whose representation actually changes;
- evaluate with one inference step and report effect cosine and norm ratio in
  addition to paired counterfactual MSE and shuffled-action gaps.

The factual, counterfactual, and effect losses are each normalized independently,
so a 15% branch rate does not shrink causal supervision to 15% of a minibatch.
The effect coefficient is `2.0` and changed-token boost is `8.0` for this first
run. Flow bootstrap and shortcut consistency are intentionally disabled; they can
be distilled back in only after the direct predictor beats copy-last and the
paired effect cosine/norm move toward `1`.

Episode boundaries are not the source of the observed contraction. Structured
HDF5 rows contain an aligned engine `state -> next_state` transition, terminal
arrivals are stored explicitly, and dataset windows do not cross trajectory
segments. This experiment also uses `seq_len: 1`, so the loss never constructs a
terminal-to-reset transition from adjacent rows. The new paired dataset task
requires counterfactual fields and fails at startup on a corpus collected without
them, making that assumption auditable rather than silently dropping the branch.

## Structured-v2 causal-paired v2 result and continuation decision (2026-07-14)

The completed 40k-step fine-tune is WandB run `0qn6haqk`, attached to
`pretrain_structured_dynamics_v2_causal_paired_v2`. It initialized from the
best checkpoint of the first causal-paired run and kept the same 50M-parameter
structured dynamics model, paired objective, and frozen tokenizer.

This is a substantive success, but not convergence:

| metric | first causal-paired run | v2 final | v2 best (step 39k) |
|---|---:|---:|---:|
| validation paired-CF MSE | 2.338 | 0.181 | **0.178** |
| validation effect cosine | 0.566 | 0.664 | **0.655** |
| validation effect norm ratio | 0.558 | 0.692 | **0.706** |
| validation self gap | — | 0.0285 | 0.0289 |
| validation opponent gap | — | 0.0097 | 0.0108 |

The result confirms that the paired counterfactual objective is learning action
geometry: the model is no longer collapsing the branch difference, and paired
error is dramatically lower than the prior run. The remaining weakness is
systematic under-magnitude and imperfect direction. Effect cosine and norm
ratio improve through training but remain noisy, while the best total validation
loss occurs around 39k and the final step is slightly worse. This looks like a
convergence tail, not a representation failure, so the 50M model should be
given a lower-learning-rate continuation before changing architecture or
reintroducing flow/bootstrap losses.

Important qualification: the foreground-weighted validation latent MSE is still
`0.4007` versus `0.3598` for copy-last. The paired intervention probes are the
stronger causal signal here, but the model has not yet earned a general
one-step mechanics pass on the aggregate transition metric.

The next experiment is
`micro-rts/pretrain_structured_dynamics_v2_causal_paired_v3`:

- initialize from `checkpoints/pretrain_structured_dynamics_v2_causal_paired_v2/best.pt`;
- retain the same tokenizer, transformer, zero-noise one-step target, loss
  coefficients, token weighting, data split, and batch size;
- train 60k additional steps at `1e-5`, decaying to `2.5e-6`;
- double fixed validation/probe batches for a less noisy decision;
- continue selecting checkpoints by validation loss while reporting paired-CF
  MSE, effect cosine, and effect norm ratio as the mechanics gates.

Decision gates for v3: continue only if paired-CF MSE decreases and effect
cosine/norm ratio move upward together. A plateau near cosine 0.65–0.70 with
norm below 0.8 means the next controlled change should increase causal-effect
weight or add an explicit norm calibration term. Do not add flow sampling,
self-consistency, larger models, or RL until the one-step paired effect is
directionally reliable and near unit magnitude.

## Structured-v2 representation and encoder-pretraining analysis (2026-07-14)

### What the tokenizer actually emits

The structured tokenizer does **not** reduce a MicroRTS frame to one vector.
For each batch element it maps the complete canonical state
`[256 cells, 16 fields]` plus eight globals to a sequence
`z_t` with shape `[195, 128]`:

- **64 spatial tokens:** categorical and numerical cell fields plus absolute
  coordinates are embedded at 16x16 resolution; each 2x2 region is linearly
  compressed and the resulting 8x8 lattice is processed by a two-layer spatial
  transformer;
- **128 entity tokens:** occupied cells are packed in row-major order and
  projected without spatial pooling, with learned padding in unused slots;
- **3 global tokens:** game, self-player, and opponent-player state.

The dynamics model projects each 128-D state token to its 512-D transformer
width. It preserves the 195-token structure; there is no mean pool or single
frame embedding. Per-channel latent mean/std from tokenizer pretraining are
frozen and used to normalize dynamics targets.

The state side is already the proposed observation autoencoder. The tokenizer
was trained for 50k steps with field-aware reconstruction of canonical
categorical fields, numerical HP/resources/targets/timers, global resources and
reservations, plus auxiliary legacy-observation and legality-mask heads. During
dynamics training it is loaded from `structured_tokenizer_v2.pt`, put in eval
mode, and `requires_grad_(False)`. Departure, factual arrival, and paired
counterfactual arrival latents are all produced under `no_grad`.

### How actions are represented

Actions are also not compressed to one vector. The two dense GridNet action
rasters are converted into at most 32 sparse joint-action events. Each event
retains:

`(role, source x/y, action type, destination x/y, direction, produced type,
attack offset)`.

The diagnostic JVM unit ID is carried in the raw record but intentionally not
embedded. Role plus source coordinate binds the event to the acting unit in the
single-occupancy departure state. Self and opponent actions use the same event
schema with a role embedding. No-action frames retain one valid sentinel event.

`ActionEventEncoder` sums learned field embeddings and a numerical attack-offset
projection, then layer-normalizes the result to produce one 512-D transformer
token per event. The full direct-prediction sequence is therefore:

`[195 departure tokens, 32 action slots, signal, step, 195 zero target slots]`

for a maximum sequence length of 424. A causal transformer lets target slots
read the complete departure/action prefix and earlier target slots. The current
implementation has no separately pooled joint-action vector and no action
decoder; action embeddings are trained jointly with the dynamics transformer.

### What the current causal-paired objective teaches

The v2/v3 causal-paired runs train the exact deployed one-step query rather than
a teacher-forced approximation:

1. encode and normalize `state_t`, factual `state_t+1`, and the cloned-engine
   counterfactual arrival with the frozen tokenizer;
2. provide zero initial target tokens with `tau=0`, `d=1`;
3. predict the factual arrival from `(z_t, joint_action)`;
4. predict the cloned counterfactual arrival from the same `z_t` and the
   alternative action;
5. directly align
   `prediction(counterfactual) - prediction(factual)` with
   `z(counterfactual) - z(factual)`.

Factual, counterfactual, and effect terms are independently normalized, with
coefficients `1, 1, 4`; changed tokens receive an 8x boost, packed entity padding
receives weight 0.05, and flow bootstrap/shortcut consistency remain disabled.
This prevents the old solution where the model reads a nearly clean arrival or
ignores actions because state predicts scripted behavior.

### Is action conditioning working as intended?

**Yes as a causal-learning mechanism; not yet as a finished simulator.** The
change from the Dreamer-4 objective to direct paired supervision produced the
signature expected from real action conditioning:

- paired effects no longer collapse toward zero;
- live effect cosine is roughly 0.73-0.75, so alternative actions move the
  prediction in substantially the correct latent direction;
- training effect norm is usually 0.78-0.85, showing remaining
  under-magnitude rather than action ignorance;
- shuffled self and opponent actions both worsen prediction;
- the independent structured evaluator improved from v2 to the current v3
  best checkpoint: changed-cell F1 `0.754 -> 0.766`, self-action gap
  `0.0117 -> 0.0124`, and paired-CF latent MSE `0.1121 -> 0.1105`.

The qualifications matter. Autonomous latent MSE is still `0.0878` versus
`0.0758` for copy-last, and decoded paired-effect F1 is only `0.0688`. Thus the
model has learned useful intervention direction but not yet exact effect
location/content. Validation effect-norm spikes above 1 are also not trustworthy
evidence of overshoot: the metric averages per-row norm ratios, so paired rows
with tiny true effects create small-denominator outliers. Aggregate norm ratio
should eventually be reported as a ratio of summed norms or stratified by true
effect magnitude.

The slow convergence is consistent with supervision density. About 20% of rows
have a cloned branch, and only about 4% of valid tokens in a paired row carry the
measured branch effect. Independent normalization prevents the causal loss from
being scaled down to 0.8% of the objective, but each minibatch still contains
few distinct interventions and the gradients cover heterogeneous mechanics
(move, attack, completion, spawn, harvest, and resource transfer). At the v3
learning rate of `1e-5`, a long convergence tail is expected.

### Would action-encoder autoencoding help?

The hypothesis is partly right:

- observation/state autoencoding is already implemented and frozen, including
  the important timers, targets, globals, assignments, and legality mask;
- an action reconstruction pretrain could verify that the summed 512-D event
  token preserves role/type/source/destination/direction/produced-type fields
  and provide a cleaner initialization;
- source/target grounding auxiliaries could additionally require the event token
  to identify the acting unit and affected destination in the departure state.

A plain action autoencoder is not sufficient by itself. The action record is
small and explicit, so reconstructing it can be solved without learning what the
action does. It would prevent information loss or embedding collisions, but it
would not establish the causal map from event to state delta. Freezing such an
encoder permanently could also choose an invertible geometry that is awkward
for mechanics prediction and prevent useful task alignment.

The best future controlled design is therefore:

1. pretrain the event encoder with exact field reconstruction and source/target
   binding diagnostics;
2. initialize dynamics from it, but keep it trainable (or freeze only for a
   short warm-up with a lower encoder learning rate afterward);
3. retain paired counterfactual effect alignment as the decisive causal loss;
4. compare against the present randomly initialized action encoder at identical
   data, model, and optimizer budgets;
5. prioritize paired/event-balanced minibatches if v3 plateaus, because that
   attacks the measured supervision bottleneck more directly than action
   reconstruction.

### Decision

Do not interrupt v3 and do not redesign the state tokenizer yet. The new input
representation and causal-paired objective are operating as intended and have
crossed from action ignorance to meaningful causal geometry. The run should be
allowed to finish, but it is premature to describe the mechanics model as
converged or imagination-ready. If the completed run plateaus near cosine 0.75,
sub-unit training norm, and low decoded paired-effect F1, the next A/B should be
event-balanced causal training with optional action-field/source-target
pretraining—not a larger dynamics model and not replacement of paired effects
with an action autoencoder.

### Follow-up: close the representation-learning loop before scaling dynamics

The causal-paired runs have identified a workable transition objective. The
remaining design problem is now cleaner: make the state/action interface
lossless, compact, and spatially aligned *before* asking the dynamics transformer
to learn mechanics. This should reduce optimization burden and attention cost,
but it must be staged carefully so compression does not recreate the earlier
aliasing failure.

#### Two different problems are currently mixed together

1. **Action interface learning.** `ActionEventEncoder` is randomly initialized
   with the dynamics model. It must simultaneously learn an invertible
   representation of role/type/source/target/parameters, align its learned
   coordinate embeddings with the tokenizer's unrelated sinusoidal state
   coordinates, bind the event to the correct unit/cell, and learn the physical
   transition. The paired objective proves this is possible, but the 0.73-0.75
   cosine convergence tail suggests that interface learning is consuming real
   capacity and samples.
2. **Sequence inefficiency.** A direct transition is a dense 424-token causal
   sequence: 195 departure + 32 actions + 2 controls + 195 target slots. Of each
   195-token state, 128 positions are entity slots even though typical inspected
   states contain roughly 18 occupied units and the corpus maximum is 80. The
   dynamics padding mask currently masks invalid *action* slots but not empty
   state/entity slots. Those padding tokens therefore consume attention, can be
   attended to, and appear again as target outputs. A learned bottleneck is
   promising, but structural padding should be removed first.

#### Research synthesis

- [Perceiver IO](https://arxiv.org/abs/2107.14795) provides the most direct
  architectural template: a fixed latent array cross-attends to arbitrary-size
  structured input, performs expensive processing only in the compact latent,
  and uses semantic output queries to recover structured outputs. This fits a
  complete MicroRTS state better than pooling to one vector.
- [Set Transformer](https://arxiv.org/abs/1810.00825) shows how learned inducing
  points reduce set-attention cost from quadratic to linear in input-set size.
  This is relevant to the unordered set of simultaneous action events.
- [Binding Actions to Objects in World Models](https://arxiv.org/abs/2204.13022)
  finds that explicit action-to-object attention improves structured world
  models. For MicroRTS, source and destination bindings should therefore remain
  explicit even if the joint action set is later resampled to fewer tokens.
- [TokenLearner](https://arxiv.org/abs/2106.11297) demonstrates that adaptive
  learned tokens can reduce visual compute substantially, but its recognition
  results do not establish lossless mechanics. Token reduction here must be
  gated by exact fields and paired interventions, not downstream accuracy alone.
- [Delta-IRIS](https://arxiv.org/abs/2406.19320) addresses long world-model
  sequences by retaining continuous current-state tokens while modeling sparse
  stochastic deltas. This supports a direct unchanged-state skip and compact
  delta prediction for MicroRTS rather than regenerating the whole state.
- [EMERALD](https://openreview.net/forum?id=zNUOZcAUxz) reports that retaining a
  spatial latent state improves world-model accuracy. It is a warning against a
  single-vector bottleneck: compact tokens should remain typed/spatial or
  object-aware.
- Recent action-tokenizer work such as
  [X-Tokenizer](https://arxiv.org/abs/2606.14752) and
  [RepWAM](https://arxiv.org/abs/2606.13674) argues that reconstruction-only
  action codes are weak interfaces and benefits from semantic/next-feature
  alignment. These are recent VLA/WAM preprints rather than direct MicroRTS
  evidence, but they reinforce the local conclusion: exact action reconstruction
  is necessary, while state/action grounding is what makes the representation
  useful for dynamics.

#### Proposed action tokenizer

Keep the raw sparse event schema as the auditable truth, but replace the current
summed-embedding module with a separately pretrained action tokenizer:

1. Encode each event with factorized field embeddings whose subspaces remain
   distinct until a final MLP/attention projection. Do not sum every field into
   one vector at the first operation.
2. Use the same fixed Fourier/sinusoidal coordinate basis as the state tokenizer
   for source and destination, or explicitly align the two coordinate spaces.
3. Decode every valid event field: role, type, source x/y, destination x/y,
   direction, produced type, attack offset, validity, and action count. Apply
   conditional masks so inactive fields do not dominate.
4. Add source/target grounding: from an event token and frozen departure-state
   tokens, identify the source entity/cell and destination cell, and predict
   source owner/type plus target terrain/occupancy and legality. This pretrains
   the interface and binding without pretending to learn the full transition.
5. Add permutation augmentation or a set encoder so joint-action meaning does
   not depend on the current row-major packing index.
6. Freeze the accepted tokenizer for the cleanest dynamics experiment, but put
   a small trainable adapter/projection after it. A later A/B can unfreeze at a
   much lower learning rate if fixed action geometry becomes a ceiling.

The first action-tokenizer experiment should retain one token per event. Thirty-
two possible action tokens are not the primary sequence bottleneck, and
compressing them immediately risks erasing simultaneous source/target bindings.
Only after exact reconstruction and grounding pass should a Set
Transformer/Perceiver resampler compare 32, 16, and 8 joint-action latents.

Acceptance gates before freezing:

- exact valid-event field reconstruction, including explicit `TYPE_NONE`;
- exact action count and no overflow regression;
- near-perfect source and destination cell retrieval;
- invariance to event permutation;
- no degradation of paired action-effect prediction when a small probe consumes
  frozen action tokens instead of raw fields.

#### Solve token count in two stages

**Stage 0: remove artificial padding.** Return a validity mask for spatial,
entity, and global state tokens; prevent attention to empty entity slots; and
bucket/pack batches by entity count so kernels process a small tier such as
16/32/64/96/128 slots instead of always 128. This is high-confidence because it
changes no semantic representation. Merely masking fixed slots improves
stability; packing/bucketing is required for actual FLOP savings.

**Stage 1: introduce a reversible state bottleneck.** Add a Perceiver-style
compact state autoencoder above the frozen 195-token tokenizer:

`195 semantic state tokens -> M compact transition tokens -> 195 semantic queries`.

Learned latent queries cross-attend to valid semantic tokens; a query decoder
reconstructs the original typed positions. Start conservatively with
`M=128`, then test `96` and `64`. Do not jump to one vector or an arbitrarily
small latent: a map may contain 80 units with independent assignments/timers.

Pretraining this bottleneck requires more than latent MSE:

- reconstruct all 195 frozen tokenizer tokens;
- decode the complete canonical fields/globals through the frozen field decoder;
- preserve occupied/assignment/timer/target accuracy and legality;
- enforce cycle consistency `E_compact(D_compact(c)) ~= c`;
- preserve transition deltas and cloned-branch geometry, requiring
  `E(z_cf)-E(z_real)` to retain direction and magnitude;
- compare 128/96/64 against the uncompressed tokenizer and oracle tokenizer.

This creates a compact, Markov state `c_t` that can be rolled forward directly.
The 195 semantic tokens become an input/output interface and diagnostic decoder,
not the sequence processed by every dynamics layer.

#### Recommended compact transition architecture

After both interfaces pass independently:

```text
complete engine state
  -> frozen structured tokenizer (195 semantic tokens)
  -> frozen compact-state encoder (M state tokens)

sparse joint action
  -> frozen grounded action tokenizer (E event tokens)
  -> optional action-set resampler (K action tokens, only after ablation)

(c_t, a_t)
  -> compact transition core
  -> predicted delta_c
  -> c_t + delta_c = c_t+1
  -> compact-state decoder / canonical field heads
```

Use cross-attention between compact state queries and action events with explicit
source/target routing. An encoder-decoder or Perceiver IO layout is preferable
to concatenating departure and output tokens into one dense causal sequence:

- compact context latents cross-attend to state and action inputs;
- only compact latents use deep self-attention;
- typed next-state queries cross-attend to the compact context in parallel;
- residual/delta output preserves unchanged state by construction.

For `M=64`, the expensive self-attention operates on roughly 64 tokens instead
of 424. Cross-attention remains linear in the 195-token semantic interface, and
the full state need only be decoded at loss/control boundaries. This is a much
safer route to efficiency than discarding spatial/object tokens before proving
reconstruction.

#### How this enters direct dynamics and flow training

The training stack should become explicitly staged:

1. **State semantic tokenizer:** keep the current frozen structured autoencoder
   and re-audit its rare fields.
2. **Action tokenizer:** pretrain exact fields plus coordinate/source-target
   grounding; freeze with a small trainable adapter.
3. **Compact-state tokenizer:** pretrain reversible 195 -> M -> 195 compression
   with canonical reconstruction, cycle, and paired-delta preservation; freeze.
4. **Deterministic transition:** train only the compact transition core and
   adapters on factual, counterfactual, and paired-effect losses. Predict a
   residual delta, and retain decoded changed-cell/event heads.
5. **Open-loop grounding:** require one-step copy-last/event gates and then
   10/50/250-tick compact rollouts with periodic canonical decoding.
6. **Flow distillation:** only after the direct compact predictor passes, train
   flow/shortcut sampling on `c_t -> c_t+1` or on `delta_c`, using M target/noise
   slots instead of 195. Keep paired-effect loss in both compact and decoded
   canonical space so flow cannot erase action geometry.
7. **Control:** expose compact state to actor/critic only after autonomous
   mechanics remains valid through long assignment timers.

Flow should not be responsible for discovering the representation. Its job is
to approximate an already grounded compact transition with flexible sampling.
The deterministic direct predictor remains the teacher and acceptance baseline.

#### Controlled experiment ladder

| ID | change | required result |
|---|---|---|
| R0 | state/entity validity mask + bucketed packing | Same mechanics metrics, lower memory/FLOPs, less padding drift. |
| R1 | pretrained grounded action tokenizer, 32 event tokens | Faster cosine/norm/paired-F1 convergence at equal updates; exact action decoding. |
| R2 | compact state AE, M=128/96/64 | Near-lossless canonical fields and paired-delta geometry before dynamics. |
| R3 | compact residual dynamics with factual+paired loss | Beat the current v3 learning curve and copy-last/event gates at equal compute. |
| R4 | action resampler K=16/8 | Retain endpoint retrieval and paired effects; otherwise keep event tokens. |
| R5 | compact flow/shortcut distillation | Match direct predictor at K=1/4 and preserve multi-step exact/event validity. |

Run R0-R3 as separate ablations; do not combine every change into one run. The
v3 checkpoint and fixed evaluator become the baseline. Report wall-clock,
peak memory, tokens processed, changed-cell F1, paired-effect F1/cosine/norm,
copy-last-normalized event MSE, exact fields, and long-rollout validity.

#### Updated decision

These changes are worth implementing, but the confidence differs by component:

- **High confidence:** mask/drop empty entity slots, share/align coordinate
  representations, pretrain exact action fields plus endpoint grounding, and use
  a residual unchanged-state path. Each addresses a measured local failure with
  little semantic risk.
- **High confidence for efficiency, conditional for accuracy:** a Perceiver IO
  compact state is likely to reduce compute and improve optimization, provided
  it passes strict reconstruction and paired-delta gates. It must remain a set of
  typed tokens, not a single vector.
- **Medium confidence:** freezing the action tokenizer will simplify dynamics,
  but a small adapter or low-rate unfreezing may be needed because an invertible
  action geometry is not automatically the easiest transition geometry.
- **Low confidence until ablated:** compressing the joint action below one token
  per issued event. Action tokens are a small part of the current cost and their
  source/target bindings are causally important.

The project has found the right *learning contract*: complete Markov state,
privileged joint actions, cloned interventions, and direct paired effects. The
next step is not a new objective or a larger model. It is to turn that contract
into pretrained, reversible state/action interfaces and let the dynamics core
learn only the compact transition. The difficulty encountered here is useful:
future Dreamer-style systems will need the same separation between semantic
tokenization, action grounding, transition learning, and generative
distillation if they are expected to simulate exact interactions rather than
produce visually plausible futures.

## R1 multi-event action-tokenizer implementation and experiment (2026-07-14)

The first isolated representation experiment is now implemented. It deliberately
does **not** include state-token compression or action-set resampling: dynamics
still receives 195 departure tokens, 32 sparse action-event slots, two controls,
and 195 target slots. This isolates action-interface pretraining from the larger
sequence-efficiency proposal.

The new action encoder produces one 512-D embedding per event rather than one
joint-action vector. Role, type, direction, produced unit, source x/y,
destination x/y, and attack offset are embedded in separate 64-D subspaces,
concatenated, and projected through an MLP. Raw JVM unit IDs remain excluded.
The no-issued-action sentinel remains an explicit valid event. Learned event-slot
positions are part of the transferable representation.

`train_action_tokenizer.py` freezes `structured_tokenizer_v2.pt` and trains the
action representation on the same factual and cloned-counterfactual HDF5 rows
with four complementary losses:

1. exact per-slot event validity and field reconstruction;
2. forward prediction of the normalized frozen-state latent delta from
   `(state_t, action_tokens)`;
3. inverse reconstruction of issued action slots from
   `(state_t, state_t+1, delta)` plus alignment to encoded action tokens;
4. direct prediction of cloned intervention geometry by matching the difference
   between forward predictions under factual and alternative actions.

This is stronger than a plain action autoencoder: exact decoding makes the code
auditable, the inverse path requires the representation to be recoverable from
observed mechanics, and the forward/paired paths require it to expose effects on
the environment. The disposable SSL heads are not loaded into dynamics. Only
the factorized event encoder and event-slot positions are transferred and frozen;
the 50M transition transformer, state/action projections, and output head remain
trainable.

The controlled experiment has three runs:

| order | config | purpose |
|---|---|---|
| 1 | `pretrain_structured_action_tokenizer_v2` | 100k-step dual forward/inverse SSL pretraining and checkpoint selection. |
| 2 | `pretrain_structured_dynamics_v2_causal_paired_action_scratch` | 160k-step factorized-encoder control from random initialization. |
| 3 | `pretrain_structured_dynamics_v2_causal_paired_action_pretrained` | Identical 160k transition run with the accepted action encoder frozen. |

Both dynamics configs retain the successful causal coefficients `(1, 1, 4)`,
8x changed-token weighting, zero-noise one-step query, batch 32, `1e-4` initial
learning rate, 1k warmup, and the proven three-stage LR envelope: 60k steps
decaying `1e-4 -> 1e-5`, a 500-step smooth restart toward `3e-5` followed by a
40k tail to `1e-5`, then 60k steps to `2.5e-6`. The longer clean schedule is
intentional: completed v3 remained causally healthy but its final
fixed-validation values still oscillated around paired-CF latent MSE `0.11`,
effect cosine `0.78`, and non-calibrated per-row norm ratios after the low-LR
tail. R1 asks whether removing action-interface learning from the transition
core reaches those gates sooner and then moves beyond them.

Acceptance is comparative, not merely low reconstruction loss. The pretrained
treatment should reach matched paired-CF/effect-cosine gates in fewer updates,
improve changed-cell and decoded paired-effect F1 at equal steps, and retain
near-exact event decoding. If it starts faster but plateaus below the scratch
control, the frozen geometry is a ceiling; the next A/B should unfreeze through
a low-rate adapter. State/action compression remains deferred until this test
closes.

## Completed causal-paired v3 mechanics audit and R1 tuning (2026-07-14)

WandB run `ikj03zhs` completed all 60k continuation steps. The fixed validation
curve continued improving long after the previous v2 checkpoint but remained
noisy: best total validation loss was `1.2666` at step 51.5k, best paired probe
MSE was `0.1083` at step 52.5k, and best reported effect cosine was `0.7932` at
step 59.5k. These optima being close but non-identical is expected for sparse
interventions; selecting by total validation loss remains reasonable because it
does not sacrifice factual mechanics for one paired statistic.

A separate deterministic 512-transition evaluator slice at the deployed
one-step query gives:

| checkpoint | latent MSE | copy MSE | occupied type acc | exact cell | changed F1 | self shuffle gap | paired latent MSE | paired effect F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 best | 0.1355 | 0.1120 | 0.9273 | 0.9388 | 0.7923 | 0.0172 | 0.1581 | 0.0813 |
| v3 best | 0.1252 | 0.1120 | 0.9560 | 0.9407 | **0.8104** | **0.0196** | 0.1481 | **0.0802** |
| v3 final | **0.1226** | 0.1120 | **0.9572** | 0.9405 | 0.8072 | 0.0194 | **0.1461** | 0.0759 |

The world model is useful but not yet imagination-ready. It predicts presence,
unit type, and changed-cell support well, and shuffled actions measurably hurt,
so this is not action collapse. Relative to v2, v3 improves broad latent error,
type accuracy, changed-cell F1, and paired latent error. However, latent MSE
still does not beat copy-last, and decoded paired-effect F1 is flat near `0.08`.
The low-LR tail is refining common-state reconstruction more reliably than exact
intervention content/location.

The 160k budget for R1 is therefore retained: it equals the successful
`60k + 40k + 60k` causal-paired sequence. The original single 160k cosine was
not retained because it would keep average LR roughly 1.8x above the schedule
that produced v3 and postpone the useful convergence tail. Both new dynamics
configs now use one resumable staged scheduler matching that measured LR
envelope, including a smooth 500-step restart before the middle stage. Causal
weights remain `(1, 1, 4)` so the scratch/pretrained comparison isolates action
representation rather than confounding it with a stronger effect objective.
New runs also report an aggregate effect-norm ratio (sum of predicted norms over
sum of target norms). Keep the legacy mean per-row ratio for continuity, but use
the aggregate value for calibration decisions because tiny-effect rows caused
the observed spikes above one.

## Grounded residual dynamics and aligned causal geometry (2026-07-14)

The live frozen-action treatment showed that action conditioning was real but
that the transition core was relearning a bridge already present during action
SSL. At step 102k its independent paired geometry reached overall cosine
`0.828` and entity cosine `0.862`, but spatial cosine remained only `0.618`
versus `0.764` for the completed v3 reference. Occupied-unit type accuracy was
also far below v3. This localized the failure to action-to-spatial routing and
absolute next-latent fitting, not erased action fields: the action tokenizer
still decoded valid events, types, sources, and targets essentially exactly.

Inspection found that the dynamics handoff transferred only
`action_encoder` and `action_position`. The action tokenizer's pretrained
`forward_state`, state-to-action cross-attention, normalization, and latent-delta
head were discarded. The 50M transformer therefore had to rediscover the
coordinate bridge while also regenerating all 195 arrival tokens from zero.
That is unnecessarily difficult for a deterministic, sparse transition.

The new experiment is
`micro-rts/pretrain_structured_dynamics_v2_causal_paired_action_grounded_residual`.
It is an opt-in, checkpoint-compatible extension with six changes:

1. transfer the complete pretrained forward router and initialize prediction as
   `z_next = z_departure + delta_pretrained + delta_correction`;
2. keep the transition correction head zero-initialized, preserving the
   pretrained geometry exactly at step zero, and update the router at 0.1x the
   core learning rate while the event encoder remains frozen;
3. scatter separately projected source and destination event features into the
   corresponding 8x8 spatial output queries, with zero-initialized projections;
4. mask empty packed-entity departure slots from transformer key/value
   attention without masking target queries needed for births/deaths;
5. oversample stored paired rows to 50% of each storage-local batch and add
   explicit cosine, robust global log-norm, and decoded canonical field losses;
6. retain an exact identity path for unchanged state and train the transformer
   only on the residual correction.

### Aligned-batch numerical bug

The audit also found a subtle source of fake causal effects. Both action SSL and
paired dynamics encoded factual arrivals at the full batch size but encoded
counterfactual arrivals only after subselecting paired rows. Attention kernels
with different batch shapes have slightly different floating-point roundoff.
For an engine-zero intervention, two semantically identical states could
therefore differ by a tiny latent vector; the old `norm > 1e-8` geometry test
then classified that numerical residue as a real action effect. This polluted
effect metrics/losses and contributed to extreme norm-ratio spikes.

Counterfactual states are now assembled as a full aligned batch, with invalid
rows replaced by their factual arrivals, encoded once at the same shape as the
factual batch, and subselected only afterward. The validation loop also sums
cosines and predicted/target norms over the complete validation set instead of
averaging per-batch ratios. The explicit norm loss uses a global log-ratio so a
tiny target cannot create an unbounded gradient.

### Step-zero checkpoint-backed geometry

`outputs/validate_grounded_residual_setup.py` loaded the real 100k action
tokenizer, verified zero correction/routing weights, and measured the
transferred residual prior before dynamics training. On the independent first
2,048 transitions (414 paired rows, 131 true nonzero effects):

| initialization/reference | factual MSE | copy MSE | overall cosine | overall norm | entity cosine | spatial cosine | spatial norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| completed v3 best | 0.1252* | 0.1120* | 0.855 | 0.898 | 0.869 | 0.764 | 0.763 |
| grounded residual step zero | 0.1017 | 0.0952 | **0.857** | **0.945** | 0.856 | **0.871** | **0.906** |

`*` The v3 factual/copy figures are from the established 512-transition
mechanics slice; geometry figures use the 2,048-transition audit. They are
included as gates, not as a claim of identical factual sampling.

On the new 2,048-transition fixed validation slice, the transferred prior beats
copy-last (`0.1552` versus `0.1665` latent MSE), with overall cosine `0.848`,
aggregate norm `0.972`, and spatial cosine `0.899`. The independent prefix is
slightly worse than copy on broad factual MSE, so the learned correction and
canonical grounding still have real work to do; the initialization is not being
mistaken for a finished world model. Global effects remain tiny and noisy, which
is why decoded resource/timer grounding is retained.

Tensor-level tests additionally verify that the new world model's step-zero
`prior_delta` is exactly equal to the original action tokenizer's forward
prediction, that only source/destination spatial slots receive explicit routing,
that empty entity slots are masked, that zero engine effects remain zero after
aligned encoding, and that the canonical loss backpropagates through the frozen
decoder. The focused structured/dataloader suite passes all 32 tests, and the
full configured causal-paired path completes a two-step CPU smoke run.

### Why this is the optimal next deterministic model

This is the smallest architecture that makes the known deterministic structure
easy rather than asking scale to compensate for a poor interface. The identity
path solves unchanged state exactly; the accepted action SSL model supplies a
causally trained first delta; explicit endpoint routing supplies the missing
spatial binding; canonical heads enforce discrete mechanics; balanced pairs
raise intervention density; and the transformer is reserved for interactions,
timers, collisions, simultaneous events, and corrections. No state or action
information is compressed further, and no larger model or stochastic flow loss
is introduced. Each addition is independently configurable, so failures remain
ablatable.

The 160k schedule is retained for a fair comparison. Promotion remains strict:
the trained model must beat copy-last on both fixed and independent slices,
exceed v3 overall/entity/spatial geometry with norm near one, recover occupied
type accuracy near one, improve decoded paired-effect F1, and then pass
10/50/250-tick deterministic rollouts. Near-perfect mechanics is a target made
plausible by this factorization, not something inferred from the step-zero
router audit.

## Residual-prior drift and trust-region correction (2026-07-15)

The first grounded-residual run was externally stopped at step 95.9k. It did
not fail numerically after the BF16 routing fix, but its best world-model state
occurred extremely early. At step 1k, unweighted validation MSE was `0.1449`,
real-probe MSE `0.1206`, and paired-CF MSE `0.0982`. By step 2k these had already
regressed to `0.1914`, `0.1649`, and `0.1432`. The monitored composite continued
falling to `1.2286` at step 94k primarily because canonical grounding loss fell
from `0.898` to `0.270`; padding MSE simultaneously rose from `0.015` to `0.083`.
Thus `val/wm/total` rewarded a different latent compromise while broad state
fidelity moved away from the useful pretrained residual prior.

The action geometry itself did not collapse. Overall/spatial/entity effect
cosines remained approximately `0.86/0.87/0.86`. Instead, two trainable branches
co-adapted: by 95k the transferred router attention and forward head had moved
`5.8%` and `8.0%` in relative parameter norm, and the zero-initialized correction
head had grown to norm `4.62`. A checkpoint ablation on the common
512-transition mechanics slice made the failure causal rather than correlational:

| 95k ablation | latent MSE | copy MSE | exact cell | changed F1 | paired latent MSE |
|---|---:|---:|---:|---:|---:|
| full co-adapted model | 0.1204 | 0.1120 | 0.9389 | 0.7968 | 0.1444 |
| remove correction only | 0.1365 | 0.1120 | 0.9399 | 0.8045 | 0.1597 |
| reset router only | 0.1538 | 0.1120 | 0.9453 | 0.8524 | 0.1763 |
| reset router and correction | **0.1077** | 0.1120 | **0.9512** | **0.9121** | **0.1301** |

Resetting either half alone is harmful because they co-adapted, while restoring
both recovers a model that beats copy-last and is substantially better at
decoded changed mechanics. The regression was therefore caused by an
underconstrained learned correction around a moving prior, amplified by padding
weight `0.05`, changed-token boost `8`, direct grounding, and a checkpoint metric
that did not protect unweighted fidelity.

The successor experiment is
`micro-rts/pretrain_structured_dynamics_v2_causal_paired_action_residual_trust_region`.
It freezes the complete pretrained router, lowers peak LR from `1e-4` to `1e-5`,
penalizes factual and counterfactual correction magnitude, restores padding
weight to `1.0`, reduces changed-token boost to `2`, and disables direct
grounding until latent prediction has improved over the frozen prior. It also
monitors `val/causal/unweighted_mse` rather than the composite objective and uses
a shorter 40k budget with 500-step validation.

The new optional `residual_correction_coef` is a true trust region: the
transformer can still learn interactions that justify leaving the prior, but
every correction pays for its squared normalized-latent magnitude. The new
config uses coefficient `0.5`. Production CUDA BF16 smoke training passes, the
router remains exactly frozen, and the step-zero audits retain prefix geometry
`0.857/0.945` cosine/norm and fixed-validation geometry `0.865/0.977`, with
fixed-validation factual MSE `0.1569` beating copy `0.1647`. Canonical grounding
should only return as a separately gated fine-tune after unweighted, paired-CF,
decoded mechanics, and rollout gates all improve over this frozen-prior floor.
