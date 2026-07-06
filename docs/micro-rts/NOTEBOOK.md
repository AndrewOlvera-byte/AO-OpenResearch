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
