"""``Runner`` — config-toggleable training loop for throughput experiments.

Two modes, switched by ``RunConfig.overlap``:

- ``overlap=False`` (synchronous): collect a rollout, then update. Strictly on-policy,
  no lag — the debugging / ablation baseline.
- ``overlap=True`` (double-buffered, APPO-style): a background thread collects the next
  rollout using a frozen *actor* snapshot while the main thread runs the GPU update on
  the *learner*. Two RolloutBuffers ping-pong so reader/writer never touch the same one.
  This works because gym_microrts' ``env.step`` releases the GIL (Java side), so CPU/Java
  collection genuinely overlaps GPU training. Cost: bounded 1-iteration policy lag.

The actor/learner split (separate inference vs optimization copies) is what makes the
concurrent read (collector) / write (optimizer) race-free; the actor is refreshed from
the learner only while the collector is idle.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from environments.microrts_env import EnvConfig, MicroRTSVecEnv
from models.cnn_policy import CNNPolicy
from models.random_policy import RandomPolicy
from trainers.PPOTrainer import PPOTrainer

from .buffer import RolloutBuffer
from .collector import Collector


@dataclass
class RunConfig:
    # env / scale — defaults are the sweep-optimal setup on an RTX 5070 Ti
    # (~10k steps/s; see docs/micro-rts/NOTEBOOK.md throughput sweep).
    num_envs: int = 256
    horizon: int = 128
    mode: str = "bot"                       # "bot" | "selfplay"
    bots: tuple = ("randomBiasedAI",)
    backend: str = "serial"                 # "serial" | "multiprocess"
    num_workers: int = 4                    # multiprocess only
    # learning
    policy: str = "cnn"                     # "cnn" | "random"
    device: str = "cuda"
    epochs: int = 4
    minibatches: int = 4
    # throughput toggle
    overlap: bool = True
    iters: int = 20


def _build_env(cfg: RunConfig):
    env_cfg = EnvConfig(num_envs=cfg.num_envs, mode=cfg.mode, bots=cfg.bots)
    if cfg.backend == "multiprocess":
        from .vecpool import MultiprocessPool
        return MultiprocessPool(env_cfg, num_workers=cfg.num_workers)
    return MicroRTSVecEnv(env_cfg)


def _build_policy(cfg: RunConfig, env):
    if cfg.policy == "random":
        return RandomPolicy(env.action_nvec, cfg.device)
    return CNNPolicy(env.obs_shape, env.action_nvec, cfg.device)


class Runner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.env = _build_env(cfg)
        self.learner = _build_policy(cfg, self.env)
        # In overlap mode collection must read a snapshot, not the live learner.
        self.actor = _build_policy(cfg, self.env) if cfg.overlap else self.learner
        self.collector = Collector(self.env, self.actor, cfg.horizon, cfg.device)
        self.trainable = cfg.policy == "cnn"
        self.trainer = (
            PPOTrainer(self.learner, epochs=cfg.epochs, minibatches=cfg.minibatches)
            if self.trainable else None
        )
        self._bufs = [self._new_buffer(), self._new_buffer()]

    def _new_buffer(self):
        c = self.cfg
        return RolloutBuffer(c.horizon, self.env.num_envs, self.env.obs_shape,
                             len(self.env.action_nvec), c.device)

    def _sync_actor(self):
        if self.actor is not self.learner:
            self.actor.load_state_dict(self.learner.state_dict())

    def _update(self, buf):
        if self.trainer is not None:
            self.trainer.update(buf)

    def run(self) -> dict:
        runner = self._run_overlap if self.cfg.overlap else self._run_sync
        t0 = time.perf_counter()
        runner()
        dt = time.perf_counter() - t0
        steps = self.cfg.iters * self.cfg.horizon * self.env.num_envs
        return {"sps": steps / dt, "iters_per_s": self.cfg.iters / dt,
                "wall_s": dt, "steps": steps}

    def _run_sync(self):
        for _ in range(self.cfg.iters):
            self._sync_actor()
            self._update(self.collector.collect(self._bufs[0]))

    def _run_overlap(self):
        self._sync_actor()
        buf = self.collector.collect(self._bufs[0])      # prime
        cur = 0
        with ThreadPoolExecutor(max_workers=1) as pool:
            for _ in range(self.cfg.iters):
                nxt = 1 - cur
                future = pool.submit(self.collector.collect, self._bufs[nxt])
                self._update(buf)                         # GPU train on current
                nxt_buf = future.result()                # wait for background collect
                self._sync_actor()                        # collector idle -> safe to refresh
                buf, cur = nxt_buf, nxt

    def close(self):
        self.env.close()
