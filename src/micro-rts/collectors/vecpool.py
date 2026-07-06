"""``MultiprocessPool`` — W worker processes, each owning its own JVM + MicroRTS env,
aggregated behind the single ``VecEnv`` interface.

This is the real cross-process throughput tier (one JVM per process is mandatory —
JPype can't restart a JVM). Bulk data (obs/reward/done/action) lives in a single
shared-memory ``TensorDict``; each worker writes its contiguous row-slice. A per-worker
``Pipe`` carries only tiny control signals.

The API is deliberately split so ``send`` returns immediately while workers step in the
background and ``recv`` blocks for the result — this is exactly the primitive the Phase 3
async collector uses to overlap collection with the policy update.

Note: workers step in lock-step (barrier on ``recv``); the "return first-ready N of M"
straggler optimization from PufferLib's EnvPool is a later refinement and does not change
this interface.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.multiprocessing as mp
from tensordict import TensorDict

from environments.base import VecEnv
from environments.microrts_env import EnvConfig, MicroRTSVecEnv


def _probe(cfg: EnvConfig, conn) -> None:
    """Build an env once just to report its spaces, then exit.

    Send plain Python types only — a torch tensor over a pipe uses fd-passing that
    needs the sender alive, and this process exits immediately.
    """
    env = MicroRTSVecEnv(cfg)
    conn.send((env.obs_shape, env.action_nvec.tolist()))
    conn.close()


def _worker(cfg: EnvConfig, conn, shared: TensorDict) -> None:
    env = MicroRTSVecEnv(cfg)
    conn.send(True)  # ready handshake
    try:
        while True:
            cmd = conn.recv()
            if cmd == "close":
                break
            t = env.reset() if cmd == "reset" else env.step(shared["action"])
            shared["obs"].copy_(t["obs"])
            shared["reward"].copy_(t["reward"])
            shared["done"].copy_(t["done"])
            conn.send(True)
    finally:
        env.close()
        conn.close()


class MultiprocessPool(VecEnv):
    def __init__(self, cfg: EnvConfig, num_workers: int):
        assert cfg.num_envs % num_workers == 0, "num_envs must divide evenly across workers"
        per = cfg.num_envs // num_workers
        self.num_envs = cfg.num_envs
        worker_cfg = replace(cfg, num_envs=per)

        ctx = mp.get_context("spawn")
        try:
            mp.set_sharing_strategy("file_system")
        except (RuntimeError, ValueError):
            pass

        # Probe spaces (only a JVM-owning process knows them) before sizing shared mem.
        pconn, cconn = ctx.Pipe()
        probe = ctx.Process(target=_probe, args=(worker_cfg, cconn), daemon=True)
        probe.start()
        self.obs_shape, nvec = pconn.recv()
        self.action_nvec = torch.tensor(nvec, dtype=torch.long)
        probe.join()

        self.shared = TensorDict(
            {
                "obs": torch.zeros(self.num_envs, *self.obs_shape),
                "reward": torch.zeros(self.num_envs),
                "done": torch.zeros(self.num_envs, dtype=torch.bool),
                "action": torch.zeros(self.num_envs, len(self.action_nvec), dtype=torch.long),
            },
            batch_size=[self.num_envs],
        ).share_memory_()
        self.env_id = torch.arange(self.num_envs)

        self.conns, self.procs = [], []
        for w in range(num_workers):
            sl = slice(w * per, (w + 1) * per)
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker, args=(worker_cfg, child_conn, self.shared[sl]), daemon=True
            )
            proc.start()
            self.conns.append(parent_conn)
            self.procs.append(proc)
        for c in self.conns:  # wait for ready handshakes
            c.recv()

    def async_reset(self, seed: int | None = None) -> None:
        for c in self.conns:
            c.send("reset")

    def send(self, actions: torch.Tensor) -> None:
        self.shared["action"].copy_(actions.detach().to("cpu").view(self.num_envs, -1).long())
        for c in self.conns:
            c.send("step")

    def recv(self) -> TensorDict:
        for c in self.conns:  # barrier: wait for every worker's ack
            c.recv()
        out = self.shared.clone()
        out.set("env_id", self.env_id.clone())
        out.set("trunc", out["done"].clone())
        return out

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send("close")
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
