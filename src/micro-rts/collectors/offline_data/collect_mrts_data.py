"""CLI: collect offline MicroRTS rollouts for DreamerV4 pretraining (format v3).

v3 data contract (NEXT_PLAN.md): every step stores BOTH players' gridnet actions
— the scripted opponent's via the patched jar's ``opponentAction`` field, the
self-play partner's natively — plus per-trajectory provenance (``policy_id``
into the ``policies`` legend, ``action_noise`` = the ε used). The corpus is a
**collection matrix**: blocks of different player-1 controllers, ε-noise levels,
seats and opponent types, all in ONE store so a single loader glob covers the
whole mix. Diversity of *policies* (not just states) is what makes the joint
action channels identifiable for the world model.

Two ways to specify the matrix:

1. ``--plan`` (repeatable): one block per flag, comma-separated ``key=value``::

       --plan mode=bot,policy=masked_random,steps=2000
       --plan mode=bot,policy=ckpts/ppo_best.pt,eps=0.15,steps=8000,seats=mix
       --plan mode=selfplay,policy=ckpts/ppo_best.pt,eps=0.05,steps=3000

   Keys: ``mode`` (bot|selfplay), ``policy`` ('masked_random' or a checkpoint
   path), ``steps`` (timesteps per lane), ``eps`` (per-cell ε-greedy noise,
   default 0), ``seats`` (bot mode: ``0`` = Python is player 0 everywhere
   [default], ``mix`` = alternate seats across lanes so the scripted bot also
   plays the player-1 role — needs the patched jar).

2. ``--preset-v3``: expands to the NEXT_PLAN.md target mix over
   ``--total-steps`` (per lane): ~20% clean strong play (seats mixed), ~40%
   ε-noised strong play (ε 0.05/0.15/0.3), ~15% weak/mid checkpoint, ~15%
   self-play (both channels non-scripted), ~10% pure masked-random::

       docker compose exec research python \
         src/micro-rts/collectors/offline_data/collect_mrts_data.py \
         --name tokdyn_pretrain_v3 --num-envs 24 --preset-v3 \
         --strong-ckpt checkpoints/league_hard/best.pt \
         --mid-ckpt checkpoints/league_hard/latest_early.pt \
         --total-steps 60000

Default opponents cycle randomBiasedAI / workerRushAI / lightRushAI — coacAI is
deliberately HELD OUT of pretraining (eval-only, as before). Self-play lanes are
tagged with the ``self`` opponent-legend entry.

NOTE: envs are cached per (map, mode, seats) and never closed mid-run —
gym_microrts's ``close()`` shuts down the whole JVM, which cannot restart
in-process. The OS reclaims everything at exit.

Output: ``/data/micro-rts/<name>__<UTC>__<git8>.h5``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Runnable as a bare script (`python .../collect_mrts_data.py`): put the package
# roots on sys.path so `collectors.*`/`environments.*`/`core.*` import, matching
# the pytest pythonpath (src/micro-rts, src).
if __package__ in (None, ""):
    _here = Path(__file__).resolve()
    sys.path.insert(0, str(_here.parents[2]))  # src/micro-rts
    sys.path.insert(0, str(_here.parents[3]))  # src

# Held-out opponent: coacAI stays eval-only (never in the pretraining corpus).
DEFAULT_BOTS = ["randomBiasedAI", "workerRushAI", "lightRushAI"]


def _git_sha() -> str:
    # ``-c safe.directory=*`` so it still works when the repo is a bind mount
    # owned by a different uid inside the container (git's dubious-ownership guard).
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", "rev-parse", "--short=8", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


@dataclass
class Block:
    """One collection-matrix block (see module docstring)."""
    mode: str = "bot"            # bot | selfplay
    policy: str = "masked_random"
    steps: int = 0               # timesteps per lane
    eps: float = 0.0             # per-cell ε-greedy noise on the base policy
    seats: str = "0"             # bot mode: "0" | "mix"

    def policy_name(self) -> str:
        base = "masked_random" if self.policy in ("masked_random", "random") \
            else Path(self.policy).stem
        return base

    def describe(self) -> str:
        s = f"{self.mode}:{self.policy_name()}"
        if self.eps > 0:
            s += f":eps{self.eps:g}"
        if self.mode == "bot" and self.seats == "mix":
            s += ":seatsmix"
        return s


def parse_plan(spec: str) -> Block:
    kw = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.partition("=")
        kw[k.strip()] = v.strip()
    b = Block(
        mode=kw.get("mode", "bot"),
        policy=kw.get("policy", "masked_random"),
        steps=int(kw.get("steps", 0)),
        eps=float(kw.get("eps", 0.0)),
        seats=kw.get("seats", "0"),
    )
    if b.mode not in ("bot", "selfplay"):
        raise SystemExit(f"--plan: bad mode {b.mode!r} in {spec!r}")
    if b.steps <= 0:
        raise SystemExit(f"--plan: steps must be > 0 in {spec!r}")
    if b.seats not in ("0", "mix"):
        raise SystemExit(f"--plan: seats must be '0' or 'mix' in {spec!r}")
    return b


def preset_v3(total: int, strong: str | None, mid: str | None) -> list[Block]:
    """The NEXT_PLAN.md corpus mix, as per-lane step counts over ``total``."""
    if not strong:
        raise SystemExit("--preset-v3 needs --strong-ckpt (a trained PPO gridnet "
                         "checkpoint, e.g. from base_rlFS_expert_masked_league)")
    frac_mid = 0.15 if mid else 0.0
    blocks = [
        Block("bot", strong, int(0.20 * total), 0.0, "mix"),     # clean strong
        Block("bot", strong, int(0.10 * total), 0.05, "0"),      # light noise
        Block("bot", strong, int((0.20 + (0.0 if mid else 0.05)) * total),
              0.15, "mix"),                                      # the workhorse
        Block("bot", strong, int(0.10 * total), 0.30, "0"),      # heavy noise
        Block("selfplay", strong, int(0.15 * total), 0.05),      # both channels real
        Block("bot", "masked_random",
              int((0.10 + (0.0 if mid else 0.10)) * total), 0.0, "0"),  # flailing
    ]
    if mid:
        blocks.insert(4, Block("bot", mid, int(0.075 * total), 0.0, "mix"))
        blocks.insert(5, Block("bot", mid, int(0.075 * total), 0.15, "0"))
    return [b for b in blocks if b.steps > 0]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True,
                   help="collection name; the file is <name>__<UTC>__<git8>.h5")
    p.add_argument("--out-dir", default="/data/micro-rts",
                   help="output directory (default: /data/micro-rts)")
    p.add_argument("--maps", nargs="+",
                   default=["maps/16x16/basesWorkers16x16.xml"],
                   help="map xml paths (must share a grid size)")
    p.add_argument("--bots", nargs="+", default=list(DEFAULT_BOTS),
                   help="scripted player-2 opponents, cycled across env lanes "
                        "(coacAI is held out by default — keep it eval-only)")
    p.add_argument("--num-envs", type=int, default=24,
                   help="parallel game lanes per env (single in-process JVM)")
    p.add_argument("--plan", action="append", default=[], metavar="SPEC",
                   help="collection-matrix block: mode=bot|selfplay,policy=...,"
                        "steps=N[,eps=E][,seats=0|mix] (repeatable)")
    p.add_argument("--preset-v3", action="store_true",
                   help="expand the NEXT_PLAN.md v3 mix over --total-steps")
    p.add_argument("--strong-ckpt", default=None,
                   help="preset-v3: trained PPO gridnet checkpoint (strong play)")
    p.add_argument("--mid-ckpt", default=None,
                   help="preset-v3: early/mid-training checkpoint (optional)")
    p.add_argument("--total-steps", type=int, default=60000,
                   help="preset-v3: total timesteps per lane across all blocks")
    # Legacy single-policy flags (used when neither --plan nor --preset-v3 given).
    p.add_argument("--steps-per-map", type=int, default=20000,
                   help="legacy: timesteps per lane per map for a single block")
    p.add_argument("--policy", default="masked_random",
                   help="legacy: 'masked_random' or a checkpoint path for player 1")
    p.add_argument("--steps-per-segment", type=int, default=512,
                   help="writer flush granularity (bounds writer RAM)")
    p.add_argument("--max-episode-steps", type=int, default=2000,
                   help="engine episode cap before auto-reset")
    p.add_argument("--policy-device", default="cpu")
    p.add_argument("--gzip", type=int, default=4, help="gzip level 0-9")
    p.add_argument("--chunk-rows", type=int, default=256,
                   help="HDF5 chunk size along the time axis")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward-weight", nargs=6, type=float, default=None,
                   help="6 reward-component weights (default: env default)")
    p.add_argument("--full-state", action=argparse.BooleanOptionalAction, default=True,
                   help="collect HDF5 v4 complete engine transitions (default: on; "
                        "use --no-full-state only for legacy v3 collection)")
    p.add_argument("--counterfactual-frac", type=float, default=0.15,
                   help="fraction of rows marked with a cloned-engine masked-random "
                        "self-action branch (default: 0.15; requires --full-state)")
    return p.parse_args(argv)


def build_output_path(out_dir: str, name: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{name}__{stamp}__{_git_sha()}.h5"


def main(argv=None) -> int:
    import numpy as np
    import torch

    from environments.dream_env import DreamEnv
    from environments.microrts_env import DEFAULT_REWARD_WEIGHT, EnvConfig, MicroRTSVecEnv

    from collectors.offline_data.collector import OfflineCollector
    from collectors.offline_data.HDF5Writer import HDF5Writer
    from collectors.offline_data.policies import EpsilonGreedyPolicy, load_policy

    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.preset_v3:
        blocks = preset_v3(args.total_steps, args.strong_ckpt, args.mid_ckpt)
    elif args.plan:
        blocks = [parse_plan(s) for s in args.plan]
    else:  # legacy single-policy collection
        blocks = [Block("bot", args.policy, args.steps_per_map, 0.0, "0")]

    rw = tuple(args.reward_weight) if args.reward_weight else DEFAULT_REWARD_WEIGHT
    if args.counterfactual_frac and not args.full_state:
        raise SystemExit("--counterfactual-frac requires --full-state")
    n = args.num_envs
    if any(b.mode == "selfplay" for b in blocks):
        assert n % 2 == 0, "selfplay blocks need an even --num-envs"

    # Legends. Opponents: the bots + 'self' for self-play lanes. Policies: one
    # entry per distinct (controller, seat-role) so provenance filters can slice
    # seat-swapped play; ε lives in traj/action_noise, not the legend.
    opponents = list(args.bots) + ["self"]
    self_opp = len(args.bots)
    policy_names: list[str] = []

    def legend_id(name: str) -> int:
        if name not in policy_names:
            policy_names.append(name)
        return policy_names.index(name)

    for b in blocks:
        legend_id(b.policy_name())
        if b.mode == "bot" and b.seats == "mix":
            legend_id(b.policy_name() + "#p1seat")

    out_path = build_output_path(args.out_dir, args.name)
    print(f"[collect] writing -> {out_path}", flush=True)
    print(f"[collect] maps={args.maps} bots={args.bots} num_envs={n}", flush=True)
    for i, b in enumerate(blocks):
        print(f"[collect]   block {i}: {b.describe()} steps/lane={b.steps} "
              f"(={b.steps * n} transitions/map)", flush=True)

    # Env cache per (map_id, mode, seats): envs are NEVER closed mid-run (see
    # module docstring — gym close() kills the JVM for good).
    envs: dict[tuple, DreamEnv] = {}

    def get_env(map_id: int, map_path: str, block: Block) -> DreamEnv:
        key = (map_id, block.mode, block.seats if block.mode == "bot" else "-")
        if key not in envs:
            if block.mode == "selfplay":
                cfg = EnvConfig(num_envs=n, map_path=map_path,
                                max_steps=args.max_episode_steps, mode="selfplay",
                                reward_weight=rw, gridnet=True,
                                full_state=args.full_state)
            else:
                seats = tuple(i % 2 for i in range(n)) if block.seats == "mix" else None
                cfg = EnvConfig(num_envs=n, map_path=map_path,
                                max_steps=args.max_episode_steps, mode="bot",
                                bots=tuple(args.bots), reward_weight=rw,
                                gridnet=True, opponent_action=True,
                                player_ids=seats, full_state=args.full_state)
            envs[key] = DreamEnv(MicroRTSVecEnv(cfg))
        return envs[key]

    writer = None
    shapes = None
    t0 = time.time()
    total_transitions = 0
    try:
        for map_id, map_path in enumerate(args.maps):
            for block in blocks:
                env = get_env(map_id, map_path, block)
                base_env = env.env
                n_comp = len(base_env.action_nvec) - 1      # gridnet per-cell comps (7)
                cur = (tuple(base_env.obs_shape),
                       (base_env._grid_cells, n_comp),
                       tuple(base_env.mask_shape))
                if writer is None:
                    shapes = cur
                    # Store the true terminal arrival frame iff the env surfaces
                    # it (requires the patched jar exposing pre-reset frames).
                    has_term = "terminal_obs" in env.reset().keys()
                    if not has_term:
                        print("[collect] WARNING: env does not expose terminal_obs; "
                              "continue/win-reward supervision will be unavailable")
                    writer = HDF5Writer(
                        out_path,
                        obs_shape=cur[0], action_shape=cur[1], mask_shape=cur[2],
                        action_nvec=base_env.action_nvec.tolist(),
                        grid_hw=base_env.obs_shape[1:], reward_weight=rw,
                        maps=args.maps, opponents=opponents, policies=policy_names,
                        gzip=args.gzip, chunk_rows=args.chunk_rows,
                        config=vars(args), git_sha=_git_sha(),
                        store_terminal_obs=has_term,
                        store_full_state=args.full_state,
                        state_shape=(base_env._grid_cells, 16),
                        store_counterfactual=args.counterfactual_frac > 0,
                    )
                elif cur != shapes:
                    raise SystemExit(
                        f"map '{map_path}' has shapes {cur} != first map {shapes}; "
                        "collect maps of differing grid size in separate runs (--name)")

                policy = load_policy(block.policy, base_env.obs_shape,
                                     base_env.action_nvec, device=args.policy_device)
                if block.eps > 0:
                    policy = EpsilonGreedyPolicy(policy, base_env.action_nvec,
                                                 block.eps, device=args.policy_device)

                if block.mode == "selfplay":
                    lane_opp = np.full(n, self_opp, dtype=np.int32)
                    lane_pol = np.full(n, legend_id(block.policy_name()), np.int32)
                else:
                    lane_opp = np.array([i % len(args.bots) for i in range(n)],
                                        dtype=np.int32)
                    base_id = legend_id(block.policy_name())
                    if block.seats == "mix":
                        seat_id = legend_id(block.policy_name() + "#p1seat")
                        lane_pol = np.array(
                            [seat_id if i % 2 else base_id for i in range(n)],
                            dtype=np.int32)
                    else:
                        lane_pol = np.full(n, base_id, np.int32)

                coll = OfflineCollector(env, policy, writer,
                                        device=args.policy_device,
                                        steps_per_segment=args.steps_per_segment,
                                        selfplay_pairs=block.mode == "selfplay",
                                        counterfactual_frac=args.counterfactual_frac)
                coll.collect(block.steps, map_id=map_id, opponent_id=lane_opp,
                             policy_id=lane_pol, action_noise=block.eps)
                total_transitions += block.steps * n
                dt = time.time() - t0
                print(f"[collect] map {map_id} block '{block.describe()}' done — "
                      f"{total_transitions} transitions so far "
                      f"({total_transitions / max(dt, 1e-9):.0f} sps)", flush=True)
    finally:
        if writer is not None:
            writer.close()

    dt = time.time() - t0
    print(f"[collect] done: {total_transitions} transitions in {dt:.1f}s "
          f"({total_transitions / max(dt, 1e-9):.0f} sps) -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
