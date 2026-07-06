"""Full-game match play + report aggregation, shared by the trainer eval and the
standalone eval entrypoint.

``run_match_eval`` plays on a *provided* env (it does not build or close it) so the
caller owns the env lifecycle — important because closing a MicroRTS env shuts the
shared JVM, which cannot restart. The env must use the win-loss reward preset so a
terminal reward is ``+1`` win / ``-1`` loss / ``0`` draw (timeout).
"""

from __future__ import annotations

import torch

from .elo import DEFAULT_ANCHOR, aggregate_rating, implied_rating


def new_stats(bots) -> dict:
    return {b: {"wins": 0, "losses": 0, "draws": 0, "games": 0, "steps": 0} for b in set(bots)}


@torch.no_grad()
def run_match_eval(policy, env, env_bots, games, max_steps, device, deterministic=True) -> dict:
    """Play until every distinct bot in ``env_bots`` has >= ``games`` finished
    games. ``env_bots[i]`` is the opponent in env lane ``i``."""
    stats = new_stats(env_bots)
    trans = env.reset()
    ep_len = torch.zeros(env.num_envs, dtype=torch.long)
    step_cap = games * max_steps * 4  # safety bound
    steps = 0
    while min(s["games"] for s in stats.values()) < games and steps < step_cap:
        mask = trans.get("mask", None)
        if mask is not None:
            mask = mask.to(device)
        out = policy.step(trans["obs"].to(device), mask, deterministic=deterministic)
        trans = env.step(out["action"])
        ep_len += 1
        steps += 1
        done = trans["done"]
        if done.any():
            reward = trans["reward"]
            for i in torch.nonzero(done, as_tuple=False).flatten().tolist():
                rec = stats[env_bots[i]]
                if rec["games"] >= games:
                    ep_len[i] = 0
                    continue
                r = float(reward[i])
                rec["wins" if r > 0 else "losses" if r < 0 else "draws"] += 1
                rec["games"] += 1
                rec["steps"] += int(ep_len[i])
                ep_len[i] = 0
    return stats


def build_report(stats, anchors=None) -> dict:
    per_bot = {}
    for bot, r in stats.items():
        g = r["games"]
        per_bot[bot] = {
            **r,
            "win_rate": (r["wins"] / g) if g else float("nan"),
            "draw_rate": (r["draws"] / g) if g else float("nan"),
            "avg_len": (r["steps"] / g) if g else float("nan"),
            "elo": implied_rating(r["wins"], r["draws"], g,
                                  (anchors or {}).get(bot, DEFAULT_ANCHOR)),
        }
    total_g = sum(r["games"] for r in stats.values())
    total_w = sum(r["wins"] for r in stats.values())
    total_d = sum(r["draws"] for r in stats.values())
    return {
        "per_bot": per_bot,
        "overall": {
            "games": total_g,
            "win_rate": (total_w / total_g) if total_g else float("nan"),
            "draw_rate": (total_d / total_g) if total_g else float("nan"),
            "elo": aggregate_rating(stats, anchors),
        },
    }
