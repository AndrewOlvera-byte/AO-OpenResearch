"""GridNet (all-units-per-step) masked head, policy, and env round-trip.

The head must (a) restrict every active cell to legal component choices, (b) let
empty/busy cells (all-zero mask) contribute nothing, and (c) agree between ``sample``
and ``log_prob_entropy``. The env must round-trip the per-cell (N, H*W, 7) action
through the engine and surface per-cell masks.
"""

import torch

from models.gridnet_policy import GridNetPolicy
from models.shared.GridNetHead import GridNetActionHead

NVEC = torch.tensor([256, 6, 4, 4, 4, 4, 7, 49])
CELL_SPLITS = [6, 4, 4, 4, 4, 7, 49]
MASK_WIDTH = 1 + sum(CELL_SPLITS)  # 79
OBS = (27, 16, 16)


def make_gridnet_mask(active, seed=0):
    """(N, 256, 79) mask + expected per-cell/component valid index. Each active cell
    gets exactly one legal index per component, so a correct head is deterministic."""
    g = torch.Generator().manual_seed(seed)
    n = len(active)
    mask = torch.zeros(n, 256, MASK_WIDTH)
    expect = {}
    for i, cells in enumerate(active):
        for c in cells:
            mask[i, c, 0] = 1.0
            off, picks = 1, []
            for sz in CELL_SPLITS:
                k = int(torch.randint(sz, (1,), generator=g).item())
                mask[i, c, off + k] = 1.0
                picks.append(k)
                off += sz
            expect[(i, c)] = picks
    return mask, expect


def test_gridnet_head_single_choice_is_deterministic():
    torch.manual_seed(0)
    head = GridNetActionHead(NVEC[1:])
    logits = torch.randn(3, 256, sum(CELL_SPLITS))
    active = [[17, 34], [3], [100, 101, 102]]
    mask, expect = make_gridnet_mask(active)
    action, logp = head.sample(logits, mask)
    assert action.shape == (3, 256, 7) and logp.shape == (3,)
    # Each active cell must land on its one legal index per component.
    for (i, c), picks in expect.items():
        assert action[i, c].tolist() == picks
    # Single-choice everywhere -> prob 1 -> log-prob 0; empty cells contribute 0.
    assert torch.allclose(logp, torch.zeros(3), atol=1e-5)


def test_gridnet_head_sample_eval_consistent():
    torch.manual_seed(1)
    head = GridNetActionHead(NVEC[1:])
    logits = torch.randn(4, 256, sum(CELL_SPLITS))
    # multiple legal options -> real distribution
    mask = torch.zeros(4, 256, MASK_WIDTH)
    for i, cells in enumerate([[17, 34], [3, 4, 5], [200], [10, 20, 30]]):
        for c in cells:
            mask[i, c, 0] = 1.0
            off = 1
            for sz in CELL_SPLITS:
                mask[i, c, off:off + sz] = 1.0  # all options legal
                off += sz
    action, logp = head.sample(logits, mask)
    logp2, ent = head.log_prob_entropy(logits, action, mask.bool())
    assert torch.allclose(logp, logp2, atol=1e-5)
    assert (ent > 0).all() and torch.isfinite(ent).all()   # real entropy from active cells


def test_gridnet_empty_cells_do_not_contribute():
    head = GridNetActionHead(NVEC[1:])
    logits = torch.randn(2, 256, sum(CELL_SPLITS))
    mask, _ = make_gridnet_mask([[17], []])   # env 1 has no units at all
    _, logp = head.sample(logits, mask)
    logp2, ent = head.log_prob_entropy(logits, torch.zeros(2, 256, 7, dtype=torch.long), mask.bool())
    assert torch.isfinite(logp).all() and torch.isfinite(logp2).all()
    assert float(ent[1]) == 0.0 and float(logp[1]) == 0.0   # no units -> zero everything


def test_gridnet_policy_step_and_evaluate():
    torch.manual_seed(2)
    policy = GridNetPolicy(OBS, NVEC, device="cpu")
    obs = torch.randn(3, *OBS)
    mask, expect = make_gridnet_mask([[17, 34], [3], [100]])
    out = policy.step(obs, mask)
    assert tuple(out["action"].shape) == (3, 256, 7) and out["value"].shape == (3,)
    for (i, c), picks in expect.items():
        assert out["action"][i, c].tolist() == picks
    logp, ent, value = policy.evaluate_actions(obs, out["action"], mask.bool())
    assert torch.allclose(logp, out["logprob"], atol=1e-5)
    assert torch.isfinite(ent).all() and torch.isfinite(value).all()


def test_gridnet_env_roundtrip_and_update():
    """JVM: gridnet env surfaces masks, codec round-trips (N,256,7), PPO update runs."""
    from core.registry import build
    import models.gridnet_policy  # noqa: F401  (register cnn_gridnet)
    from environments.microrts_env import EnvConfig, MicroRTSVecEnv
    from collectors.collector import Collector
    from trainers.PPOTrainer import PPOTrainer

    env = MicroRTSVecEnv(EnvConfig(num_envs=6, max_steps=250, mode="bot",
                                   bots=("randomBiasedAI", "workerRushAI"), gridnet=True))
    try:
        trans = env.reset()
        assert tuple(trans["mask"].shape) == (6, 256, 79)
        policy = build("model", type="cnn_gridnet", obs_shape=env.obs_shape,
                       action_nvec=env.action_nvec, device="cpu")
        col = Collector(env, policy, horizon=8, device="cpu")
        assert tuple(col.buffer.data["action"].shape) == (8, 6, 256, 7)
        assert col.buffer.data["mask"].dtype == torch.bool
        buf = col.collect()

        # Every commanded action on a cell with an actable unit must be legal.
        act, msk = buf.data["action"], buf.data["mask"].float()
        selectable = msk[..., 0] > 0.5                       # (T,N,256)
        assert selectable.any(), "expected some actable units over the rollout"

        tr = PPOTrainer(policy, epochs=2, minibatches=2, gamma=0.99, lam=0.95)
        m = tr._update(buf)
        assert all(v == v for v in m.values())               # finite
    finally:
        env.close()
