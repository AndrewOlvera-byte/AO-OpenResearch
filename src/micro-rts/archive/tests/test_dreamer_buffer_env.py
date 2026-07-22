"""Sequence buffer / DreamEnv collation / DreamCollector / WorldModelMemory tests
(no JVM).

A tiny in-memory ``FakeVecEnv`` stands in for MicroRTS so the data-plane logic
(ring-buffer sampling, ``is_first`` tagging, collector fills, rolling world-model
memory) is tested without the Java engine. ``is_first`` is the crucial collation
difference the world model needs.
"""

import torch
from tensordict import TensorDict

from collectors.sequence_buffer import SequenceReplayBuffer
from environments.dream_env import DreamEnv
from collectors.dream_collector import DreamCollector
from models.dreamer.memory import WorldModelMemory

OBS = (6, 8, 8)
NVEC = [64, 6, 4, 4, 4, 4, 7, 49]
MASK_W = 1 + sum(NVEC[1:])


class FakeVecEnv:
    """Minimal VecEnv: obs=step counter broadcast, done every ``period`` steps."""

    def __init__(self, num_envs=3, period=4):
        self.num_envs = num_envs
        self.obs_shape = OBS
        self.action_nvec = torch.as_tensor(NVEC)
        self.mask_shape = (NVEC[0], MASK_W)
        self.gridnet = True
        self._t = 0
        self._pending = None

    def _pack(self, done):
        n = self.num_envs
        return TensorDict({
            "obs": torch.full((n, *OBS), float(self._t)),
            "mask": torch.ones(n, NVEC[0], MASK_W),
            "reward": torch.ones(n),
            "done": torch.as_tensor(done),
        }, batch_size=[n])

    def async_reset(self, seed=None):
        self._t = 0
        self._pending = self._pack([False] * self.num_envs)

    def send(self, actions):
        self._t += 1
        done = [(self._t % 4 == 0)] * self.num_envs
        self._pending = self._pack(done)

    def recv(self):
        out, self._pending = self._pending, None
        return out

    def reset(self):
        self.async_reset()
        return self.recv()

    def step(self, actions):
        self.send(actions)
        return self.recv()

    def close(self):
        pass


class FakePolicy:
    """Returns the Policy-protocol keys; ``with_z`` adds latents like DreamerV4."""

    def __init__(self, with_z=False):
        self.with_z = with_z

    def step(self, obs, mask=None, deterministic=False):
        n = obs.shape[0]
        out = {
            "action": torch.zeros(n, NVEC[0], 7, dtype=torch.long),
            "logprob": torch.zeros(n),
            "value": torch.zeros(n),
        }
        if self.with_z:
            out["z"] = torch.rand(n, 4, 8)
        return TensorDict(out, batch_size=[n])


# --- sequence buffer -----------------------------------------------------
def test_sequence_buffer_sampling_shapes_and_contiguity():
    buf = SequenceReplayBuffer(capacity=10, num_envs=3, obs_shape=OBS,
                               action_shape=(NVEC[0], 7), mask_shape=(NVEC[0], MASK_W))
    for t in range(8):
        buf.add(obs=torch.full((3, *OBS), float(t)),
                action=torch.zeros(3, NVEC[0], 7, dtype=torch.long),
                mask=torch.ones(3, NVEC[0], MASK_W),
                reward=torch.ones(3), cont=torch.ones(3),
                is_first=torch.zeros(3, dtype=torch.bool))
    assert buf.size == 8
    batch = buf.sample(batch=5, seq_len=4)
    assert batch["obs"].shape == (5, 4, *OBS)
    assert batch["is_first"].shape == (5, 4)
    assert batch["opponent_action"].shape == (5, 4, NVEC[0], 7)
    assert not batch["opponent_valid"].any()  # omitted stream is explicitly unknown
    # obs stores the timestep index -> each sampled sequence must be consecutive ints.
    per_step = batch["obs"].reshape(5, 4, -1)[..., 0]
    diffs = per_step[:, 1:] - per_step[:, :-1]
    assert torch.all(diffs == 1)


def test_sequence_buffer_ring_overwrites_oldest():
    buf = SequenceReplayBuffer(capacity=4, num_envs=1, obs_shape=OBS,
                               action_shape=(NVEC[0], 7), mask_shape=(NVEC[0], MASK_W))
    for t in range(6):  # capacity 4, so 0,1 are overwritten
        buf.add(obs=torch.full((1, *OBS), float(t)),
                action=torch.zeros(1, NVEC[0], 7, dtype=torch.long),
                mask=torch.ones(1, NVEC[0], MASK_W),
                reward=torch.ones(1), cont=torch.ones(1),
                is_first=torch.zeros(1, dtype=torch.bool))
    assert buf.size == 4
    batch = buf.sample(batch=8, seq_len=2)
    vals = batch["obs"].reshape(8, 2, -1)[..., 0]
    assert vals.min() >= 2  # steps 0,1 are gone


def test_can_sample_gate():
    buf = SequenceReplayBuffer(capacity=8, num_envs=1, obs_shape=OBS,
                               action_shape=(NVEC[0], 7), mask_shape=(NVEC[0], MASK_W))
    assert not buf.can_sample(4)
    for _ in range(4):
        buf.add(obs=torch.zeros(1, *OBS), action=torch.zeros(1, NVEC[0], 7, dtype=torch.long),
                mask=torch.ones(1, NVEC[0], MASK_W), reward=torch.ones(1),
                cont=torch.ones(1), is_first=torch.zeros(1, dtype=torch.bool))
    assert buf.can_sample(4) and not buf.can_sample(5)


# --- dream env collation -------------------------------------------------
def test_dream_env_is_first_flags():
    env = DreamEnv(FakeVecEnv(num_envs=2, period=4))
    first = env.reset()
    assert bool(first["is_first"].all())        # reset obs are episode starts
    flags = []
    for _ in range(8):
        td = env.step(torch.zeros(2, NVEC[0], 7, dtype=torch.long))
        flags.append((bool(td["done"].any()), bool(td["is_first"].any())))
    # is_first mirrors the previous step's done (engine auto-reset). With period 4,
    # done fires at steps 4 and 8 (1-indexed), so is_first tracks done exactly here.
    for done, is_first in flags:
        assert done == is_first


def test_dream_env_delegates_attributes():
    env = DreamEnv(FakeVecEnv())
    assert env.num_envs == 3
    assert env.obs_shape == OBS
    assert env.gridnet is True


def test_sequence_buffer_uint8_storage_float_samples():
    """The ring keeps one-hot obs as uint8 (4x smaller, off-GPU by default);
    sample() must hand the models float32 back."""
    buf = SequenceReplayBuffer(capacity=8, num_envs=2, obs_shape=OBS,
                               action_shape=(NVEC[0], 7), mask_shape=(NVEC[0], MASK_W))
    for t in range(6):
        buf.add(obs=torch.full((2, *OBS), float(t)),
                action=torch.zeros(2, NVEC[0], 7, dtype=torch.long),
                mask=torch.ones(2, NVEC[0], MASK_W),
                reward=torch.ones(2), cont=torch.ones(2),
                is_first=torch.zeros(2, dtype=torch.bool))
    assert buf.data["obs"].dtype == torch.uint8
    assert str(buf.data.device) == "cpu"
    batch = buf.sample(batch=3, seq_len=6)      # only one valid window: rows 0..5
    assert batch["obs"].dtype == torch.float32
    assert batch["obs"].max() == 5.0            # values survive the uint8 round-trip


# --- collector -----------------------------------------------------------
def test_dream_collector_fills_buffer():
    env = DreamEnv(FakeVecEnv(num_envs=3))
    coll = DreamCollector(env, FakePolicy(), horizon=5, capacity=32)
    buf = coll.collect()
    assert buf.size == 5
    coll.collect()
    assert buf.size == 10
    batch = buf.sample(batch=4, seq_len=4)
    assert batch["mask"].shape == (4, 4, NVEC[0], MASK_W)
    assert batch["cont"].dtype == torch.float32


class FakeTerminalVecEnv(FakeVecEnv):
    """FakeVecEnv that also surfaces the patched jar's ``terminal_obs``: at done
    steps the true pre-reset arrival frame (encoded here as ``100 + t``)."""

    def _pack(self, done):
        td = super()._pack(done)
        term = torch.zeros(self.num_envs, *OBS)
        for i, d in enumerate(done):
            if d:
                term[i] = 100.0 + self._t
        td.set("terminal_obs", term)
        return td


def test_dream_collector_terminal_splice():
    """With terminal_splice, the reset slot stores the terminal arrival frame
    (mask zeroed, is_first pushed one slot later) — mirroring the offline
    MRTSSequenceDataset splice so cont=0 / win-reward rows exist online."""
    env = DreamEnv(FakeTerminalVecEnv(num_envs=2, period=4))
    coll = DreamCollector(env, FakePolicy(), horizon=10, capacity=32,
                          terminal_splice=True)
    buf = coll.collect()
    obs = buf.data["obs"][:10, 0].reshape(10, -1)[:, 0].float()   # env 0 timeline
    mask = buf.data["mask"][:10, 0]
    first = buf.data["is_first"][:10, 0]
    cont = buf.data["cont"][:10, 0]
    # done fires at env steps 4 and 8 -> stored rows 4 and 8 are the reset slots,
    # spliced with terminal obs (>= 100), mask all-zero, is_first False; rows 5
    # and 9 start the new episode instead.
    for t in (4, 8):
        assert obs[t] >= 100.0, f"row {t} not spliced: {obs[t]}"
        assert not mask[t].any()
        assert not first[t]
        assert first[t + 1]
        assert cont[t - 1] == 0.0        # arrive-aligned cont=0 target one row back
    # non-splice rows are untouched real frames
    assert obs[2] < 100.0 and mask[2].all()


def test_dream_collector_splice_off_keeps_reset_semantics():
    env = DreamEnv(FakeTerminalVecEnv(num_envs=2, period=4))
    coll = DreamCollector(env, FakePolicy(), horizon=10, capacity=32,
                          terminal_splice=False)
    buf = coll.collect()
    obs = buf.data["obs"][:10, 0].reshape(10, -1)[:, 0].float()
    first = buf.data["is_first"][:10, 0]
    assert obs.max() < 100.0             # no terminal frames stored
    assert first[4] and first[8]         # is_first stays on the reset slots


class FakeRawRewardVecEnv(FakeVecEnv):
    """FakeVecEnv with raw_rewards: win component +1 at every done step."""

    def _pack(self, done):
        td = super()._pack(done)
        raw = torch.zeros(self.num_envs, 6)
        raw[:, 0] = torch.as_tensor(done).float()      # win at each done
        td.set("raw_rewards", raw)
        return td


class FakeOpponentActionVecEnv(FakeVecEnv):
    def _pack(self, done):
        td = super()._pack(done)
        td.set("opponent_action", torch.full(
            (self.num_envs, NVEC[0], 7), self._t, dtype=torch.long))
        return td


def test_dream_collector_stores_exact_bot_opponent_action():
    env = DreamEnv(FakeOpponentActionVecEnv(num_envs=2))
    coll = DreamCollector(env, FakePolicy(), horizon=5, capacity=16)
    buf = coll.collect()
    assert buf.data["opponent_valid"][:5].all()
    # Opponent action is surfaced by the result of each same-tick env step.
    got = buf.data["opponent_action"][:5, 0, 0, 0]
    assert torch.equal(got, torch.arange(1, 6))


def test_dream_collector_episode_stats():
    """The collector tracks finished-episode returns and win/loss/timeout counts
    (the RL observability fix: the failed hybrid run was blind between evals)."""
    env = DreamEnv(FakeRawRewardVecEnv(num_envs=2, period=4))
    coll = DreamCollector(env, FakePolicy(), horizon=10, capacity=32)
    assert coll.pop_episode_stats() == {}               # nothing finished yet
    coll.collect()
    stats = coll.pop_episode_stats()
    # 10 steps, done at t=4 and t=8 -> 2 episodes per lane finished.
    assert stats["collect/episodes"] == 4.0
    assert stats["collect/ep_return"] == 4.0            # reward 1/step, 4 steps
    assert stats["collect/wins"] == 4.0 and stats["collect/win_rate"] == 1.0
    assert stats["collect/losses"] == 0.0 and stats["collect/timeouts"] == 0.0
    assert coll.pop_episode_stats() == {}               # drained


class FakeStatefulPolicy(FakePolicy):
    """FakePolicy with the state_dict surface the league snapshot pool needs."""

    def __init__(self, with_z=False, tag=0.0):
        super().__init__(with_z)
        self._sd = {"tokenizer.w": torch.tensor([tag]),
                    "world_model.w": torch.tensor([99.0]),
                    "action_expert.w": torch.tensor([tag])}
        self.loaded = None

    def state_dict(self):
        return dict(self._sd)

    def load_state_dict(self, sd, strict=True):
        self.loaded = sd
        return [], []


def test_dream_league_collector_phases_and_switch():
    """Bot warmup -> self-play block: the buffer keeps N learner lanes, the first
    row after the env switch is force-tagged is_first, snapshots exclude the world
    model, and the opponent is loaded from the pool."""
    from collectors.dream_collector import DreamLeagueCollector

    bot_env = DreamEnv(FakeVecEnv(num_envs=2))
    sp_env = DreamEnv(FakeVecEnv(num_envs=4))       # 2 games -> 2 learner lanes
    learner, opponent = FakeStatefulPolicy(tag=1.0), FakeStatefulPolicy(tag=0.0)
    coll = DreamLeagueCollector(
        bot_env, sp_env, learner, opponent, horizon=4, capacity=32,
        bot_steps=8,                    # exactly one collect (4 steps x 2 lanes)
        mix_bot_block=0, mix_selfplay_block=8,   # then self-play forever
        snapshot_every=8, pool_capacity=4)

    assert coll.phase() == "bot"
    buf = coll.collect()                # bot block
    assert buf.N == 2 and buf.size == 4
    assert coll.pool_size == 1          # snapshot pushed at 8 collected steps
    assert all(not k.startswith("world_model.")
               for k in coll.pool.sample())

    assert coll.phase() == "selfplay"
    coll.collect()                      # self-play block
    assert coll.opponent.loaded is not None
    assert bool(buf.data["is_first"][4].all())   # switch row starts fresh contexts
    assert buf.size == 8


def test_dream_league_bot_only_when_selfplay_block_zero():
    from collectors.dream_collector import DreamLeagueCollector

    coll = DreamLeagueCollector(
        DreamEnv(FakeVecEnv(num_envs=2)), DreamEnv(FakeVecEnv(num_envs=4)),
        FakeStatefulPolicy(), FakeStatefulPolicy(), horizon=2, capacity=16,
        bot_steps=0, mix_bot_block=0, mix_selfplay_block=0)
    assert coll.phase(0) == "bot" and coll.phase(10 ** 9) == "bot"


# --- world-model memory ----------------------------------------------------
def test_memory_rolls_a_bounded_window():
    mem = WorldModelMemory(context_len=3)
    assert not mem.ready
    for t in range(5):
        mem.append(z=torch.full((2, 4, 8), float(t)),
                   action=torch.full((2, NVEC[0], 7), t, dtype=torch.long),
                   is_first=torch.tensor([t == 0, False]))
    assert len(mem) == 3                          # only the last 3 frames kept
    ctx = mem.context()
    assert ctx["z"].shape == (2, 3, 4, 8)
    assert ctx["action"].shape == (2, 3, NVEC[0], 7)
    assert ctx["is_first"].shape == (2, 3)
    assert ctx["z"][0, :, 0, 0].tolist() == [2.0, 3.0, 4.0]  # frames 2,3,4
    mem.reset()
    assert not mem.ready


def test_memory_preserves_is_first_flags():
    mem = WorldModelMemory(context_len=4)
    firsts = [torch.tensor([True]), torch.tensor([False]), torch.tensor([True])]
    for i, f in enumerate(firsts):
        mem.append(z=torch.zeros(1, 4, 8), action=torch.zeros(1, NVEC[0], 7).long(),
                   is_first=f)
    ctx = mem.context()
    assert ctx["is_first"][0].tolist() == [True, False, True]


def test_collector_feeds_memory_when_policy_returns_latents():
    env = DreamEnv(FakeVecEnv(num_envs=3))
    mem = WorldModelMemory(context_len=4)
    coll = DreamCollector(env, FakePolicy(with_z=True), horizon=6, capacity=32, memory=mem)
    coll.collect()
    assert len(mem) == 4                          # bounded by context_len
    ctx = mem.context()
    assert ctx["z"].shape == (3, 4, 4, 8)


def test_collector_without_latents_leaves_memory_empty():
    env = DreamEnv(FakeVecEnv(num_envs=3))
    mem = WorldModelMemory(context_len=4)
    coll = DreamCollector(env, FakePolicy(with_z=False), horizon=3, capacity=32, memory=mem)
    coll.collect()
    assert not mem.ready
