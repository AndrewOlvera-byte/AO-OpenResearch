"""``AtariVecEnv`` — in-process vectorized Atari (ALE) env for DreamerV4.

Steps ``N`` independent ``ale_py.ALEInterface`` instances behind the same async
``VecEnv`` API the collectors use (``async_reset`` / ``send`` / ``recv`` + sync
``reset`` / ``step`` sugar). Preprocessing follows the standard Dreamer/Atari recipe:

- **frameskip** (default 4) with **max-pool over the last two raw frames** (kills the
  2-frame sprite flicker),
- grayscale ``210x160`` screens **resized to 64x64** and scaled to ``[0, 1]``,
- optional **frame stacking**. A true recurrent Dreamer actor can use a single frame;
  this Atari wrapper defaults by config to stacks for the current feed-forward actor,
- random **no-op resets** (up to ``noop_max``) to decorrelate episode starts,
- ``sticky actions`` off by default (``repeat_action_probability = 0``).

Optional reward shaping is deliberately kept outside the DreamerV4 dynamics: for
Pong we can add a small potential-based alignment reward computed from the raw ALE
screen, while still surfacing ``raw_reward`` for evaluation and score reporting.

Auto-reset is internal: a ``done`` env is reset and the returned obs is already the
next episode's first frame, so ``is_first`` (needed by the world model) equals the
previous step's ``done`` — surfaced directly on every transition.

Rollout is serial (one core): ALE releases little of the GIL, so a thread pool does
not help. At frameskip 4 this sustains ~6k agent-steps/s for 16 envs, which is far
above Dreamer's data appetite (Atari100k = 400k frames total) — rollout is not the
bottleneck; the model update is. Scaling past one core would need a subprocess pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from ale_py import ALEInterface, LoggerMode, roms

# Quiet ALE's per-ROM banner (it prints on every loadROM otherwise).
try:
    ALEInterface.setLoggerMode(LoggerMode.Error)
except Exception:
    pass

RAW_H, RAW_W = 210, 160


@dataclass
class AtariEnvConfig:
    game: str = "pong"
    num_envs: int = 16
    frameskip: int = 4
    resize: int = 64
    grayscale: bool = True
    frame_stack: int = 1
    max_steps: int = 27000          # agent steps (~108k frames) per episode cap
    noop_max: int = 30
    repeat_action_probability: float = 0.0
    full_action_space: bool = False
    seed: int = 0
    clip_reward: bool = False
    reward_scale: float = 1.0
    dense_reward: str = "none"
    dense_reward_coef: float = 0.0
    dense_reward_gamma: float = 0.997
    dense_reward_clip: float = 0.05


class AtariVecEnv:
    def __init__(self, cfg: AtariEnvConfig):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.frameskip = cfg.frameskip
        self.resize = cfg.resize
        self.grayscale = cfg.grayscale
        rom = roms.get_rom_path(cfg.game)

        self._ales: list[ALEInterface] = []
        for i in range(self.num_envs):
            ale = ALEInterface()
            ale.setInt("random_seed", cfg.seed + i)
            ale.setFloat("repeat_action_probability", cfg.repeat_action_probability)
            ale.loadROM(rom)
            self._ales.append(ale)
        a0 = self._ales[0]
        self.action_set = (a0.getLegalActionSet() if cfg.full_action_space
                           else a0.getMinimalActionSet())
        self.num_actions = len(self.action_set)
        # Uniform-with-MicroRTS handle so model builders can read a MultiDiscrete-like
        # size; Atari is a single Discrete(num_actions).
        self.action_nvec = torch.tensor([self.num_actions], dtype=torch.long)

        c = 1 if cfg.grayscale else 3
        self.frame_channels = c
        self.frame_stack = max(1, int(cfg.frame_stack))
        self.obs_shape = (c * self.frame_stack, cfg.resize, cfg.resize)
        self._rng = np.random.default_rng(cfg.seed)
        self._ep_steps = np.zeros(self.num_envs, dtype=np.int64)
        # Scratch buffers for max-pool over the last two frames.
        self._frame_buf = np.zeros((self.num_envs, 2, RAW_H, RAW_W), dtype=np.uint8)
        self._pong_phi = np.zeros(self.num_envs, dtype=np.float32)
        self._obs_stack: torch.Tensor | None = None
        self._pending: TensorDict | None = None

    # --- raw screen helpers ----------------------------------------------
    def _grab(self, i: int, slot: int) -> None:
        self._frame_buf[i, slot] = self._ales[i].getScreenGrayscale()

    def _noop_reset(self, i: int) -> None:
        ale = self._ales[i]
        ale.reset_game()
        for _ in range(int(self._rng.integers(0, self.cfg.noop_max + 1))):
            ale.act(int(self.action_set[0]))  # NOOP
            if ale.game_over():
                ale.reset_game()
        self._ep_steps[i] = 0

    def _frame_from_buf(self) -> torch.Tensor:
        """Max-pool last two raw frames -> resize -> [0,1] float (N,C,resize,resize)."""
        maxed = self._max_frame()
        x = torch.from_numpy(maxed).unsqueeze(1).float()                    # (N,1,210,160)
        x = F.interpolate(x, size=(self.resize, self.resize), mode="area")
        x = x / 255.0
        if not self.grayscale:
            x = x.repeat(1, 3, 1, 1)
        return x.contiguous()

    def _max_frame(self) -> np.ndarray:
        return np.maximum(self._frame_buf[:, 0], self._frame_buf[:, 1])

    @staticmethod
    def _pong_potential_from_frame(frame: np.ndarray) -> float:
        """State potential for Pong: right paddle aligned with the ball.

        The detector intentionally uses broad image regions rather than ROM-specific
        RAM addresses. It ignores score pixels, side paddles, and the center divider;
        if the ball is temporarily invisible, the neutral potential is zero.
        """
        mask = frame > 100
        player = mask[34:194, 132:152]
        py, _ = np.nonzero(player)
        if py.size == 0:
            return 0.0
        paddle_y = float(py.mean() + 34)

        play = mask[34:194, 20:132].copy()
        play[:, 56:66] = False     # center dashed divider in raw-frame coordinates
        by, bx = np.nonzero(play)
        if by.size == 0:
            return 0.0
        ball_x = float(bx.mean() + 20)
        ball_y = float(by.mean() + 34)
        alignment = 1.0 - min(abs(paddle_y - ball_y) / 80.0, 1.0)
        proximity = min(max((ball_x - 72.0) / 60.0, 0.0), 1.0)
        return float(alignment * proximity)

    def _pong_potential(self) -> np.ndarray:
        frames = self._max_frame()
        return np.asarray([self._pong_potential_from_frame(frames[i])
                           for i in range(self.num_envs)], dtype=np.float32)

    def _shape_reward(self, raw_reward: np.ndarray, done: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        task_reward = np.sign(raw_reward) if self.cfg.clip_reward else raw_reward.astype(np.float32)
        shaping = np.zeros_like(task_reward, dtype=np.float32)
        dense = str(self.cfg.dense_reward or "none").lower()
        if dense in ("pong_potential", "pong", "potential") and self.cfg.dense_reward_coef:
            phi_next = self._pong_potential()
            terminal_phi_next = np.where(done, 0.0, phi_next)
            shaping = (float(self.cfg.dense_reward_gamma) * terminal_phi_next - self._pong_phi)
            shaping *= float(self.cfg.dense_reward_coef)
            clip = float(self.cfg.dense_reward_clip or 0.0)
            if clip > 0:
                shaping = np.clip(shaping, -clip, clip)
            self._pong_phi = phi_next
        reward = (task_reward + shaping) * float(self.cfg.reward_scale)
        return reward.astype(np.float32), shaping.astype(np.float32)

    def _repeat_stack(self, frame: torch.Tensor) -> torch.Tensor:
        return frame.repeat(1, self.frame_stack, 1, 1).contiguous()

    def _reset_obs_stack(self, frame: torch.Tensor) -> torch.Tensor:
        self._obs_stack = self._repeat_stack(frame)
        return self._obs_stack

    def _advance_obs_stack(self, frame: torch.Tensor, done: np.ndarray) -> torch.Tensor:
        if self.frame_stack == 1:
            self._obs_stack = frame
            return frame
        if self._obs_stack is None:
            return self._reset_obs_stack(frame)
        c = self.frame_channels
        self._obs_stack = torch.cat([self._obs_stack[:, c:], frame], dim=1).contiguous()
        done_t = torch.as_tensor(done, dtype=torch.bool)
        if bool(done_t.any()):
            self._obs_stack[done_t] = self._repeat_stack(frame[done_t])
        return self._obs_stack

    def _pack(self, obs, reward, done, is_first, raw_reward=None, shaping_reward=None) -> TensorDict:
        n = self.num_envs
        if raw_reward is None:
            raw_reward = reward
        if shaping_reward is None:
            shaping_reward = np.zeros(n, dtype=np.float32)
        return TensorDict(
            {
                "obs": obs,
                "reward": torch.as_tensor(reward, dtype=torch.float32),
                "raw_reward": torch.as_tensor(raw_reward, dtype=torch.float32),
                "shaping_reward": torch.as_tensor(shaping_reward, dtype=torch.float32),
                "done": torch.as_tensor(done, dtype=torch.bool),
                "trunc": torch.as_tensor(done, dtype=torch.bool),
                "is_first": torch.as_tensor(is_first, dtype=torch.bool),
                "env_id": torch.arange(n),
            },
            batch_size=[n],
        )

    # --- async API -------------------------------------------------------
    def async_reset(self, seed: int | None = None) -> None:
        for i in range(self.num_envs):
            self._noop_reset(i)
            self._grab(i, 0)
            self._grab(i, 1)
        self._pong_phi = self._pong_potential()
        obs = self._reset_obs_stack(self._frame_from_buf())
        self._pending = self._pack(obs, np.zeros(self.num_envs), np.zeros(self.num_envs, bool),
                                   np.ones(self.num_envs, bool))

    def send(self, actions: torch.Tensor) -> None:
        a = actions.detach().to("cpu").view(-1).long().numpy()
        raw_rewards = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=bool)
        cap = self.frameskip
        for i, ale in enumerate(self._ales):
            act = int(self.action_set[int(a[i])])
            for k in range(cap):
                if ale.game_over():
                    break
                raw_rewards[i] += ale.act(act)
                if k == cap - 2:
                    self._grab(i, 0)
                elif k == cap - 1:
                    self._grab(i, 1)
            self._ep_steps[i] += 1
            done[i] = ale.game_over() or self._ep_steps[i] >= self.cfg.max_steps
            if done[i]:
                self._noop_reset(i)          # auto-reset -> returned obs is next episode start
                self._grab(i, 0)
                self._grab(i, 1)
        rewards, shaping = self._shape_reward(raw_rewards, done)
        obs = self._advance_obs_stack(self._frame_from_buf(), done)
        # is_first for the NEXT transition equals this step's done (auto-reset).
        self._pending = self._pack(obs, rewards, done, done, raw_rewards, shaping)

    def recv(self) -> TensorDict:
        assert self._pending is not None, "call async_reset()/send() before recv()"
        out, self._pending = self._pending, None
        return out

    def reset(self, seed: int | None = None) -> TensorDict:
        self.async_reset(seed)
        return self.recv()

    def step(self, actions: torch.Tensor) -> TensorDict:
        self.send(actions)
        return self.recv()

    def close(self) -> None:
        self._ales.clear()
