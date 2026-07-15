"""AtariVecEnv tests — shapes, obs range, is_first collation, stepping."""

import torch

from environments.atari_env import AtariVecEnv, AtariEnvConfig


def _env(n=2, **kw):
    return AtariVecEnv(AtariEnvConfig(game="pong", num_envs=n, max_steps=50, **kw))


def test_reset_shapes_and_obs_range():
    env = _env()
    td = env.reset()
    assert td["obs"].shape == (2, 1, 64, 64)
    assert td["obs"].dtype == torch.float32
    assert 0.0 <= float(td["obs"].min()) and float(td["obs"].max()) <= 1.0
    assert bool(td["is_first"].all())          # reset frames are episode starts
    assert env.num_actions == 6                # Pong minimal action set
    env.close()


def test_step_shapes_and_action_range_accepted():
    env = _env()
    env.reset()
    td = env.step(torch.randint(0, env.num_actions, (2,)))
    assert td["obs"].shape == (2, 1, 64, 64)
    assert td["reward"].shape == (2,) and td["done"].shape == (2,)
    assert td["raw_reward"].shape == (2,) and td["shaping_reward"].shape == (2,)
    assert td["is_first"].shape == (2,)
    env.close()


def test_is_first_tracks_done_over_episode_cap():
    # max_steps=5 forces a truncation-done quickly; is_first must mirror prior done.
    env = _env(n=2)
    env.cfg.max_steps = 5
    env.reset()
    seen_done_then_first = False
    prev_done = torch.zeros(2, dtype=torch.bool)
    for _ in range(12):
        td = env.step(torch.zeros(2, dtype=torch.long))
        # is_first of this transition equals this transition's own done (auto-reset).
        assert torch.equal(td["is_first"], td["done"])
        if prev_done.any():
            seen_done_then_first = True
        prev_done = td["done"]
    assert seen_done_then_first
    env.close()


def test_grayscale_false_gives_three_channels():
    env = _env(grayscale=False)
    td = env.reset()
    assert td["obs"].shape == (2, 3, 64, 64)
    env.close()


def test_frame_stack_repeats_on_reset_and_rolls_on_step():
    env = _env(frame_stack=4)
    first = env.reset()["obs"]
    assert first.shape == (2, 4, 64, 64)
    assert torch.allclose(first[:, 0], first[:, 1])
    td = env.step(torch.randint(0, env.num_actions, (2,)))
    assert td["obs"].shape == (2, 4, 64, 64)
    assert torch.allclose(td["obs"][:, 0], first[:, 1])
    env.close()


def test_pong_potential_prefers_paddle_ball_alignment():
    aligned = torch.zeros(210, 160, dtype=torch.uint8).numpy()
    aligned[100:121, 140:144] = 255
    aligned[109:113, 118:122] = 255
    misaligned = aligned.copy()
    misaligned[109:113, 118:122] = 0
    misaligned[50:54, 118:122] = 255
    assert AtariVecEnv._pong_potential_from_frame(aligned) > AtariVecEnv._pong_potential_from_frame(misaligned)


def test_dense_reward_surfaces_bounded_shaping_and_raw_score():
    env = _env(dense_reward="pong_potential", dense_reward_coef=0.05, dense_reward_clip=0.05)
    try:
        env.reset()
        td = env.step(torch.randint(0, env.num_actions, (2,)))
        assert "raw_reward" in td.keys() and "shaping_reward" in td.keys()
        assert float(td["shaping_reward"].abs().max()) <= 0.050001
        assert torch.allclose(td["reward"], td["raw_reward"] + td["shaping_reward"])
    finally:
        env.close()
