from __future__ import annotations

import torch

from models.incomplete_info.world_action import (
    FactorizedDynamicsConfig,
    FactorizedWorldActionDynamics,
    PredictiveBeliefConfig,
    WorldActionBeliefEncoder,
)
from models.incomplete_info.world_action.encoder import split_branches
from models.incomplete_info.world_action.losses import semantic_transition_target


def tiny_belief_cfg():
    return PredictiveBeliefConfig(
        d_model=64,
        d_latent=32,
        n_heads=4,
        frame_depth=1,
        temporal_depth=1,
        mixer_depth=1,
        predictor_depth=1,
        n_self_tokens=2,
        n_opponent_tokens=2,
        n_static_tokens=1,
        n_interaction_tokens=2,
        context_length=8,
        horizons=(1, 2),
    )


def test_factorized_encoder_shapes_and_causal_history():
    torch.manual_seed(0)
    cfg = tiny_belief_cfg()
    model = WorldActionBeliefEncoder(24, 16, cfg).eval()
    spatial = torch.randn(2, 5, 4, 24)
    visibility = torch.ones(2, 5, 4, 1)
    action = torch.randn(2, 5, 3, 16)
    valid = torch.ones(2, 5, 3, dtype=torch.bool)
    changed = action.clone()
    changed[:, -1].add_(100.0)

    first = model(spatial, visibility, action, valid)
    second = model(spatial, visibility, changed, valid)
    assert first["tokens"].shape == (2, 5, cfg.n_tokens, cfg.d_latent)
    assert first["self"].shape[-2] == cfg.n_self_tokens
    assert first["opponent"].shape[-2] == cfg.n_opponent_tokens
    # Causal temporal SDPA prevents the final action from changing prior rows.
    torch.testing.assert_close(first["tokens"][:, :-1], second["tokens"][:, :-1])


def test_opponent_flow_has_no_simultaneous_ego_action_path():
    torch.manual_seed(1)
    cfg = FactorizedDynamicsConfig(
        d_model=64,
        d_latent=32,
        n_heads=4,
        depth=2,
        n_self_tokens=2,
        n_opponent_tokens=2,
        n_static_tokens=1,
        n_interaction_tokens=2,
        n_opponent_plan_tokens=2,
    )
    model = FactorizedWorldActionDynamics(16, 20, cfg).eval()
    departure = torch.randn(3, sum(cfg.branch_sizes), cfg.d_latent)
    noisy = torch.randn_like(departure)
    plan = torch.randn(3, cfg.n_opponent_plan_tokens, 20)
    action_a = torch.randn(3, 4, 16)
    action_b = torch.randn(3, 4, 16)
    tau = torch.full((3,), 0.5)
    velocity_a = split_branches(
        model.velocity(noisy, tau, departure, action_a, plan), cfg.branch_sizes
    )
    velocity_b = split_branches(
        model.velocity(noisy, tau, departure, action_b, plan), cfg.branch_sizes
    )
    torch.testing.assert_close(velocity_a["opponent"], velocity_b["opponent"])
    assert velocity_a["static"].count_nonzero() == 0


def test_semantic_transition_target_is_change_only():
    state = torch.zeros(1, 2, 256, 16, dtype=torch.long)
    globals_ = torch.zeros(1, 2, 8, dtype=torch.long)
    unchanged = semantic_transition_target(
        state, state.clone(), globals_, globals_.clone(), downsample=2
    )
    assert unchanged.shape == (1, 2, 64, 8)
    assert unchanged.count_nonzero() == 0

    changed_state = state.clone()
    changed_state[..., 0, 1] = 1
    changed_state[..., 0, 3] = 1
    changed = semantic_transition_target(
        state, changed_state, globals_, globals_.clone(), downsample=2
    )
    assert changed[..., 0, :6].abs().sum() > 0
    # Other map patches do not receive a dense reconstruction target.
    assert changed[..., 1:, :6].count_nonzero() == 0
