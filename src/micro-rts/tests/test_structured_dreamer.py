"""Pure-tensor contracts for guarded structured Dreamer RL."""

import torch

from loss.structured_dreamer import (
    sequence_transition_batch,
    structured_agent_head_loss,
    structured_pmpo_losses,
)
from models.dreamer_v2.agent import StructuredDreamer, StructuredDreamerConfig


H = W = 4
CELLS = H * W
NVEC = [CELLS, 6, 4, 4, 4, 4, 7, 49]
MASK_W = 1 + sum(NVEC[1:])


def _state(batch=2, time=4):
    state = torch.zeros(batch, time, CELLS, 16, dtype=torch.long)
    state[..., 2] = -1
    state[..., 3] = -1
    state[..., 4] = -1
    state[..., 8:14] = -1
    for cell, owner in ((5, 1), (10, 2)):
        state[..., cell, 1] = 1
        state[..., cell, 2] = cell
        state[..., cell, 3] = owner
        state[..., cell, 4] = 3
        state[..., cell, 5] = 1
    glob = torch.zeros(batch, time, 8, dtype=torch.long)
    glob[..., 0] = torch.arange(time)
    glob[..., 1:3] = 5
    glob[..., 6] = -1
    return state, glob


def _model():
    cfg = StructuredDreamerConfig.from_dict({
        "tokenizer": {
            "d_cell": 16, "d_latent": 16, "downsample": 2, "depth": 1,
            "n_heads": 4, "max_unit_types": 8, "max_entities": 8,
            "mask_width": MASK_W, "legacy_obs_channels": 6,
        },
        "dynamics": {
            "d_model": 32, "depth": 2, "n_heads": 4, "max_action_events": 8,
            "k_max": 4, "prior_fraction": 0.0, "skip_fraction": 0.0,
            "initial_noise": "zero", "residual_prediction": True,
        },
        "actor_critic": {
            "dec_channels": 16, "summary_hidden": 32, "critic_hidden": [32],
            "reward_bins": 31, "critic_bins": 31, "imagine_horizon": 2,
            "imagine_flow_steps": 1,
        },
        "freeze": {"tokenizer": True, "action_encoder": False,
                   "action_router": False},
    })
    return StructuredDreamer((H, W), NVEC, cfg)


def _batch():
    state, glob = _state()
    action = torch.zeros(2, 4, CELLS, 7, dtype=torch.long)
    mask = torch.zeros(2, 4, CELLS, MASK_W, dtype=torch.bool)
    mask[..., 5, :] = True
    return {
        "full_state": state,
        "full_globals": glob,
        "action": action,
        "opponent_action": action.clone(),
        "opponent_valid": torch.ones(2, 4, dtype=torch.bool),
        "mask": mask,
        "reward": torch.randn(2, 4),
        "cont": torch.ones(2, 4),
        "is_first": torch.zeros(2, 4, dtype=torch.bool),
    }


def test_sequence_transition_batch_drops_reset_crossings():
    batch = _batch()
    batch["is_first"][0, 2] = True
    out = sequence_transition_batch(batch)
    assert out["state"].shape[0] == 5  # six candidate pairs minus one reset crossing
    assert not out["counterfactual_valid"].any()


def test_agent_heads_do_not_backprop_into_pretrained_world():
    model = _model()
    loss, metrics = structured_agent_head_loss(model, _batch())
    assert loss.isfinite() and metrics["agent/continue_acc"].isfinite()
    loss.backward()
    assert any(p.grad is not None for p in model.scalar_heads.parameters())
    assert any(p.grad is not None for p in model.action_expert.behavior_prior.parameters())
    assert all(p.grad is None for p in model.dynamics.parameters())
    assert all(p.grad is None for p in model.tokenizer.parameters())


def test_opponent_bc_accepts_explicit_none_marker_from_java_channel():
    model = _model()
    batch = _batch()
    batch["opponent_action"][..., 10, 6] = 255
    loss, _ = structured_agent_head_loss(model, batch)
    assert loss.isfinite()


def test_pmpo_routes_actor_and_critic_but_not_world_gradients():
    model = _model()
    state, glob = _state(batch=3, time=1)
    with torch.no_grad():
        z = model.tokenizer.encode(state[:, 0], glob[:, 0])
        imagined = model.imagine(z, horizon=2)
    actor_loss, critic_loss, metrics = structured_pmpo_losses(model, imagined)
    assert actor_loss.isfinite() and critic_loss.isfinite()
    assert 0 <= metrics["ac/advantage_positive_frac"] <= 1
    actor_loss.backward()
    critic_loss.backward()
    assert any(p.grad is not None for p in model.action_expert.actor_policy.parameters())
    assert any(p.grad is not None for p in model.action_expert.critic.parameters())
    assert all(p.grad is None for p in model.dynamics.parameters())


def test_optimizer_parameter_groups_are_disjoint():
    model = _model()
    opts = model.build_optimizers()
    groups = {
        name: {id(p) for group in opt.param_groups for p in group["params"]}
        for name, opt in opts.items()
    }
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert groups[left].isdisjoint(groups[right]), (left, right)
