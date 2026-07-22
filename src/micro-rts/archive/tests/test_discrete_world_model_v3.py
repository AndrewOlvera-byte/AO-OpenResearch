import pytest
import torch

from entrypoints.train_discrete_dynamics import _audit_step_zero_geometry
from models.dreamer_v2 import (
    DiscreteActionTokenizerConfig,
    DiscreteActionTokenizerPretrainer,
    DiscreteDynamicsConfig,
    DiscreteStructuredTokenizer,
    DiscreteStructuredWorldModel,
    DiscreteTokenizerConfig,
    discrete_action_jepa_loss,
    discrete_causal_paired_loss,
    discrete_prior_geometry,
    discrete_reconstruction_loss,
    load_discrete_action_jepa,
)


H = W = 4
CELLS = H * W
MASK_WIDTH = 79


def make_batch(batch=2):
    state = torch.zeros(batch, 1, CELLS, 16, dtype=torch.long)
    state[..., 2] = -1
    state[..., 3] = -1
    state[..., 4] = -1
    state[..., 8:14] = -1
    for cell, role, uid in ((5, 1, 11), (10, 2, 22)):
        state[..., cell, 1] = 1
        state[..., cell, 2] = uid
        state[..., cell, 3] = role
        state[..., cell, 4] = 3
        state[..., cell, 5] = 1
    globals_ = torch.zeros(batch, 1, 8, dtype=torch.long)
    globals_[..., 1:3] = 5
    globals_[..., 6] = -1
    action = torch.zeros(batch, 1, CELLS, 7, dtype=torch.long)
    opponent = torch.zeros_like(action)
    action[..., 5, 0] = 1
    action[..., 5, 1] = 1
    next_state = state.clone()
    next_state[..., 5, 7] = 1
    next_state[..., 5, 8] = 1
    next_state[..., 5, 9] = 1
    next_state[..., 5, 10] = 2
    next_state[..., 5, 11] = 1
    next_state[..., 5, 13] = 0
    next_state[..., 5, 14] = 10
    next_state[..., 5, 15] = 10
    return {
        "state": state,
        "globals": globals_,
        "next_state": next_state,
        "next_globals": globals_.clone(),
        "action": action,
        "opponent_action": opponent,
        "counterfactual_valid": torch.zeros(batch, 1, dtype=torch.bool),
        "counterfactual_next_state": next_state.clone(),
        "counterfactual_next_globals": globals_.clone(),
        "counterfactual_action": action.clone(),
        "counterfactual_opponent_action": opponent.clone(),
    }


def tokenizer_cfg():
    return DiscreteTokenizerConfig(
        d_model=32,
        depth=1,
        n_heads=4,
        spatial_downsample=2,
        codebook_size=16,
        codebook_depth=2,
        n_global_tokens=2,
        max_unit_types=8,
        mask_width=MASK_WIDTH,
        legacy_obs_channels=6,
    )


def action_cfg():
    return DiscreteActionTokenizerConfig(
        d_model=32,
        field_dim=4,
        n_heads=4,
        inverse_depth=1,
        max_action_events=8,
        max_unit_types=8,
    )


def dynamics_cfg():
    return DiscreteDynamicsConfig(
        d_model=32,
        depth=1,
        n_heads=4,
        max_action_events=8,
        max_unit_types=8,
        action_field_dim=4,
        pretrained_action_router=True,
        zero_init_correction=True,
    )


def test_discrete_tokenizer_uses_hard_product_codes_and_backprops():
    batch = make_batch()
    model = DiscreteStructuredTokenizer((H, W), tokenizer_cfg())
    decoded, codes, aux = model(batch["state"], batch["globals"])
    assert codes.dtype == torch.long
    assert codes.shape[-2:] == (model.n_tokens, model.codebook_depth)
    assert model.n_code_tokens == (4 + 2) * 2
    reconstructed = model.decode_codes(codes)
    assert reconstructed["present"].shape == decoded["present"].shape
    assert torch.allclose(reconstructed["present"], decoded["present"])
    loss, metrics = discrete_reconstruction_loss(
        model,
        decoded,
        batch["state"],
        batch["globals"],
        aux["code_probs"],
    )
    loss.backward()
    assert model.code_logits.weight.grad is not None
    assert "dtok/exact_roundtrip" in metrics


def test_discrete_tokenizer_reports_exact_raster_and_mask_certification():
    batch = make_batch()
    model = DiscreteStructuredTokenizer((H, W), tokenizer_cfg())
    decoded, _codes, aux = model(batch["state"], batch["globals"])
    raster = (
        (decoded["legacy_obs"] >= 0)
        .reshape(batch["state"].shape[0], 1, H, W, -1)
        .movedim(-1, -3)
    )
    mask = decoded["mask"] >= 0
    loss, metrics = discrete_reconstruction_loss(
        model,
        decoded,
        batch["state"],
        batch["globals"],
        aux["code_probs"],
        raster,
        mask,
    )
    assert torch.isfinite(loss)
    assert metrics["dtok/exact_raster"] == 1
    assert metrics["dtok/exact_mask"] == 1


def test_discrete_action_jepa_trains_exact_events_and_next_code_router():
    batch = make_batch()
    batch["counterfactual_valid"][0, 0] = True
    batch["counterfactual_action"][0, 0, 5, 1] = 3
    batch["counterfactual_next_state"][0, 0, 5, 10] = 0
    tokenizer = DiscreteStructuredTokenizer((H, W), tokenizer_cfg()).eval()
    tokenizer.requires_grad_(False)
    model = DiscreteActionTokenizerPretrainer(tokenizer, (H, W), action_cfg())
    loss, metrics = discrete_action_jepa_loss(model, tokenizer, batch)
    loss.backward()
    assert model.action_encoder.project[0].weight.grad is not None
    assert model.router.state_in.weight.grad is not None
    assert tokenizer.code_logits.weight.grad is None
    assert "dajepa/changed_code_acc" in metrics


def test_discrete_causal_transformer_has_no_flow_and_uses_categorical_loss():
    batch = make_batch()
    batch["counterfactual_valid"][0, 0] = True
    batch["counterfactual_action"][0, 0, 5, 1] = 3
    batch["counterfactual_next_state"][0, 0, 5, 10] = 0
    model = DiscreteStructuredWorldModel((H, W), tokenizer_cfg(), dynamics_cfg())
    assert not hasattr(model.dynamics, "flow_x_head")
    with torch.no_grad():
        codes = model.tokenizer.encode_codes(
            batch["state"][:, 0], batch["globals"][:, 0]
        )
        events, valid, _ = model.action_events(
            batch["state"][:, 0], batch["action"][:, 0], batch["opponent_action"][:, 0]
        )
        output = model.dynamics(
            codes,
            codes,
            events,
            valid,
            router_state_tokens=model.router_tokens(codes),
        )
        assert torch.equal(output["logits"], output["prior_logits"])
        generated = model.dynamics.generate(
            codes,
            events,
            valid,
            router_state_tokens=model.router_tokens(codes),
        )
        assert generated.shape == codes.shape
    loss, metrics = discrete_causal_paired_loss(model, batch)
    loss.backward()
    assert model.dynamics.transformer.layers[0].linear1.weight.grad is not None
    assert model.dynamics.correction_heads[0].weight.grad is not None
    assert "dynamics/exact_cell_teacher_forced" in metrics


def test_discrete_action_jepa_checkpoint_transfers_router_exactly():
    tokenizer = DiscreteStructuredTokenizer((H, W), tokenizer_cfg())
    source = DiscreteActionTokenizerPretrainer(tokenizer, (H, W), action_cfg())
    target = DiscreteStructuredWorldModel((H, W), tokenizer_cfg(), dynamics_cfg())
    target.tokenizer.load_state_dict(tokenizer.state_dict())
    load_discrete_action_jepa(target, {"model": source.state_dict()})
    assert torch.equal(
        source.action_encoder.project[0].weight,
        target.dynamics.action_encoder.project[0].weight,
    )
    assert torch.equal(
        source.router.heads[0].weight,
        target.dynamics.action_router.heads[0].weight,
    )
    assert torch.equal(source.action_position, target.dynamics.action_position)

    batch = make_batch()
    with torch.no_grad():
        codes = tokenizer.encode_codes(batch["state"][:, 0], batch["globals"][:, 0])
        events, valid, _ = source.action_events(
            batch["state"][:, 0],
            batch["action"][:, 0],
            batch["opponent_action"][:, 0],
        )
        source_logits = source.forward_logits(
            tokenizer.embed_code_tokens(codes), source.encode_events(events), valid
        )
        target_logits = target.dynamics.action_router(
            target.router_tokens(codes), target.dynamics.encode_actions(events), valid
        )
    assert torch.equal(source_logits, target_logits)


def test_discrete_prior_geometry_reports_paired_categorical_effects():
    batch = make_batch()
    batch["counterfactual_valid"][0, 0] = True
    batch["counterfactual_action"][0, 0, 5, 1] = 3
    batch["counterfactual_next_state"][0, 0, 5, 10] = 0
    model = DiscreteStructuredWorldModel((H, W), tokenizer_cfg(), dynamics_cfg())
    metrics = discrete_prior_geometry(model, batch)
    assert metrics["geometry/paired_rows"] == 1
    assert metrics["geometry/paired_effect_codes"] >= 0
    assert all(torch.isfinite(value) for value in metrics.values())


def test_discrete_step_zero_geometry_gate_rejects_uncertified_prior():
    batch = make_batch()
    model = DiscreteStructuredWorldModel((H, W), tokenizer_cfg(), dynamics_cfg())
    with pytest.raises(RuntimeError, match="step-zero categorical geometry failed"):
        _audit_step_zero_geometry(
            model,
            [batch],
            torch.device("cpu"),
            {
                "preflight_batches": 1,
                "preflight_min_effect_codes": 10**9,
                "preflight_min_bidirectional_preference": 0.0,
                "preflight_max_action_overflow": 1.0,
            },
        )


def test_discrete_router_handles_empty_joint_action_without_nans():
    batch = make_batch()
    batch["action"].zero_()
    batch["opponent_action"].zero_()
    tokenizer = DiscreteStructuredTokenizer((H, W), tokenizer_cfg()).eval()
    tokenizer.requires_grad_(False)
    action_model = DiscreteActionTokenizerPretrainer(tokenizer, (H, W), action_cfg())
    action_loss, action_metrics = discrete_action_jepa_loss(
        action_model, tokenizer, batch
    )
    assert torch.isfinite(action_loss)
    assert all(torch.isfinite(value) for value in action_metrics.values())

    dynamics_model = DiscreteStructuredWorldModel(
        (H, W), tokenizer_cfg(), dynamics_cfg()
    )
    dynamics_loss, dynamics_metrics = discrete_causal_paired_loss(dynamics_model, batch)
    assert torch.isfinite(dynamics_loss)
    assert all(torch.isfinite(value) for value in dynamics_metrics.values())
