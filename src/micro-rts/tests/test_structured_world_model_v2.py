import numpy as np
import pytest
import torch

from collectors.offline_data import HDF5Writer, MRTSSequenceDataset, build_mrts_loader
from models.dreamer_v2 import (
    ActionTokenizerConfig,
    ActionTokenizerPretrainer,
    FactorizedActionEventEncoder,
    StructuredDynamicsConfig,
    StructuredTokenizerConfig,
    StructuredWorldModelV2,
    action_tokenizer_ssl_loss,
    dense_actions_to_events,
    structured_reconstruction_loss,
    validate_structured_state,
)
from models.dreamer_v2.dynamics import (
    ActionEventEncoder,
    structured_causal_paired_loss,
    structured_dreamer4_loss,
    structured_flow_loss,
)
from entrypoints.migrate_structured_none_actions import main as migrate_none_actions
from entrypoints.train_dreamer_dynamics import load_pretrained_action_tokenizer
from entrypoints.pretrain_common import make_lr_scheduler


H = W = 4
CELLS = H * W
NVEC = [CELLS, 6, 4, 4, 4, 4, 7, 49]
MASK_W = 1 + sum(NVEC[1:])


def make_state(batch=2, time=3):
    s = torch.zeros(batch, time, CELLS, 16, dtype=torch.long)
    s[..., 2] = -1
    s[..., 3] = -1
    s[..., 4] = -1
    s[..., 8:14] = -1
    # Own worker at (1,1), opponent worker at (2,2).
    for cell, role, uid in ((5, 1, 11), (10, 2, 22)):
        s[..., cell, 1] = 1
        s[..., cell, 2] = uid
        s[..., cell, 3] = role
        s[..., cell, 4] = 3
        s[..., cell, 5] = 1
    g = torch.zeros(batch, time, 8, dtype=torch.long)
    g[..., 0] = torch.arange(time)
    g[..., 1:3] = 5
    g[..., 6] = -1
    return s, g


def tiny_model(action_encoder_type="legacy", **dynamics_overrides):
    tc = StructuredTokenizerConfig(
        d_cell=16,
        d_latent=16,
        downsample=2,
        depth=1,
        n_heads=4,
        max_unit_types=8,
        max_entities=8,
        mask_width=MASK_W,
        legacy_obs_channels=6,
    )
    dc = StructuredDynamicsConfig(
        d_model=32,
        depth=2,
        n_heads=4,
        max_action_events=8,
        k_max=4,
        prior_fraction=0.5,
        skip_fraction=0.5,
        action_encoder_type=action_encoder_type,
        action_field_dim=8,
        **dynamics_overrides,
    )
    return StructuredWorldModelV2((H, W), tc, dc)


def test_staged_lr_schedule_reproduces_causal_paired_continuations():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = make_lr_scheduler(
        optimizer,
        total_steps=160000,
        warmup_steps=1000,
        min_frac=0.025,
        stages=(
            {"end_step": 60000, "start_frac": 1.0, "end_frac": 0.1},
            {
                "end_step": 100000,
                "start_frac": 0.3,
                "end_frac": 0.1,
                "warmup_steps": 500,
            },
            {"end_step": 160000, "start_frac": 0.1, "end_frac": 0.025},
        ),
    )
    factor = scheduler.lr_lambdas[0]
    assert factor(999) == pytest.approx(1.0)
    assert factor(60000) == pytest.approx(0.1)
    assert factor(60001) == pytest.approx(0.1004)
    assert factor(60500) == pytest.approx(0.3)
    assert factor(100000) == pytest.approx(0.1)
    assert factor(100001) == pytest.approx(0.1, rel=1e-6)
    assert factor(160000) == pytest.approx(0.025)


def test_schema_and_action_events_bind_source_target():
    state, glob = make_state(batch=1, time=1)
    validate_structured_state(state, glob, (H, W))
    action = torch.zeros(1, 1, CELLS, 7, dtype=torch.long)
    opp = torch.zeros_like(action)
    action[..., 5, 0] = 1  # move own worker
    action[..., 5, 1] = 1  # right: (1,1) -> (2,1)
    opp[..., 10, 0] = 5  # opponent attack
    opp[..., 10, 6] = 17  # 7x7 offset: dx=0,dy=-1 -> (2,1)
    events, valid, overflow = dense_actions_to_events(
        state, action, opp, (H, W), max_events=8
    )
    assert valid.sum() == 2 and overflow.item() == 0
    own, enemy = events[0, 0, 0], events[0, 0, 1]
    assert own.tolist()[:9] == [1, 11, 1, 1, 1, 2, 1, 1, -1]
    assert enemy.tolist()[:7] == [2, 22, 2, 2, 5, 2, 1]


def test_explicit_none_marker_becomes_an_action_event():
    state, _ = make_state(batch=1, time=1)
    action = torch.zeros(1, 1, CELLS, 7, dtype=torch.long)
    opp = torch.zeros_like(action)
    opp[..., 10, 6] = 255
    events, valid, overflow = dense_actions_to_events(
        state, action, opp, (H, W), max_events=8
    )
    assert valid.sum() == 1 and overflow.item() == 0
    assert events[0, 0, 0].tolist()[:5] == [2, 22, 2, 2, 0]


def test_structured_tokenizer_reconstructs_and_backprops():
    model = tiny_model()
    state, glob = make_state()
    decoded, z = model.tokenizer(state, glob)
    assert z.shape == (2, 3, 15, 16)  # 2x2 spatial + 8 entities + 3 globals
    loss, metrics = structured_reconstruction_loss(
        model.tokenizer, decoded, state, glob
    )
    assert torch.isfinite(loss) and "tok/remaining_acc" not in metrics
    loss.backward()
    assert model.tokenizer.compress.weight.grad is not None


def test_learned_representation_is_invariant_to_raw_unit_ids():
    torch.manual_seed(0)
    model = tiny_model().eval()
    state, glob = make_state(batch=1, time=1)
    changed = state.clone()
    changed[..., 5, 2] = 987654
    changed[..., 10, 2] = 123456
    with torch.no_grad():
        original_z = model.tokenizer.encode(state, glob)
        changed_z = model.tokenizer.encode(changed, glob)
    torch.testing.assert_close(original_z, changed_z, rtol=0, atol=0)


def test_action_encoder_is_invariant_to_raw_unit_ids():
    torch.manual_seed(0)
    encoder = ActionEventEncoder(16, (H, W), max_unit_types=8).eval()
    events = torch.tensor([[[1, 11, 1, 1, 1, 2, 1, 1, -1, -1]]])
    changed = events.clone()
    changed[..., 1] = 987654
    with torch.no_grad():
        original = encoder(events)
        modified = encoder(changed)
    torch.testing.assert_close(original, modified, rtol=0, atol=0)


def test_factorized_action_encoder_keeps_multiple_tokens_and_ignores_unit_ids():
    torch.manual_seed(0)
    encoder = FactorizedActionEventEncoder(
        32, (H, W), max_unit_types=8, field_dim=8
    ).eval()
    events = torch.tensor(
        [[
            [1, 11, 1, 1, 1, 2, 1, 1, -1, -1],
            [2, 22, 2, 2, 5, 2, 1, -1, -1, 17],
        ]]
    )
    changed = events.clone()
    changed[..., 1] += 100000
    with torch.no_grad():
        encoded = encoder(events)
        modified = encoder(changed)
    assert encoded.shape == (1, 2, 32)
    assert not torch.equal(encoded[:, 0], encoded[:, 1])
    torch.testing.assert_close(encoded, modified, rtol=0, atol=0)


def test_action_tokenizer_dual_ssl_objective_backprops_through_encoder():
    torch.manual_seed(0)
    world = tiny_model(action_encoder_type="factorized")
    cfg = ActionTokenizerConfig(
        d_model=32,
        field_dim=8,
        n_heads=4,
        inverse_depth=1,
        max_action_events=8,
        max_unit_types=8,
    )
    action_model = ActionTokenizerPretrainer(
        world.tokenizer.n_tokens,
        world.tokenizer.d_latent,
        (H, W),
        cfg,
    )
    state, glob = make_state(batch=2, time=2)
    nxt = state.clone()
    nxt[..., 5, 5] += 1
    cf_nxt = state.clone()
    cf_nxt[..., 5, 5] += 2
    action = torch.zeros(2, 2, CELLS, 7, dtype=torch.long)
    opponent = torch.zeros_like(action)
    action[..., 5, 0] = 1
    action[..., 5, 1] = 1
    cf_action = action.clone()
    cf_action[..., 5, 1] = 2
    batch = {
        "state": state,
        "globals": glob,
        "next_state": nxt,
        "next_globals": glob.clone(),
        "action": action,
        "opponent_action": opponent,
        "counterfactual_action": cf_action,
        "counterfactual_opponent_action": opponent.clone(),
        "counterfactual_next_state": cf_nxt,
        "counterfactual_next_globals": glob.clone(),
        "counterfactual_valid": torch.tensor(
            [[True, False], [True, True]], dtype=torch.bool
        ),
    }
    loss, metrics = action_tokenizer_ssl_loss(
        action_model, world.tokenizer.eval(), batch
    )
    assert torch.isfinite(loss)
    assert set(
        (
            "action_tok/reconstruction",
            "action_tok/inverse",
            "action_tok/forward",
            "action_tok/paired_effect",
            "action_tok/effect_cosine",
            "action_tok/effect_norm_ratio_aggregate",
        )
    ) <= set(metrics)
    loss.backward()
    assert action_model.action_encoder.project[0].weight.grad is not None
    assert action_model.forward_head[0].weight.grad is not None
    assert action_model.inverse_queries.grad is not None
    assert action_model.event_heads["action_type"].weight.grad is not None


def test_action_tokenizer_checkpoint_loads_encoder_and_slot_positions(tmp_path):
    torch.manual_seed(0)
    world = tiny_model(action_encoder_type="factorized")
    cfg = ActionTokenizerConfig(
        d_model=32,
        field_dim=8,
        n_heads=4,
        inverse_depth=1,
        max_action_events=8,
        max_unit_types=8,
    )
    pretrained = ActionTokenizerPretrainer(
        world.tokenizer.n_tokens,
        world.tokenizer.d_latent,
        (H, W),
        cfg,
    )
    path = tmp_path / "action_tokenizer.pt"
    torch.save({"model": pretrained.state_dict(), "step": 123}, path)
    world.dynamics.action_encoder.requires_grad_(False)
    payload = load_pretrained_action_tokenizer(world, path)
    assert payload["step"] == 123
    for name, expected in pretrained.action_encoder.state_dict().items():
        torch.testing.assert_close(
            world.dynamics.action_encoder.state_dict()[name], expected
        )
    torch.testing.assert_close(
        world.dynamics.action_position, pretrained.action_position
    )


def test_pretrained_router_transfers_exact_residual_geometry(tmp_path):
    torch.manual_seed(0)
    world = tiny_model(
        action_encoder_type="factorized",
        residual_prediction=True,
        pretrained_action_router=True,
        explicit_spatial_action_routing=True,
        mask_empty_entity_tokens=True,
    )
    cfg = ActionTokenizerConfig(
        d_model=32,
        field_dim=8,
        n_heads=4,
        inverse_depth=1,
        max_action_events=8,
        max_unit_types=8,
    )
    pretrained = ActionTokenizerPretrainer(
        world.tokenizer.n_tokens,
        world.tokenizer.d_latent,
        (H, W),
        cfg,
    ).eval()
    path = tmp_path / "action_tokenizer_router.pt"
    torch.save({"model": pretrained.state_dict(), "step": 321}, path)
    load_pretrained_action_tokenizer(world, path)
    world.eval()

    state, glob = make_state(batch=2, time=1)
    action = torch.zeros(2, CELLS, 7, dtype=torch.long)
    opponent = torch.zeros_like(action)
    action[:, 5, 0] = 1
    action[:, 5, 1] = 1
    flat_state, flat_glob = state[:, 0], glob[:, 0]
    events, valid, _ = world.action_events(flat_state, action, opponent)
    z = world.dynamics.normalize(world.tokenizer.encode(flat_state, flat_glob))
    action_tokens = pretrained.encode_events(events)
    expected_delta = pretrained.forward_delta(z, action_tokens, valid)
    zero = torch.zeros(z.shape[0], dtype=torch.long)
    with torch.no_grad():
        result = world.dynamics(
            z,
            torch.zeros_like(z),
            events,
            valid,
            zero,
            zero,
            state_token_valid=world.state_token_valid(flat_state),
        )
    # The transition correction and explicit route are zero-initialized, so the
    # new world model begins at the exact pretrained delta predictor.
    torch.testing.assert_close(result["prior_delta"], expected_delta)
    torch.testing.assert_close(result["base"], z + expected_delta)
    assert torch.count_nonzero(result["correction"]) == 0
    assert world.state_token_valid(flat_state)[:, world.tokenizer.n_spatial :].sum() == 10


def test_spatial_action_route_binds_only_source_and_destination_slots():
    torch.manual_seed(0)
    world = tiny_model(
        action_encoder_type="factorized",
        residual_prediction=True,
        explicit_spatial_action_routing=True,
    )
    torch.nn.init.eye_(world.dynamics.source_spatial_route.weight)
    torch.nn.init.eye_(world.dynamics.target_spatial_route.weight)
    event = torch.tensor([[[1, -1, 0, 0, 1, 3, 3, 1, -1, -1]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    tokens = world.dynamics.encode_action_tokens(event)
    route = world.dynamics.spatial_action_route(event, tokens, valid)
    active = route.abs().sum(-1).bool().nonzero(as_tuple=False)[:, 1].tolist()
    # 4x4 cells downsample to a 2x2 spatial lattice: (0,0)->slot 0 and
    # (3,3)->slot 3. No entity/global query receives the explicit route.
    assert active == [0, 3]


def test_spatial_action_route_supports_bfloat16_autocast():
    world = tiny_model(
        action_encoder_type="factorized",
        residual_prediction=True,
        explicit_spatial_action_routing=True,
    )
    event = torch.tensor([[[1, -1, 0, 0, 1, 3, 3, 1, -1, -1]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        tokens = world.dynamics.encode_action_tokens(event)
        route = world.dynamics.spatial_action_route(event, tokens, valid)
    assert route.dtype == torch.bfloat16
    assert torch.isfinite(route).all()


def test_causal_flow_loss_and_sampling():
    torch.manual_seed(0)
    model = tiny_model()
    state, glob = make_state()
    nxt = state.clone()
    nxt[..., 5, 5] = 2  # changed HP
    action = torch.zeros(2, 3, CELLS, 7, dtype=torch.long)
    opp = torch.zeros_like(action)
    action[..., 5, 0] = 1
    action[..., 5, 1] = 1
    batch = {
        "state": state,
        "globals": glob,
        "next_state": nxt,
        "next_globals": glob.clone(),
        "action": action,
        "opponent_action": opp,
    }
    loss, metrics = structured_flow_loss(model, batch, structured_coef=0.1)
    assert torch.isfinite(loss)
    assert set(("flow/matching", "flow/prior", "flow/skip")) <= set(metrics)
    loss.backward()
    assert model.dynamics.flow_x_head.weight.grad is not None
    assert model.dynamics.shortcut_skip_head.weight.grad is not None
    with torch.no_grad():
        z = model.tokenizer.encode(state[:, 0], glob[:, 0])
        ev, valid, _ = model.action_events(state[:, 0], action[:, 0], opp[:, 0])
        pred = model.dynamics.sample_next(z, ev, valid, steps=2)
    assert pred.shape == z.shape and torch.isfinite(pred).all()


def test_dreamer4_structured_loss_uses_paper_objectives():
    torch.manual_seed(0)
    model = tiny_model()
    state, glob = make_state()
    nxt = state.clone()
    nxt[..., 5, 5] = 2
    action = torch.zeros(2, 3, CELLS, 7, dtype=torch.long)
    opp = torch.zeros_like(action)
    action[..., 5, 0] = 1
    action[..., 5, 1] = 1
    batch = {
        "state": state,
        "globals": glob,
        "next_state": nxt,
        "next_globals": glob.clone(),
        "action": action,
        "opponent_action": opp,
    }
    loss, metrics = structured_dreamer4_loss(model, batch, self_frac=0.25)
    assert torch.isfinite(loss)
    assert set(
        (
            "flow/matching",
            "flow/mse",
            "flow/consistency",
            "flow/total",
            "wm/total",
        )
    ) <= set(metrics)
    assert "flow/prior" not in metrics and "tok/total" not in metrics
    loss.backward()
    assert model.dynamics.flow_x_head.weight.grad is not None
    assert model.dynamics.shortcut_skip_head.weight.grad is None

    with pytest.raises(ValueError, match="self_frac"):
        structured_dreamer4_loss(model, batch, self_frac=1.0)


def test_causal_paired_loss_trains_deployed_one_step_query():
    torch.manual_seed(0)
    model = tiny_model(residual_prediction=True)
    model.dynamics.cfg.initial_noise = "zero"
    state, glob = make_state()
    nxt = state.clone()
    nxt[..., 5, 5] = 2
    cf_nxt = state.clone()
    cf_nxt[..., 5, 5] = 3
    # One stored paired branch has no engine effect. It must remain exactly zero
    # in latent space rather than becoming a sub-batch attention roundoff effect.
    cf_nxt[0, 0] = nxt[0, 0]
    action = torch.zeros(2, 3, CELLS, 7, dtype=torch.long)
    opp = torch.zeros_like(action)
    action[..., 5, 0] = 1
    action[..., 5, 1] = 1
    cf_action = action.clone()
    cf_action[..., 5, 1] = 2
    batch = {
        "state": state,
        "globals": glob,
        "next_state": nxt,
        "next_globals": glob.clone(),
        "action": action,
        "opponent_action": opp,
        "counterfactual_action": cf_action,
        "counterfactual_opponent_action": opp.clone(),
        "counterfactual_next_state": cf_nxt,
        "counterfactual_next_globals": glob.clone(),
        "counterfactual_valid": torch.tensor(
            [[True, False, True], [True, True, False]]
        ),
    }
    loss, metrics = structured_causal_paired_loss(
        model,
        batch,
        padding_token_weight=0.05,
        effect_cosine_coef=0.25,
        effect_norm_coef=0.1,
        canonical_grounding_coef=0.1,
        canonical_changed_boost=4.0,
        residual_correction_coef=0.5,
    )
    assert torch.isfinite(loss)
    assert set(
        (
            "causal/factual",
            "causal/counterfactual",
            "causal/effect",
            "causal/effect_cosine",
            "causal/effect_norm_ratio",
            "causal/effect_norm_ratio_aggregate",
            "causal/effect_cosine_loss",
            "causal/effect_norm_loss",
            "causal/grounding_factual",
            "causal/correction_regularizer",
            "grounding/factual_present_acc",
            "causal/valid_token_fraction",
        )
    ) <= set(metrics)
    # Eight entity slots are allocated, but only the two occupied slots count.
    assert metrics["causal/valid_token_fraction"] < 1.0
    assert metrics["causal/effect_geometry_rows"] == 3
    loss.backward()
    assert model.dynamics.flow_x_head.weight.grad is not None
    assert model.dynamics.shortcut_skip_head.weight.grad is None

    # A one-step sample is exactly the trained tau=0,d=1 query while the unused
    # shortcut head remains at its zero initialization.
    with torch.no_grad():
        flat_state, flat_glob = state[:, 0], glob[:, 0]
        z = model.tokenizer.encode(flat_state, flat_glob)
        normalized = model.dynamics.normalize(z)
        events, valid, _ = model.action_events(flat_state, action[:, 0], opp[:, 0])
        noise = model.dynamics.initial_noise_like(normalized)
        assert torch.count_nonzero(noise) == 0
        zero = torch.zeros(z.shape[0], dtype=torch.long)
        direct = model.dynamics(normalized, noise, events, valid, zero, zero)["base"]
        sampled = model.dynamics.sample_next(z, events, valid, steps=1)
    torch.testing.assert_close(model.dynamics.normalize(sampled), direct)


def test_hdf5_v4_structured_roundtrip(tmp_path):
    path = tmp_path / "v4.h5"
    writer = HDF5Writer(
        path,
        obs_shape=(6, H, W),
        action_shape=(CELLS, 7),
        mask_shape=(CELLS, MASK_W),
        action_nvec=NVEC,
        grid_hw=(H, W),
        reward_weight=[1] * 6,
        maps=["m"],
        opponents=["b"],
        store_full_state=True,
        state_shape=(CELLS, 16),
        store_counterfactual=True,
        chunk_rows=4,
    )
    state, glob = make_state(batch=1, time=3)
    for t in range(3):
        writer.add_batch(
            {
                "obs": np.zeros((1, 6, H, W), np.uint8),
                "action": np.zeros((1, CELLS, 7), np.uint8),
                "opponent_action": np.zeros((1, CELLS, 7), np.uint8),
                "mask": np.zeros((1, CELLS, MASK_W), np.uint8),
                "reward": np.zeros(1, np.float32),
                "raw_rewards": np.zeros((1, 6), np.float32),
                "done": np.zeros(1, bool),
                "is_first": np.array([t == 0]),
                "state": state[:, t],
                "next_state": state[:, t],
                "globals": glob[:, t],
                "next_globals": glob[:, t],
                "counterfactual_action": np.zeros((1, CELLS, 7), np.uint8),
                "counterfactual_opponent_action": np.zeros((1, CELLS, 7), np.uint8),
                "counterfactual_next_state": state[:, t],
                "counterfactual_next_globals": glob[:, t],
                "counterfactual_valid": np.array([t % 2 == 0]),
            }
        )
    writer.end_segment(map_id=0, opponent_id=0)
    writer.close()
    ds = MRTSSequenceDataset(path, seq_len=2, task="structured_dynamics")
    item = ds[0]
    assert item["state"].shape == (2, CELLS, 16)
    assert item["next_globals"].shape == (2, 8)
    assert item["state"].dtype == torch.int64
    assert int(ds.attrs["format_version"]) == 4
    ds.close()
    paired = MRTSSequenceDataset(path, seq_len=2, task="structured_dynamics_paired")
    paired_item = paired[0]
    assert paired_item["counterfactual_next_state"].shape == (2, CELLS, 16)
    assert paired_item["counterfactual_valid"].dtype == torch.bool
    paired.close()
    action_tok = MRTSSequenceDataset(
        path, seq_len=2, task="structured_action_tokenizer"
    )
    action_item = action_tok[0]
    assert action_item["counterfactual_next_state"].shape == (2, CELLS, 16)
    assert action_item["counterfactual_valid"].dtype == torch.bool
    action_tok.close()
    balanced = build_mrts_loader(
        path,
        task="structured_dynamics_paired",
        seq_len=1,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        paired_batch_fraction=0.5,
    )
    balanced_batch = next(iter(balanced))
    assert balanced_batch["counterfactual_valid"].sum() == 1


def test_none_action_migration_changes_actions_not_states(tmp_path):
    path = tmp_path / "legacy_v4.h5"
    writer = HDF5Writer(
        path,
        obs_shape=(6, H, W),
        action_shape=(CELLS, 7),
        mask_shape=(CELLS, MASK_W),
        action_nvec=NVEC,
        grid_hw=(H, W),
        reward_weight=[1] * 6,
        maps=["m"],
        opponents=["b"],
        store_full_state=True,
        state_shape=(CELLS, 16),
        chunk_rows=1,
    )
    state, glob = make_state(batch=1, time=1)
    nxt = state.clone()
    nxt[..., 10, 7] = 1
    nxt[..., 10, 8] = 0
    nxt[..., 10, 13] = 0
    nxt[..., 10, 14] = 10
    nxt[..., 10, 15] = 9
    writer.add_batch(
        {
            "obs": np.zeros((1, 6, H, W), np.uint8),
            "action": np.zeros((1, CELLS, 7), np.uint8),
            "opponent_action": np.zeros((1, CELLS, 7), np.uint8),
            "mask": np.zeros((1, CELLS, MASK_W), np.uint8),
            "reward": np.zeros(1, np.float32),
            "raw_rewards": np.zeros((1, 6), np.float32),
            "done": np.zeros(1, bool),
            "is_first": np.ones(1, bool),
            "state": state[:, 0],
            "next_state": nxt[:, 0],
            "globals": glob[:, 0],
            "next_globals": glob[:, 0],
        }
    )
    writer.end_segment(map_id=0, opponent_id=0)
    writer.close()
    original_state = state[:, 0].numpy().copy()
    migrate_none_actions(["--data", str(path), "--write", "--chunk", "1"])
    import h5py

    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["state"][:], original_state)
        assert int(f["opponent_action"][0, 10, 6]) == 255
    action_tokenizer_ssl_loss,
