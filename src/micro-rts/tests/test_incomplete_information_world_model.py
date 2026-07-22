import h5py
import numpy as np
import torch
import torch.nn.functional as F

from collectors.offline_data import MRTSSequenceDataset
from models.dreamer_v2 import StructuredTokenizer, StructuredTokenizerConfig
from models.incomplete_info import (
    BeliefDynamicsConfig,
    BeliefDynamicsModel,
    CausalHistoryTransformer,
    EgoTokenizerConfig,
    EgoTokenizerPretrainer,
    HistoryConfig,
    IncompleteInformationWorldModel,
    IntentPriorConfig,
    JointFlowConfig,
    OpponentPlanTokenizer,
    OpponentPlanTokenizerConfig,
    OpponentIntentPriorModel,
    SelfActionTokenizer,
    SelfActionTokenizerConfig,
    IncompleteBeliefDreamer,
    belief_dynamics_loss,
    ego_tokenizer_loss,
    joint_flow_world_model_loss,
    opponent_intent_prior_loss,
    opponent_plan_tokenizer_loss,
    self_action_tokenizer_loss,
)


H = W = 4
OBS_GROUPS = (5, 5, 3, 8, 6)


def _batch(batch=2, time=4):
    state = torch.zeros(batch, time, H * W, 16, dtype=torch.long)
    state[..., 2:4] = -1
    state[..., 8:14] = -1
    globals_ = torch.zeros(batch, time, 8, dtype=torch.long)
    action = torch.zeros(batch, time, H * W, 8, dtype=torch.long)
    obs = torch.cat(
        [
            F.one_hot(torch.zeros(batch, time, H, W, dtype=torch.long), size)
            .movedim(-1, -3)
            .float()
            for size in OBS_GROUPS
        ],
        dim=-3,
    )
    return {
        "state": state,
        "globals": globals_,
        "next_state": state.clone(),
        "next_globals": globals_.clone(),
        "action": action,
        "opponent_action": action.clone(),
        "local_obs": obs,
        "local_visibility": torch.ones(batch, time, 1, H, W, dtype=torch.bool),
        "is_first": torch.zeros(batch, time, dtype=torch.bool),
    }


def _components():
    tokenizer_cfg = StructuredTokenizerConfig(
        d_cell=16,
        d_latent=16,
        downsample=2,
        depth=1,
        n_heads=4,
        max_entities=8,
        legacy_obs_channels=27,
        mask_width=79,
    )
    teacher = StructuredTokenizer((H, W), tokenizer_cfg).requires_grad_(False).eval()
    ego = EgoTokenizerConfig(d_model=16, d_latent=16, depth=1, n_heads=4, n_registers=2)
    action = SelfActionTokenizerConfig(
        d_model=32,
        d_latent=16,
        field_dim=4,
        n_heads=4,
        max_action_events=4,
    )
    opponent = OpponentPlanTokenizerConfig(
        d_model=32,
        d_latent=16,
        field_dim=4,
        n_heads=4,
        depth=1,
        max_action_events=4,
        n_plan_tokens=2,
        horizons=(0, 1, 2),
    )
    history = HistoryConfig(
        d_model=32, depth=1, n_heads=4, n_registers=2, context_length=8
    )
    flow = JointFlowConfig(d_model=32, depth=1, n_heads=4, sample_steps=2)
    return teacher, ego, action, opponent, history, flow


def test_all_staged_losses_are_finite_and_backpropagate():
    batch = _batch()
    teacher, ego_cfg, action_cfg, opponent_cfg, history_cfg, flow_cfg = _components()

    ego = EgoTokenizerPretrainer((H, W), ego_cfg)
    ego_loss, _ = ego_tokenizer_loss(ego, batch, full_state_tokenizer=teacher)
    ego_loss.backward()
    assert torch.isfinite(ego_loss)
    assert ego.tokenizer.encoder[0].weight.grad is not None

    action = SelfActionTokenizer(teacher.n_tokens, (H, W), action_cfg)
    action_loss, _ = self_action_tokenizer_loss(action, teacher, batch)
    action_loss.backward()
    assert torch.isfinite(action_loss)
    assert action.event_encoder.role.weight.grad is not None

    opponent = OpponentPlanTokenizer(teacher.n_tokens, (H, W), opponent_cfg)
    opponent_loss, _ = opponent_plan_tokenizer_loss(opponent, teacher, batch)
    opponent_loss.backward()
    assert torch.isfinite(opponent_loss)
    assert opponent.plan_queries.grad is not None

    world = IncompleteInformationWorldModel(
        teacher,
        (H, W),
        ego_cfg=ego_cfg,
        self_action_cfg=action_cfg,
        opponent_cfg=opponent_cfg,
        history_cfg=history_cfg,
        flow_cfg=flow_cfg,
    ).freeze_teachers()
    joint_loss, metrics = joint_flow_world_model_loss(world, batch)
    joint_loss.backward()
    assert torch.isfinite(joint_loss)
    assert world.flow.state_out.weight.grad is not None
    assert all(torch.isfinite(value) for value in metrics.values())


def test_history_is_causal_and_flow_partition_is_exact():
    torch.manual_seed(4)
    history = CausalHistoryTransformer(
        n_spatial=4,
        max_action_events=3,
        input_dim=8,
        cfg=HistoryConfig(
            d_model=16, depth=2, n_heads=4, n_registers=2, context_length=8
        ),
    ).eval()
    spatial = torch.randn(1, 5, 4, 8)
    action = torch.randn(1, 5, 3, 8)
    valid = torch.ones(1, 5, 3, dtype=torch.bool)
    first = history(spatial, action, valid)["registers"]
    changed = spatial.clone()
    changed[:, -1] += 100
    second = history(changed, action, valid)["registers"]
    torch.testing.assert_close(first[:, :-1], second[:, :-1], atol=1e-5, rtol=1e-5)

    teacher, ego_cfg, action_cfg, opponent_cfg, history_cfg, flow_cfg = _components()
    world = IncompleteInformationWorldModel(
        teacher,
        (H, W),
        ego_cfg=ego_cfg,
        self_action_cfg=action_cfg,
        opponent_cfg=opponent_cfg,
        history_cfg=history_cfg,
        flow_cfg=flow_cfg,
    )
    joint = torch.randn(2, world.flow.n_tokens, world.flow.latent_dim)
    state, plan = world.split_joint(joint)
    assert state.shape[-2] == teacher.n_tokens
    assert plan.shape[-2] == opponent_cfg.n_plan_tokens
    torch.testing.assert_close(torch.cat((state, plan), dim=-2), joint)


def test_multimodal_intent_prior_is_history_conditioned_and_trainable():
    torch.manual_seed(7)
    batch = _batch(batch=2, time=4)
    teacher, ego_cfg, action_cfg, opponent_cfg, history_cfg, _ = _components()
    model = OpponentIntentPriorModel(
        teacher,
        (H, W),
        ego_cfg=ego_cfg,
        self_action_cfg=action_cfg,
        opponent_cfg=opponent_cfg,
        history_cfg=history_cfg,
        intent_cfg=IntentPriorConfig(
            d_model=32, depth=1, n_heads=4, n_modes=3, contrast_dim=8
        ),
    ).freeze_teachers()
    loss, metrics = opponent_intent_prior_loss(model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.intent_prior.plan_out.weight.grad is not None
    assert model.history.spatial_in.weight.grad is not None
    assert metrics["intent/active_modes"] >= 1
    sample = model.sample_intent(batch)
    assert sample["all_plan_tokens"].shape[-3] == 3
    assert sample["plan_tokens"].shape[-2] == opponent_cfg.n_plan_tokens


def test_belief_dynamics_uses_frozen_intent_condition_and_backpropagates():
    torch.manual_seed(11)
    batch = _batch(batch=2, time=4)
    teacher, ego_cfg, action_cfg, opponent_cfg, history_cfg, _ = _components()
    intent = OpponentIntentPriorModel(
        teacher,
        (H, W),
        ego_cfg=ego_cfg,
        self_action_cfg=action_cfg,
        opponent_cfg=opponent_cfg,
        history_cfg=history_cfg,
        intent_cfg=IntentPriorConfig(
            d_model=32, depth=1, n_heads=4, n_modes=3, contrast_dim=8
        ),
    )
    model = BeliefDynamicsModel(
        intent,
        BeliefDynamicsConfig(d_model=32, depth=1, n_heads=4, sample_steps=2),
    ).freeze_conditioner()
    loss, metrics = belief_dynamics_loss(
        model,
        batch,
        action_rank_coef=2.0,
        occupied_boost=4.0,
        hidden_occupied_boost=8.0,
        rank_anchors=2,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.flow.state_out.weight.grad is not None
    assert model.intent_model.history.spatial_in.weight.grad is None
    assert metrics["belief_dynamics/target_rms"] >= 0
    assert torch.isfinite(metrics["belief_dynamics/action_rank"])
    assert torch.isfinite(metrics["belief_dynamics/action_advantage"])
    sample = model.sample_next(batch, sample_intent=False)
    assert sample["state_tokens"].shape[-2] == teacher.n_tokens


def test_deploy_condition_keeps_latest_causal_history_row():
    batch = _batch(batch=2, time=4)
    teacher, ego_cfg, action_cfg, opponent_cfg, history_cfg, _ = _components()
    intent = OpponentIntentPriorModel(
        teacher,
        (H, W),
        ego_cfg=ego_cfg,
        self_action_cfg=action_cfg,
        opponent_cfg=opponent_cfg,
        history_cfg=history_cfg,
        intent_cfg=IntentPriorConfig(
            d_model=32, depth=1, n_heads=4, n_modes=3, contrast_dim=8
        ),
    )
    model = BeliefDynamicsModel(
        intent,
        BeliefDynamicsConfig(d_model=32, depth=1, n_heads=4, sample_steps=1),
    ).freeze_conditioner()
    training = model.encode_condition(batch, sample_intent=False)
    deploy = model.encode_deploy_condition(batch, sample_intent=False)
    assert training["history_registers"].shape[1] == 2
    assert deploy["history_registers"].shape[1] == 4
    sample = model.sample_deploy_next(batch, sample_intent=False)
    assert sample["state_tokens"].shape[:2] == (2, 4)


def test_online_fog_projection_uses_only_ego_sight_disks():
    obs = _batch(batch=1, time=1)["local_obs"][:, 0]
    # Own worker at (0, 0), enemy worker well outside radius three.
    obs[:, :, 0, 0] = 0
    obs[:, 0, 0, 0] = obs[:, 5, 0, 0] = 1
    obs[:, 11, 0, 0] = obs[:, 17, 0, 0] = obs[:, 21, 0, 0] = 1
    obs[:, :, 3, 3] = 0
    obs[:, 0, 3, 3] = obs[:, 5, 3, 3] = 1
    obs[:, 12, 3, 3] = obs[:, 17, 3, 3] = obs[:, 21, 3, 3] = 1
    local, visible = IncompleteBeliefDreamer.project_fog(obs)
    assert visible[0, 0, 0, 0]
    assert not visible[0, 0, 3, 3]
    assert local[0, 10, 3, 3] == 1
    assert local[0, 12, 3, 3] == 0


def test_explicit_action_residual_is_zero_initialized_and_safe_without_actions():
    torch.manual_seed(12)
    batch = _batch(batch=2, time=4)
    teacher, ego_cfg, action_cfg, opponent_cfg, history_cfg, _ = _components()
    intent = OpponentIntentPriorModel(
        teacher,
        (H, W),
        ego_cfg=ego_cfg,
        self_action_cfg=action_cfg,
        opponent_cfg=opponent_cfg,
        history_cfg=history_cfg,
        intent_cfg=IntentPriorConfig(
            d_model=32, depth=1, n_heads=4, n_modes=3, contrast_dim=8
        ),
    )
    model = BeliefDynamicsModel(
        intent,
        BeliefDynamicsConfig(
            d_model=32,
            depth=1,
            n_heads=4,
            sample_steps=1,
            explicit_action_residual=True,
        ),
    ).freeze_conditioner()
    condition = model.encode_condition(batch, sample_intent=False)
    anchors = condition["history_registers"].shape[1]
    noise = torch.randn(2, anchors, teacher.n_tokens, teacher.d_latent)
    correction = model.flow.action_residual(
        noise,
        condition["action_tokens"],
        condition["action_valid"],
    )
    assert torch.isfinite(correction).all()
    assert torch.count_nonzero(correction) == 0


def test_anchor_residual_flow_trains_on_valid_counterfactual_pairs():
    torch.manual_seed(13)
    batch = _batch(batch=2, time=4)
    batch["local_obs"][:, :, :, 0, 0] = 0
    for channel in (0, 5, 11, 17, 21):
        batch["local_obs"][:, :, channel, 0, 0] = 1
    batch["counterfactual_action"] = batch["action"].clone()
    batch["counterfactual_action"][:, :, 0, 0] = 1
    batch["counterfactual_next_state"] = batch["next_state"].clone()
    batch["counterfactual_next_globals"] = batch["next_globals"].clone()
    batch["counterfactual_valid"] = torch.ones(2, 4, dtype=torch.bool)
    teacher, ego_cfg, action_cfg, opponent_cfg, history_cfg, _ = _components()
    intent = OpponentIntentPriorModel(
        teacher,
        (H, W),
        ego_cfg=ego_cfg,
        self_action_cfg=action_cfg,
        opponent_cfg=opponent_cfg,
        history_cfg=history_cfg,
        intent_cfg=IntentPriorConfig(
            d_model=32, depth=1, n_heads=4, n_modes=3, contrast_dim=8
        ),
    )
    model = BeliefDynamicsModel(
        intent,
        BeliefDynamicsConfig(
            d_model=32,
            depth=1,
            n_heads=4,
            sample_steps=2,
            current_belief_anchor=True,
            axial_state_position=True,
            direct_action_attention=True,
        ),
    ).freeze_conditioner()
    loss, metrics = belief_dynamics_loss(
        model,
        batch,
        prior_coef=0.2,
        grounding_coef=0.0,
        history_rank_coef=0.0,
        intent_rank_coef=0.0,
        anchor_coef=1.0,
        counterfactual_coef=1.0,
        counterfactual_effect_coef=1.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.flow.anchor_out[-1].weight.grad is not None
    assert model.flow.direct_action_gate.grad is not None
    assert torch.isfinite(metrics["belief_dynamics/current_anchor"])
    assert torch.isfinite(metrics["belief_dynamics/counterfactual"])
    sample = model.sample_next(batch, sample_intent=False)
    assert sample["state_tokens"].shape[-2:] == (
        teacher.n_tokens,
        teacher.d_latent,
    )


def test_incomplete_dataset_toggle_and_episode_bounded_windows(tmp_path):
    path = tmp_path / "fog.h5"
    rows = 6
    obs = np.zeros((rows, 27, H, W), np.uint8)
    obs[:, [0, 5, 10, 13, 21]] = 1
    state = np.zeros((rows, H * W, 16), np.int16)
    action = np.zeros((rows, H * W, 8), np.uint8)
    with h5py.File(path, "w") as file:
        file.attrs["format_version"] = 4
        file.attrs["grid_hw"] = (H, W)
        file.attrs["action_nvec"] = (H * W, 6, 4, 4, 4, 4, 7, 49)
        file.attrs["has_ego_obs"] = True
        file.attrs["fog_augmentation_complete"] = True
        file.create_dataset("obs", data=obs, chunks=(1, 27, H, W))
        file.create_dataset("ego_obs", data=obs)
        file.create_dataset("ego_visibility", data=np.ones((rows, 1, H, W), np.uint8))
        file.create_dataset("state", data=state)
        file.create_dataset("next_state", data=state)
        file.create_dataset("globals", data=np.zeros((rows, 8), np.int16))
        file.create_dataset("next_globals", data=np.zeros((rows, 8), np.int16))
        file.create_dataset("action", data=action)
        file.create_dataset("opponent_action", data=action)
        file.create_dataset("reward", data=np.zeros(rows, np.float32))
        file.create_dataset("done", data=np.zeros(rows, np.uint8))
        file.create_dataset("is_first", data=np.array([1, 0, 0, 1, 0, 0], np.uint8))
        traj = file.create_group("traj")
        traj.create_dataset("start", data=np.array([0]))
        traj.create_dataset("length", data=np.array([rows]))
        for name in ("map_id", "opponent_id", "policy_id"):
            traj.create_dataset(name, data=np.array([0]))

    ego = MRTSSequenceDataset(
        path, seq_len=3, task="incomplete_dynamics", observation_mode="ego"
    )
    assert len(ego) == 2
    assert set(("local_obs", "local_visibility", "state", "action")) <= set(ego[0])
    oracle = MRTSSequenceDataset(
        path, seq_len=3, task="incomplete_dynamics", observation_mode="oracle_full"
    )
    assert oracle[0]["local_visibility"].all()
    np.testing.assert_array_equal(oracle[0]["local_obs"].numpy(), obs[:3])
