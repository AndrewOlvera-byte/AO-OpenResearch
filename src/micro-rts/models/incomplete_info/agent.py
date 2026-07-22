"""Frozen v2 belief world model with Dreamer/PMPO agent heads.

The actor never consumes privileged state.  Online observations are projected to
canonical fog from the ego units visible in the raster, accumulated into the same
history format used during pretraining, and mapped to structured belief tokens.
Privileged structured state remains available to the trainer only for phase-2
reward/continue/behaviour-head supervision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import contextlib
from pathlib import Path

import torch
import torch.nn as nn
from tensordict import TensorDict

from models.dreamer_v2.agent import (
    StructuredActionExpert,
    StructuredActorCriticConfig,
    StructuredScalarHeads,
)
from models.dreamer_v2.config import StructuredTokenizerConfig
from models.dreamer_v2.temporal_jepa import structured_tokenizer_state_dict
from models.dreamer_v2.tokenizer import StructuredTokenizer

from .config import (
    BeliefDynamicsConfig,
    EgoTokenizerConfig,
    HistoryConfig,
    IntentPriorConfig,
    OpponentPlanTokenizerConfig,
    SelfActionTokenizerConfig,
)
from .model import BeliefDynamicsModel, OpponentIntentPriorModel


def _resolve(path: str | Path, root: str | Path | None = None) -> Path:
    value = Path(path)
    if value.is_file():
        return value
    if root is not None and (Path(root) / value).is_file():
        return Path(root) / value
    raise FileNotFoundError(f"incomplete-belief checkpoint not found: {path}")


def _load_stage(module, path, root=None, prefixes=("",)):
    checkpoint = torch.load(_resolve(path, root), map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    for prefix in prefixes:
        candidate = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not candidate:
            continue
        missing, unexpected = module.load_state_dict(candidate, strict=False)
        if not unexpected and len(missing) < len(module.state_dict()):
            return checkpoint
    raise ValueError(f"{path}: no compatible weights for {type(module).__name__}")


def load_belief_dynamics(checkpoint_path, device="cpu", root=None):
    """Reconstruct the trainable-only v2 checkpoint and all frozen stages."""
    path = _resolve(checkpoint_path, root)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    root = Path(root or path.parents[2] if len(path.parents) > 2 else Path.cwd())
    intent_path = _resolve(checkpoint["intent_prior_ckpt"], root)
    intent_checkpoint = torch.load(intent_path, map_location="cpu", weights_only=False)
    grid_hw = tuple(checkpoint.get("grid_hw", intent_checkpoint.get("grid_hw", (16, 16))))

    tokenizer_path = _resolve(
        checkpoint.get(
            "full_state_tokenizer_ckpt",
            intent_checkpoint["full_state_tokenizer_ckpt"],
        ),
        root,
    )
    tokenizer_checkpoint = torch.load(
        tokenizer_path, map_location="cpu", weights_only=False
    )
    tokenizer_cfg = StructuredTokenizerConfig.from_dict(
        checkpoint.get("tokenizer_cfg", tokenizer_checkpoint["tokenizer_cfg"])
    )
    tokenizer = StructuredTokenizer(grid_hw, tokenizer_cfg)
    tokenizer.load_state_dict(structured_tokenizer_state_dict(tokenizer_checkpoint))
    tokenizer.requires_grad_(False).eval()

    ego_cfg = EgoTokenizerConfig.from_dict(intent_checkpoint["ego_tokenizer_cfg"])
    action_cfg = SelfActionTokenizerConfig.from_dict(
        intent_checkpoint["self_action_tokenizer_cfg"]
    )
    opponent_cfg = OpponentPlanTokenizerConfig.from_dict(
        intent_checkpoint["opponent_tokenizer_cfg"]
    )
    history_cfg = HistoryConfig.from_dict(intent_checkpoint["history_cfg"])
    intent_cfg = IntentPriorConfig.from_dict(intent_checkpoint["intent_prior_cfg"])
    intent = OpponentIntentPriorModel(
        tokenizer,
        grid_hw,
        ego_cfg=ego_cfg,
        self_action_cfg=action_cfg,
        opponent_cfg=opponent_cfg,
        history_cfg=history_cfg,
        intent_cfg=intent_cfg,
    )
    _load_stage(
        intent.ego_tokenizer,
        intent_checkpoint["ego_tokenizer_ckpt"],
        root,
        prefixes=("tokenizer.", ""),
    )
    _load_stage(
        intent.self_action_tokenizer,
        intent_checkpoint["self_action_tokenizer_ckpt"],
        root,
    )
    _load_stage(
        intent.opponent_tokenizer,
        intent_checkpoint["opponent_tokenizer_ckpt"],
        root,
    )
    _load_stage(intent, intent_path, root)

    flow_cfg = BeliefDynamicsConfig.from_dict(checkpoint["belief_dynamics_cfg"])
    model = BeliefDynamicsModel(intent, flow_cfg)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed = {f"intent_model.{name}" for name in intent.state_dict()}
    bad_missing = [name for name in missing if name not in allowed]
    if bad_missing or unexpected:
        raise ValueError(
            f"{path}: incompatible belief checkpoint; "
            f"missing={bad_missing[:8]} unexpected={list(unexpected)[:8]}"
        )
    return model.to(device).freeze_conditioner().eval(), checkpoint


@dataclass
class IncompleteBeliefAgentConfig:
    belief_checkpoint: str = "checkpoints/pretrain_belief_dynamics_medium_v2/best.pt"
    checkpoint_root: str = "."
    actor_critic: StructuredActorCriticConfig = field(
        default_factory=StructuredActorCriticConfig
    )
    agent_lr: float = 1.0e-4
    actor_lr: float = 3.0e-5
    critic_lr: float = 1.0e-4
    grad_clip: float = 10.0
    sample_intent: bool = True
    flow_steps: int = 1

    @classmethod
    def from_dict(cls, values):
        values = dict(values or {})
        kwargs = {k: v for k, v in values.items() if k in {f.name for f in fields(cls)}}
        ac = kwargs.get("actor_critic")
        if isinstance(ac, dict):
            valid = {f.name for f in fields(StructuredActorCriticConfig)}
            item = {k: v for k, v in ac.items() if k in valid}
            if isinstance(item.get("critic_hidden"), list):
                item["critic_hidden"] = tuple(item["critic_hidden"])
            kwargs["actor_critic"] = StructuredActorCriticConfig(**item)
        return cls(**kwargs)

    def to_dict(self):
        return asdict(self)


class IncompleteBeliefDreamer(nn.Module):
    """Deployable fog-history policy and short-horizon v2 imagination adapter."""

    uses_history_state = True

    def __init__(self, action_nvec, cfg: IncompleteBeliefAgentConfig, device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.belief_model, self.source_checkpoint = load_belief_dynamics(
            cfg.belief_checkpoint, self.device, cfg.checkpoint_root
        )
        self.tokenizer = self.belief_model.full_state_tokenizer
        self.action_nvec = torch.as_tensor(action_nvec, dtype=torch.long)
        self.grid_cells = int(self.action_nvec[0])
        self.n_components = len(self.action_nvec) - 1
        self.action_expert = StructuredActionExpert(
            self.tokenizer, self.action_nvec, cfg.actor_critic
        )
        self.scalar_heads = StructuredScalarHeads(self.tokenizer, cfg.actor_critic)
        self.collect_with_prior = True
        self._online_obs = self._online_action = self._online_first = None
        self.belief_model.requires_grad_(False).eval()
        self.to(self.device)

    @staticmethod
    def visibility_from_observation(obs: torch.Tensor) -> torch.Tensor:
        """Canonical sight disks using only visible ego-unit raster channels."""
        h, w = obs.shape[-2:]
        own = obs[..., 11, :, :] > 0.5
        unit_type = obs[..., 13:21, :, :].argmax(-3) - 1
        radii = (0, 5, 3, 3, 2, 2, 3)
        visible = torch.zeros_like(own)
        for type_id, radius in enumerate(radii):
            source = own & (unit_type == type_id)
            if not source.any():
                continue
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    y0, y1 = max(0, -dy), min(h, h - dy)
                    x0, x1 = max(0, -dx), min(w, w - dx)
                    visible[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx] |= \
                        source[..., y0:y1, x0:x1]
        return visible[..., None, :, :]

    @classmethod
    def project_fog(cls, obs: torch.Tensor):
        visibility = cls.visibility_from_observation(obs)
        empty = torch.zeros(obs.shape[-3], device=obs.device, dtype=obs.dtype)
        empty[[0, 5, 10, 13, 21]] = 1
        shape = (*((1,) * (obs.ndim - 3)), obs.shape[-3], 1, 1)
        local = torch.where(visibility, obs, empty.reshape(shape))
        return local, visibility

    def _empty_action(self, *lead, device=None):
        return torch.zeros(
            *lead, self.grid_cells, self.n_components,
            dtype=torch.long, device=device or self.device,
        )

    def _history_batch(self, local_obs, visibility, action, is_first):
        return {
            "local_obs": local_obs,
            "local_visibility": visibility,
            "action": action,
            "is_first": is_first,
        }

    @torch.no_grad()
    def infer_next(self, local_obs, visibility, action, is_first):
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with amp:
            out = self.belief_model.sample_deploy_next(
                self._history_batch(local_obs, visibility, action, is_first),
                sample_intent=self.cfg.sample_intent,
                steps=self.cfg.flow_steps,
            )
        # Agent heads run outside this local autocast during real collection.
        out["state_tokens"] = out["state_tokens"].float()
        return out["state_tokens"][:, -1], out

    def _append_online(self, obs, is_first):
        local, visibility = self.project_fog(obs)
        action = self._empty_action(obs.shape[0], 1, device=obs.device)
        local, visibility = local[:, None], visibility[:, None]
        first = is_first.bool()[:, None]
        if self._online_obs is None or self._online_obs.shape[0] != obs.shape[0]:
            self._online_obs, self._online_action, self._online_first = local, action, first
        else:
            self._online_obs = torch.cat((self._online_obs, local), 1)
            self._online_action = torch.cat((self._online_action, action), 1)
            self._online_first = torch.cat((self._online_first, first), 1)
        limit = self.belief_model.intent_model.history.cfg.context_length
        self._online_obs = self._online_obs[:, -limit:]
        self._online_action = self._online_action[:, -limit:]
        self._online_first = self._online_first[:, -limit:]
        return visibility

    @torch.no_grad()
    def step(self, obs, mask=None, deterministic=False, *, is_first=None):
        obs = obs.to(self.device)
        if is_first is None:
            is_first = torch.zeros(obs.shape[0], dtype=torch.bool, device=self.device)
            if self._online_obs is None:
                is_first.fill_(True)
        else:
            is_first = is_first.to(self.device).bool()
        self._append_online(obs, is_first)
        # V2 has no current-belief posterior.  A no-op query is the documented
        # departure-state approximation; candidate actions are used in dreams.
        belief, condition = self.infer_next(
            self._online_obs,
            self.visibility_from_observation(self._online_obs),
            self._online_action,
            self._online_first,
        )
        use_mask = mask.to(self.device) if mask is not None else (
            torch.sigmoid(self.tokenizer.decode_mask(belief)) > 0.5
        ).float()
        policy = (
            self.action_expert.behavior_prior
            if self.collect_with_prior else self.action_expert.actor_policy
        )
        action, logprob, entropy = policy.sample(belief, use_mask, deterministic)
        self._online_action[:, -1] = action
        probs = condition["mode_probabilities"][:, -1]
        return TensorDict(
            {
                "action": action,
                "logprob": logprob,
                "entropy": entropy,
                "value": self.action_expert.value(belief),
                "z": belief,
                "intent_entropy": -(probs * probs.clamp_min(1e-8).log()).sum(-1),
            },
            batch_size=[obs.shape[0]],
        )

    def role_bc_inputs(self, state, action, role: int):
        source = state[..., 1].bool() & (state[..., 3] == role) & ~state[..., 7].bool()
        safe = torch.zeros_like(action)
        safe[..., 0] = torch.where(source, action[..., 0], 0)
        width = 1 + sum(self.action_expert.opponent_policy.dist.comp_splits)
        mask = action.new_zeros((*action.shape[:-1], width), dtype=torch.bool)
        mask[..., 0] = source
        active_types = (None, 1, 2, 3, 4, 4, 5)
        offset = 1
        for component, (size, active_type) in enumerate(zip(
            self.action_expert.opponent_policy.dist.comp_splits, active_types
        )):
            active = source if active_type is None else source & (action[..., 0] == active_type)
            mask[..., offset:offset + size] = active[..., None]
            if component:
                safe[..., component] = torch.where(active, action[..., component], 0)
            offset += size
        return safe, mask.float()

    def prepare_context(self, batch):
        obs = batch["obs"].to(self.device)
        local, visibility = self.project_fog(obs)
        return {
            "local_obs": local,
            "local_visibility": visibility,
            "action": batch["action"].to(self.device).long().clone(),
            "is_first": batch["is_first"].to(self.device).bool(),
            "mask": batch["mask"].to(self.device).float(),
        }

    @torch.no_grad()
    def imagine_from_batch(self, batch, horizon=None):
        horizon = int(horizon or self.cfg.actor_critic.imagine_horizon)
        ctx = self.prepare_context(batch)
        # The recorded final action must not leak into the departure belief.
        ctx["action"][:, -1] = 0
        z, _ = self.infer_next(
            ctx["local_obs"], ctx["local_visibility"], ctx["action"], ctx["is_first"]
        )
        z_seq, actions, masks, rewards, conts = [z], [], [], [], []
        for step in range(horizon):
            mask = ctx["mask"][:, -1] if step == 0 else (
                torch.sigmoid(self.tokenizer.decode_mask(z)) > 0.5
            ).float()
            action, _, _ = self.action_expert.actor_policy.sample(z, mask)
            ctx["action"][:, -1] = action
            z_next, _ = self.infer_next(
                ctx["local_obs"], ctx["local_visibility"],
                ctx["action"], ctx["is_first"],
            )
            reward_logits, continue_logit = self.scalar_heads(z_next)
            decoded = self.tokenizer.decode(z_next)
            raster = (decoded["legacy_obs"] >= 0).to(ctx["local_obs"].dtype)
            h, w = self.tokenizer.h, self.tokenizer.w
            raster = raster.reshape(raster.shape[0], h, w, -1).permute(0, 3, 1, 2)
            local, visibility = self.project_fog(raster)
            ctx["local_obs"] = torch.cat((ctx["local_obs"], local[:, None]), 1)
            ctx["local_visibility"] = torch.cat(
                (ctx["local_visibility"], visibility[:, None]), 1
            )
            ctx["action"] = torch.cat(
                (ctx["action"], self._empty_action(z.shape[0], 1)), 1
            )
            ctx["is_first"] = torch.cat(
                (ctx["is_first"], torch.zeros(z.shape[0], 1, dtype=torch.bool, device=z.device)), 1
            )
            limit = self.belief_model.intent_model.history.cfg.context_length
            for name in ("local_obs", "local_visibility", "action", "is_first"):
                ctx[name] = ctx[name][:, -limit:]
            z_seq.append(z_next)
            actions.append(action)
            masks.append(mask)
            rewards.append(self.scalar_heads.reward_coder.mean(reward_logits))
            conts.append(torch.sigmoid(continue_logit))
            z = z_next
        return {
            "z": torch.stack(z_seq, 1),
            "action": torch.stack(actions, 1),
            "mask": torch.stack(masks, 1),
            "reward": torch.stack(rewards, 1),
            "cont": torch.stack(conts, 1),
        }

    def build_optimizers(self):
        groups = {
            "agent": ([p for module in (
                self.scalar_heads,
                self.action_expert.behavior_prior,
                self.action_expert.opponent_policy,
            ) for p in module.parameters()], self.cfg.agent_lr),
            "actor": (list(self.action_expert.actor_policy.parameters()), self.cfg.actor_lr),
            "critic": ([p for module in (
                self.action_expert.value_summary, self.action_expert.critic,
            ) for p in module.parameters()], self.cfg.critic_lr),
        }
        return {
            name: torch.optim.AdamW(params, lr=lr, eps=1e-5, weight_decay=1e-4)
            for name, (params, lr) in groups.items()
        }
