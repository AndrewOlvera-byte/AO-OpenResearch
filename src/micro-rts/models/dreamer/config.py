"""``DreamerV4Config`` — declarative spec for assembling a DreamerV4 agent.

Groups the hyperparameters of the three trainable subsystems Dreamer 4 keeps as
*separate* optimization problems (Hafner et al. 2025, *Training Agents Inside of
Scalable World Models*):

- **tokenizer** — a causal grid autoencoder mapping a MicroRTS obs grid to a small
  set of continuous latent tokens and back, reusing the GridNet CNN trunk. Also
  carries the predicted action-mask head used inside imagination.
- **dynamics** — the shortcut-forcing world model: a block-causal transformer that
  *denoises* per-frame noised latents (diffusion forcing) conditioned on signal /
  step tokens and CNN-GridNet action tokens, with x-prediction, plus reward /
  continue heads on the clean pass.
- **actor_critic** ("action expert") — reads a latent world state and emits a
  masked per-cell GridNet action (GridNet decoder + action head) and a symlog
  two-hot value with a slow EMA target. Trained purely on imagined rollouts.

``freeze`` supports pretrain-then-RL splits: any frozen module is excluded from
``build_optimizers`` (its optimizer key disappears when its whole group is frozen)
and gets ``requires_grad_(False)``. E.g. pretrain the world model with
``freeze: {actor: true, critic: true}``, then RL with
``freeze: {tokenizer: true, world_model: true}``.

Instantiate from a plain dict (``DreamerV4Config.from_dict(cfg.model)``) so the
YAML config system drives it like every other model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any, Tuple


@dataclass
class TokenizerConfig:
    enc_channels: int = 64      # conv width of the encoder/decoder trunk
    d_latent: int = 32          # per-cell latent (bottleneck) token width
    downsample: int = 4         # spatial reduction H -> H/downsample (two stride-2 convs)
    tanh_bottleneck: bool = True  # squash latents to [-1, 1] like Dreamer 4


@dataclass
class DynamicsConfig:
    d_model: int = 256          # transformer width
    depth: int = 4              # number of block-causal layers
    n_heads: int = 8
    n_register: int = 4         # learned register tokens (reward/continue readout)
    mlp_ratio: float = 4.0
    time_every: int = 2         # apply temporal (causal) attention every N layers
    dropout: float = 0.0
    space_mode: str = "wm_agent"  # full within-frame token mixing
    scale_pos_embeds: bool = True
    k_max: int = 16             # max shortcut step count (power of two)
    reward_bins: int = 255      # symlog two-hot reward head (DreamerV3 stabilizer)
    action_emb: int = 32        # per-component action embedding width (pre-CNN)
    action_channels: int = 64   # GridNet conv width of the action-token encoder
    # Joint-action training (v3): probability of replacing a frame's opponent
    # action with the learned unknown_opp embedding, so the same model serves
    # both the conditional (opponent given) and the marginal (opponent unknown,
    # e.g. imagination without an opponent policy) — and stays fog-of-war ready.
    opp_dropout: float = 0.15
    # Cell-weighted flow loss (v4): MicroRTS boards are ~99% empty, so a plain
    # per-cell mean dilutes the action-caused signal into the static background
    # (see loss.dreamer.cell_weights). All 1.0 == old uniform behavior (v3 and
    # earlier configs that don't set these fields are unaffected).
    cell_occ_boost: float = 1.0      # weight for occupied (unit-present) cells
    cell_changed_boost: float = 1.0  # weight for cells whose state changed vs t-1
    cell_weight_floor: float = 1.0   # weight for static empty-background cells
    # v4.2: zero (NOOP) the stored self-action at cells that cannot issue one
    # (no idle own unit) before it reaches the action encoder — the collector
    # stored raw GridNet output over all cells, ~97% of which the engine never
    # executed (see loss.dreamer.mask_actions_to_sources). False preserves the
    # v3/v4/v4.1 behavior for reproducibility.
    mask_junk_actions: bool = False
    # v4.3 action injection path. "tokens" (default): actions enter ONLY as
    # separate attention tokens — Dreamer 4's scheme, built for global actions
    # (Minecraft camera/buttons); per-cell effects must be recovered through
    # attention, a long gradient path the model never used (cell-conditioned
    # CF gap +3.7% after all v4/v4.1/v4.2 fixes). "add": ADDITIONALLY project
    # the action encoder's spatially-aligned conv features onto the matching
    # spatial latent tokens (zero-init projection, MuZero-style action planes)
    # — a direct per-cell route from action to next state; the action tokens
    # stay in the sequence for global coordination.
    action_inject: str = "tokens"
    # v4.4: opponent-policy head — per-cell BC head on the transformer trunk
    # predicting the opponent's action AT each frame from the engine-executed
    # labels (privileged: available in training, never needed at deployment).
    # Auxiliary supervision that forces the trunk to encode opponent intent,
    # and the sampler `imagine` uses to play an explicit opponent inside
    # dreams instead of falling back to the unknown_opp marginal (MBOM
    # level-0; see models/dreamer/world_model.py OpponentPolicyHead).
    opp_head: bool = False
    opp_head_channels: int = 64  # deconv width of the head (mirrors mask decoder)


@dataclass
class ActorCriticConfig:
    dec_channels: int = 64      # action-expert decoder width (own decoder, not shared)
    critic_hidden: Tuple[int, ...] = (256,)
    imagine_horizon: int = 15   # imagined rollout length for policy improvement
    imagine_context: int = 8    # replay frames used as world-model context when seeding
    imagine_flow_steps: int = 4  # shortcut denoise steps per imagined frame (power of 2)
    gamma: float = 0.99
    lam: float = 0.95           # lambda-return mixing
    entropy_coef: float = 3e-3
    critic_ema: float = 0.98    # slow target-critic EMA decay
    critic_bins: int = 255      # symlog two-hot critic (matches reward head)
    # DreamerV3 percentile return normalization: advantages are divided by
    # max(limit, EMA(P[high] - P[low])) of the lambda-return batch.
    return_norm: bool = True
    return_norm_rate: float = 0.01
    return_norm_limit: float = 1.0
    return_norm_low: float = 5.0
    return_norm_high: float = 95.0


@dataclass
class FreezeConfig:
    """Per-module freeze switches for pretrain -> RL staging."""
    tokenizer: bool = False
    world_model: bool = False
    actor: bool = False
    critic: bool = False


@dataclass
class DreamerV4Config:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    freeze: FreezeConfig = field(default_factory=FreezeConfig)

    # Separate learning rates — one per optimizer (world / actor / critic).
    world_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4
    grad_clip: float = 1000.0
    # Detach tokenizer latents entering the world model (Dreamer 4 trains the
    # tokenizer as its own phase; the encoder is shaped by reconstruction only).
    detach_latents: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "DreamerV4Config":
        """Build from a (possibly partial) nested dict, ignoring unknown keys.

        Mirrors how the YAML ``model:`` block is passed around: nested sub-dicts
        (``tokenizer``/``dynamics``/``actor_critic``/``freeze``) update the
        corresponding sub-config, flat scalars set the top-level fields. ``type``
        and any other stray keys are ignored so the same dict can carry a
        registry tag.
        """
        d = dict(d or {})
        sub = {
            "tokenizer": TokenizerConfig,
            "dynamics": DynamicsConfig,
            "actor_critic": ActorCriticConfig,
            "freeze": FreezeConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sub.items():
            if name in d and isinstance(d[name], dict):
                valid = {f.name for f in fields(klass)}
                kwargs[name] = klass(**{k: v for k, v in d[name].items() if k in valid})
        top = {f.name for f in fields(cls)} - set(sub)
        for k, v in d.items():
            if k in top:
                kwargs[k] = v
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
