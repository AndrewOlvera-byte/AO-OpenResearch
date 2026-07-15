"""Player-1 policies for offline data collection.

MicroRTS' JNI only surfaces a *gridnet action* for the Python-controlled player;
the native scripted opponent (player 2, ``ai2``) never exposes the action it
chose. So the action we can actually record is always player 1's, produced by a
Python :class:`~environments.base.Policy`. Two are provided:

- :class:`MaskedRandomPolicy` — samples a **legal** per-cell gridnet action from
  the engine mask (uniform over each cell's valid components; empty/busy cells
  collapse to NOOP). No weights, no checkpoint — the zero-dependency default that
  still yields broad, legal state/action coverage for tokenizer + dynamics
  pretraining.
- :func:`load_policy` — resolves ``--policy`` on the CLI to either the masked
  random default or a trained checkpoint (a registry ``model`` type such as
  ``cnn_gridnet``), so strong demonstrations for actor distillation can be
  recorded by swapping in a trained net without touching the collector.

Both satisfy the ``Policy`` protocol (``step(obs, mask) -> {"action", ...}``) and
emit gridnet actions shaped ``(N, H*W, 7)`` — exactly what the world model's
action encoder and the future actor consume.

For the v3 corpus (NEXT_PLAN.md, Workstream C) there is additionally
:class:`EpsilonGreedyPolicy` — the identifiability workhorse: it wraps any base
policy and, per cell with probability ε, swaps that cell's action for a masked
random legal one. Coherent games whose actions are NOT a deterministic function
of the state are what make the action channel learnable for the world model.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from models.shared.GridNetHead import GridNetActionHead


class MaskedRandomPolicy:
    """Uniform-over-legal gridnet actions, driven purely by the engine mask.

    Reuses :class:`GridNetActionHead` with all-zero logits: a masked categorical
    over zero logits is uniform over the legal choices, and all-zero masks (cells
    with no actable unit) collapse to the degenerate NOOP the head already handles
    identically to the PPO path. This guarantees every recorded action is legal
    under the stored mask.
    """

    def __init__(self, action_nvec: torch.Tensor, device: str | torch.device = "cpu") -> None:
        if not torch.is_tensor(action_nvec):
            action_nvec = torch.as_tensor(list(action_nvec), dtype=torch.long)
        self.device = device
        self.grid_cells = int(action_nvec[0])          # H*W (source dim)
        self.cell_nvec = action_nvec[1:].to(device)    # per-cell [6,4,4,4,4,7,49]
        self.cell_out = int(self.cell_nvec.sum())      # 78
        self.head = GridNetActionHead(self.cell_nvec)

    @torch.no_grad()
    def step(self, obs: torch.Tensor, mask: torch.Tensor | None = None,
             deterministic: bool = False) -> TensorDict:
        n = obs.shape[0]
        assert mask is not None, "MaskedRandomPolicy needs the engine action mask"
        mask = mask.to(self.device)
        logits = torch.zeros(n, self.grid_cells, self.cell_out, device=self.device)
        action, logprob = self.head.sample(logits, mask, deterministic=deterministic)
        return TensorDict(
            {"action": action, "logprob": logprob, "value": torch.zeros(n, device=self.device)},
            batch_size=[n],
        )


class EpsilonGreedyPolicy:
    """Per-cell ε-mix of a base policy and masked-random legal actions.

    With probability ε *per cell* the base policy's action for that cell is
    replaced by a uniform sample over the cell's legal actions (from the engine
    mask). Cell granularity keeps games coherent — the policy still plays; a
    fraction of its unit orders are perturbed — while breaking the deterministic
    state->action mapping that made expert-only data unidentifiable.
    """

    def __init__(self, base, action_nvec, epsilon: float,
                 device: str | torch.device = "cpu") -> None:
        assert 0.0 <= float(epsilon) <= 1.0
        self.base = base
        self.epsilon = float(epsilon)
        self.random = MaskedRandomPolicy(action_nvec, device=device)

    @torch.no_grad()
    def step(self, obs: torch.Tensor, mask: torch.Tensor | None = None,
             deterministic: bool = False) -> TensorDict:
        out = self.base.step(obs, mask, deterministic=deterministic)
        if self.epsilon <= 0.0:
            return out
        rnd = self.random.step(obs, mask)["action"]
        action = out["action"].clone()
        swap = torch.rand(action.shape[:2], device=action.device) < self.epsilon
        action[swap] = rnd.to(action.device)[swap]
        out.set("action", action)
        return out


def load_policy(spec: str, obs_shape, action_nvec, device: str = "cpu"):
    """Resolve a ``--policy`` spec to a collection policy.

    - ``"masked_random"`` (default): :class:`MaskedRandomPolicy`.
    - a path to a ``.pt`` checkpoint: rebuild the registry ``model`` and load its
      weights (eval mode) — for recording strong, diverse demonstrations. Handles
      the PPO trainer's checkpoint format (weights under ``"model"``, the model
      block under ``config["model"]`` with its registry ``type`` + kwargs), the
      generic ``{"state_dict", "model_type", "model_kwargs"}`` form, and a bare
      ``state_dict``/``nn.Module`` (assumed ``cnn_gridnet``).
    """
    if spec in ("masked_random", "random", "", None):
        return MaskedRandomPolicy(action_nvec, device=device)

    import os

    from core.registry import build

    import importlib

    # The registry is populated as a side effect of importing model modules
    # (``@register`` decorators); the empty ``models`` package doesn't do it.
    for mod in ("gridnet_policy", "cnn_policy", "cnn_mlp_policy"):
        try:
            importlib.import_module(f"models.{mod}")
        except Exception:
            pass

    if not os.path.exists(spec):
        raise FileNotFoundError(f"policy checkpoint not found: {spec}")
    ckpt = torch.load(spec, map_location=device, weights_only=False)

    # Locate the weights (state_dict) and the model build spec (type + kwargs).
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict) \
            and "config" in ckpt:                       # PPO trainer format
        state_dict = ckpt["model"]
        model_cfg = dict(ckpt["config"].get("model", {}))
        model_type = model_cfg.pop("type", "cnn_gridnet")
        model_kwargs = model_cfg
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        model_type = ckpt.get("model_type", "cnn_gridnet")
        model_kwargs = ckpt.get("model_kwargs", {})
    else:                                               # bare state_dict / module
        state_dict = ckpt
        model_type, model_kwargs = "cnn_gridnet", {}

    model = build("model", type=model_type, obs_shape=obs_shape,
                  action_nvec=action_nvec, device=device, **model_kwargs)
    model.load_state_dict(state_dict)
    if hasattr(model, "eval"):
        model.eval()
    return model
