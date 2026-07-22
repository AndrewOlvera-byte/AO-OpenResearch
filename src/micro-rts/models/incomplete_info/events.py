from __future__ import annotations

import torch

from models.dreamer_v2.schema import dense_actions_to_events


def _pseudo_state_from_local_obs(local_obs: torch.Tensor) -> torch.Tensor:
    """Build the only state fields needed to route ego-issued actions.

    Gym-MicroRTS owner planes are none/own/enemy at channels 10:13. Own units
    are always visible under the canonical fog projection. Unit ids are not used
    by the factorized event encoder.
    """
    h, w = local_obs.shape[-2:]
    lead = local_obs.shape[:-3]
    own = local_obs[..., 11, :, :] > 0.5
    state = torch.zeros(*lead, h * w, 16, dtype=torch.long, device=local_obs.device)
    state[..., 2] = -1
    state[..., 3] = -1
    flat = own.flatten(-2)
    state[..., 1] = flat.long()
    state[..., 3] = torch.where(flat, torch.ones_like(state[..., 3]), state[..., 3])
    return state


def self_action_events(
    local_obs: torch.Tensor,
    action: torch.Tensor,
    *,
    max_events: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = _pseudo_state_from_local_obs(local_obs)
    empty = torch.zeros_like(action)
    return dense_actions_to_events(
        state,
        action,
        empty,
        local_obs.shape[-2:],
        max_events,
    )


def opponent_action_events(
    state: torch.Tensor,
    opponent_action: torch.Tensor,
    *,
    grid_hw: tuple[int, int],
    max_events: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    empty = torch.zeros_like(opponent_action)
    return dense_actions_to_events(
        state,
        empty,
        opponent_action,
        grid_hw,
        max_events,
    )
