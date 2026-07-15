"""Canonical v2 engine fields and dense GridNet -> sparse action events."""

from __future__ import annotations

import torch

CELL_FIELDS = (
    "terrain",
    "present",
    "unit_id",
    "owner",
    "unit_type",
    "hp",
    "carried",
    "assignment",
    "action_type",
    "direction",
    "target_x",
    "target_y",
    "produced_type",
    "start_tick",
    "eta",
    "remaining",
)
GLOBAL_FIELDS = (
    "tick",
    "self_resources",
    "opponent_resources",
    "self_reserved",
    "opponent_reserved",
    "reserved_positions",
    "winner",
    "gameover",
)
STATE_WIDTH = len(CELL_FIELDS)
GLOBAL_WIDTH = len(GLOBAL_FIELDS)

# role, unit_id, src_x, src_y, type, target_x, target_y, direction,
# produced_type, attack_offset
ACTION_EVENT_WIDTH = 10
# Stored only in the otherwise inactive attack-offset component when action
# type is TYPE_NONE. 255 cannot be emitted by a normal 7x7 GridNet component.
EXPLICIT_NONE_MARKER = 255


def validate_structured_state(
    state: torch.Tensor, globals_: torch.Tensor, grid_hw: tuple[int, int]
) -> None:
    """Raise on schema/range errors that would make tokenization ambiguous."""
    h, w = grid_hw
    if state.shape[-2:] != (h * w, STATE_WIDTH):
        raise ValueError(
            f"state tail {tuple(state.shape[-2:])} != {(h * w, STATE_WIDTH)}"
        )
    if globals_.shape[-1] != GLOBAL_WIDTH:
        raise ValueError(f"globals width {globals_.shape[-1]} != {GLOBAL_WIDTH}")
    present = state[..., 1]
    if not torch.all((present == 0) | (present == 1)):
        raise ValueError("present must be binary")
    owner = state[..., 3]
    if not torch.all((owner >= -1) & (owner <= 2)):
        raise ValueError("owner outside [-1,2]")
    assigned = state[..., 7]
    if not torch.all((assigned == 0) | (assigned == 1)):
        raise ValueError("assignment must be binary")
    tx, ty = state[..., 10], state[..., 11]
    valid_x = (tx == -1) | ((tx >= 0) & (tx < w))
    valid_y = (ty == -1) | ((ty >= 0) & (ty < h))
    if not torch.all(valid_x & valid_y):
        raise ValueError("assignment target outside board")


def _events_for_role(state, action, role, h, w, attack_diameter):
    """Build dense candidate records for one role using tensor operations only."""
    batch, cells = state.shape[:2]
    atype = action[..., 0]
    explicit_none = (atype == 0) & (action[..., 6] == EXPLICIT_NONE_MARKER)
    keep = (
        ((atype > 0) | explicit_none) & state[..., 1].bool() & (state[..., 3] == role)
    )

    cell = torch.arange(cells, device=state.device)
    sy = torch.div(cell, w, rounding_mode="floor").expand(batch, -1)
    sx = (cell % w).expand(batch, -1)
    tx, ty = sx.clone(), sy.clone()

    directional = (atype >= 1) & (atype <= 4)
    component = atype.clamp(0, action.shape[-1] - 1)
    direction_value = action.gather(-1, component[..., None]).squeeze(-1)
    direction = torch.where(directional, direction_value, -1)
    valid_direction = directional & (direction >= 0) & (direction < 4)
    dx_lut = action.new_tensor((0, 1, 0, -1))
    dy_lut = action.new_tensor((-1, 0, 1, 0))
    direction_idx = direction.clamp(0, 3)
    tx = torch.where(valid_direction, tx + dx_lut[direction_idx], tx)
    ty = torch.where(valid_direction, ty + dy_lut[direction_idx], ty)

    produced = torch.where(atype == 4, action[..., 5], -1)
    attacking = atype == 5
    attack_off = torch.where(attacking, action[..., 6], -1)
    half = attack_diameter // 2
    attack_idx = attack_off.clamp_min(0)
    attack_dy = torch.div(attack_idx, attack_diameter, rounding_mode="floor")
    attack_dx = attack_idx % attack_diameter
    tx = torch.where(attacking, sx + attack_dx - half, tx)
    ty = torch.where(attacking, sy + attack_dy - half, ty)

    role_field = torch.full_like(atype, role)
    records = torch.stack(
        (
            role_field,
            state[..., 2],
            sx,
            sy,
            atype,
            tx,
            ty,
            direction,
            produced,
            attack_off,
        ),
        dim=-1,
    )
    return records, keep


def dense_actions_to_events(
    state: torch.Tensor,
    self_action: torch.Tensor,
    opponent_action: torch.Tensor,
    grid_hw=(16, 16),
    max_events: int = 32,
    attack_diameter: int = 7,
):
    """Convert dense joint GridNet actions to fixed sparse event records.

    Leading dimensions of all inputs must match. Returns ``events`` with tail
    ``(max_events,10)``, a boolean validity mask, and per-item overflow counts.
    Actions at empty, wrong-role, or NOOP cells are not events.
    """
    h, w = map(int, grid_hw)
    lead = state.shape[:-2]
    flat_s = state.reshape(-1, h * w, STATE_WIDTH)
    flat_a = self_action.reshape(-1, h * w, self_action.shape[-1])
    flat_o = opponent_action.reshape(-1, h * w, opponent_action.shape[-1])
    self_events, self_valid = _events_for_role(flat_s, flat_a, 1, h, w, attack_diameter)
    opp_events, opp_valid = _events_for_role(flat_s, flat_o, 2, h, w, attack_diameter)
    candidates = torch.cat((self_events, opp_events), dim=1)
    candidate_valid = torch.cat((self_valid, opp_valid), dim=1)

    out = torch.zeros(
        flat_s.shape[0],
        max_events,
        ACTION_EVENT_WIDTH,
        dtype=torch.long,
        device=state.device,
    )
    valid = torch.zeros(
        flat_s.shape[0], max_events, dtype=torch.bool, device=state.device
    )
    # Stable cumsum gives every valid candidate its packed destination without
    # sorting or any device-to-host scalar reads.
    rank = candidate_valid.long().cumsum(dim=1) - 1
    selected = candidate_valid & (rank < max_events)
    batch_idx = torch.arange(flat_s.shape[0], device=state.device)[:, None]
    batch_idx = batch_idx.expand_as(candidate_valid)
    out[batch_idx[selected], rank[selected]] = candidates[selected]
    valid[batch_idx[selected], rank[selected]] = True
    count = candidate_valid.sum(dim=1)
    no_events = count == 0
    valid[no_events, 0] = True  # explicit NO_ISSUED_ACTIONS token
    overflow = (count - max_events).clamp_min(0)
    return (
        out.reshape(*lead, max_events, ACTION_EVENT_WIDTH),
        valid.reshape(*lead, max_events),
        overflow.reshape(*lead),
    )
