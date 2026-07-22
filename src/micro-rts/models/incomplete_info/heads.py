from __future__ import annotations

import torch
import torch.nn as nn


class EventFieldDecoder(nn.Module):
    def __init__(self, d_model: int, grid_hw=(16, 16), max_unit_types: int = 16):
        super().__init__()
        h, w = map(int, grid_hw)
        self.grid_hw = (h, w)
        self.max_unit_types = int(max_unit_types)
        self.heads = nn.ModuleDict(
            {
                "valid": nn.Linear(d_model, 1),
                "role": nn.Linear(d_model, 3),
                "action_type": nn.Linear(d_model, 6),
                "src_x": nn.Linear(d_model, w),
                "src_y": nn.Linear(d_model, h),
                "dst_x": nn.Linear(d_model, w + 2),
                "dst_y": nn.Linear(d_model, h + 2),
                "direction": nn.Linear(d_model, 6),
                "produced": nn.Linear(d_model, self.max_unit_types + 1),
                "attack": nn.Linear(d_model, 50),
            }
        )

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(tokens) for name, head in self.heads.items()}


def event_targets(events: torch.Tensor, decoder: EventFieldDecoder):
    role, _uid, sx, sy, typ, tx, ty, direction, produced, attack = events.unbind(-1)
    h, w = decoder.grid_hw
    return {
        "role": role.clamp(0, 2),
        "action_type": typ.clamp(0, 5),
        "src_x": sx.clamp(0, w - 1),
        "src_y": sy.clamp(0, h - 1),
        "dst_x": (tx + 1).clamp(0, w + 1),
        "dst_y": (ty + 1).clamp(0, h + 1),
        "direction": (direction + 1).clamp(0, 5),
        "produced": (produced + 1).clamp(0, decoder.max_unit_types),
        "attack": (attack + 1).clamp(0, 49),
    }
