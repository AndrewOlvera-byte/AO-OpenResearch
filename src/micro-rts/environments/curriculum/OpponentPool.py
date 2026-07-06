"""``OpponentPool`` — a FIFO of frozen past-policy snapshots for self-play.

Holds CPU clones of the learner's ``state_dict`` captured periodically. Sampling is
*latest-biased* (newer snapshots more likely) which empirically gives a stronger,
more stable self-play opponent distribution than uniform sampling while still
retaining older checkpoints to guard against strategy collapse / cycling.
"""

from __future__ import annotations

import random
from collections import deque

import torch


class OpponentPool:
    def __init__(self, capacity: int = 8, recency_bias: float = 2.0) -> None:
        self.capacity = capacity
        self.recency_bias = recency_bias
        self._snapshots: deque[dict] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._snapshots)

    def push(self, state_dict: dict) -> None:
        """Store a detached CPU clone so the snapshot is frozen in time."""
        clone = {k: v.detach().to("cpu").clone() for k, v in state_dict.items()}
        self._snapshots.append(clone)

    def sample(self) -> dict | None:
        """Sample a snapshot with newer entries weighted more heavily."""
        n = len(self._snapshots)
        if n == 0:
            return None
        # weights grow geometrically toward the most recent snapshot.
        weights = [self.recency_bias ** i for i in range(n)]
        return random.choices(list(self._snapshots), weights=weights, k=1)[0]
