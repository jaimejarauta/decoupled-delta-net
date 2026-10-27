"""Bounded-memory online checkpoint scheduling for irreversible recurrent states.

Provides three policies under a uniform interface so baselines and proposed schedules
are evaluated with identical memory and token-replay accounting:
  - 'zero'         : C = 0, keeps no snapshots (always replays from token 0).
  - 'ring_fifo'    : keeps the most recent C snapshots (ReplaySSM-style sliding window).
  - 'min_span'     : keeps anchor slot 0 + greedy min-merged-span thinning across [t_0, T],
                     guaranteeing O(T / C) worst-case token replay to *any* past step a <= T,
                     including cascading multi-backtracks (b1 -> a1, then b2 -> a2 < a1).
  - 'dyadic_ruler' : online binary-ruler (2-adic valuation ctz(step)) hierarchical thinning
                     adapted from Stumm & Walther / Griewank Revolve online schedules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Policy = Literal["zero", "ring_fifo", "min_span", "dyadic_ruler"]


@dataclass
class LayerRecurrentState:
    layer_idx: int
    conv_states: dict[int, torch.Tensor | None]
    recurrent_states: dict[int, torch.Tensor | None]
    is_conv_init: dict[int, bool]
    is_rec_init: dict[int, bool]
    has_prev: dict[int, bool]


@dataclass
class RecurrentSnapshot:
    token_pos: int
    step_idx: int
    level: int
    layers: list[LayerRecurrentState]
    nbytes: int


def _ctz(k: int) -> int:
    """2-adic valuation (number of trailing zero bits), with k=0 given infinite priority."""
    if k <= 0:
        return 62
    return (k & -k).bit_length() - 1


class OnlineCheckpointPool:
    """Fixed-capacity pool of recurrent-state snapshots indexed by monotonically increasing token_pos."""

    def __init__(self, capacity: int = 8, policy: Policy = "min_span"):
        if capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}")
        if policy not in ("zero", "ring_fifo", "min_span", "dyadic_ruler"):
            raise ValueError(f"unknown policy: {policy}")
        self.capacity = 0 if policy == "zero" else capacity
        self.policy: Policy = policy
        self._slots: list[RecurrentSnapshot] = []
        self._pushes: int = 0

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def positions(self) -> list[int]:
        return [s.token_pos for s in self._slots]

    @property
    def total_nbytes(self) -> int:
        return sum(s.nbytes for s in self._slots)

    def clear(self) -> None:
        self._slots.clear()
        self._pushes = 0

    def push(self, snap: RecurrentSnapshot) -> None:
        """Insert a snapshot at `snap.token_pos` and evict at most one entry if over capacity."""
        if self.capacity == 0:
            return
        if self._slots and snap.token_pos <= self._slots[-1].token_pos:
            if snap.token_pos == self._slots[-1].token_pos:
                self._slots[-1] = snap
                return
            raise ValueError(f"token_pos must be non-decreasing ({snap.token_pos} < {self._slots[-1].token_pos})")

        snap.step_idx = self._pushes
        snap.level = _ctz(self._pushes)
        self._pushes += 1
        self._slots.append(snap)

        if len(self._slots) <= self.capacity:
            return

        if self.policy == "ring_fifo":
            self._slots.pop(0)
        elif self.policy == "min_span":
            if self.capacity == 1:
                # Keep anchor slot 0 so prompt prefill is never lost.
                self._slots.pop(-1)
            elif self.capacity == 2:
                # Keep anchor slot 0 and latest slot.
                self._slots.pop(1)
            else:
                # Evict interior index i in 1..len-2 minimizing newly created gap (pos[i+1] - pos[i-1]).
                best_i = min(
                    range(1, len(self._slots) - 1),
                    key=lambda i: self._slots[i + 1].token_pos - self._slots[i - 1].token_pos,
                )
                self._slots.pop(best_i)
        elif self.policy == "dyadic_ruler":
            if self.capacity == 1:
                self._slots.pop(-1)
            else:
                # Never evict slot 0 (prompt anchor); evict lowest dyadic level among 1..len-2 (oldest tie-break).
                candidates = range(1, len(self._slots) - 1) if len(self._slots) > 2 else [1]
                evict_i = min(candidates, key=lambda i: (self._slots[i].level, i))
                self._slots.pop(evict_i)

    def nearest_at_or_before(self, target_pos: int) -> RecurrentSnapshot | None:
        """Return retained snapshot with largest token_pos <= target_pos, or None if none exists."""
        best: RecurrentSnapshot | None = None
        for s in self._slots:
            if s.token_pos <= target_pos:
                best = s
            else:
                break
        return best

    def prune_after(self, target_pos: int) -> None:
        """Discard invalidated future snapshots strictly beyond `target_pos` after a rollback."""
        self._slots = [s for s in self._slots if s.token_pos <= target_pos]

