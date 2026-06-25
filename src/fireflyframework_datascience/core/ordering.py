# Copyright 2026 Firefly Software Foundation.
"""Ordering for auto-configurations and beans (mirrors pyfly's ``@order``).

Lower values run/resolve first. Use :data:`HIGHEST_PRECEDENCE` / :data:`LOWEST_PRECEDENCE` to bracket.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

HIGHEST_PRECEDENCE = -(2**31)
LOWEST_PRECEDENCE = 2**31 - 1
DEFAULT_ORDER = 0

_ORDER_ATTR = "__firefly_ds_order__"

T = TypeVar("T")


def order(value: int) -> Callable[[T], T]:
    """Class/function decorator that sets an ordering value (lower runs first)."""

    def _decorator(obj: T) -> T:
        setattr(obj, _ORDER_ATTR, value)
        return obj

    return _decorator


def get_order(obj: object) -> int:
    """Return the ordering value for ``obj`` (:data:`DEFAULT_ORDER` if unset)."""
    return int(getattr(obj, _ORDER_ATTR, DEFAULT_ORDER))
