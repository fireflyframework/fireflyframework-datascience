# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration discovery via packaging entry points.

Each adapter package registers its auto-configuration class under the
``firefly_datascience.auto_configuration`` entry-point group; this module loads them, tolerating
adapters whose optional dependency is missing (they are simply skipped — their ``@conditional_on_class``
would have excluded them anyway).
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from fireflyframework_datascience.core.ordering import get_order

logger = logging.getLogger(__name__)

AUTO_CONFIG_GROUP = "firefly_datascience.auto_configuration"


def discover_auto_configurations(group: str = AUTO_CONFIG_GROUP) -> list[type]:
    """Return all auto-configuration classes registered under ``group``, sorted by ``@order``."""
    discovered: list[type] = []
    for ep in entry_points(group=group):
        try:
            obj = ep.load()
        except Exception as exc:  # noqa: BLE001 - missing optional extra → skip this adapter
            logger.debug("Skipping auto-configuration %r: %s", ep.name, exc)
            continue
        if isinstance(obj, type):
            discovered.append(obj)
        else:
            logger.debug("Entry point %r did not resolve to a class; skipping", ep.name)
    discovered.sort(key=get_order)
    return discovered
