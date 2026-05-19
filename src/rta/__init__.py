"""ecoRTA M4 RTA package.

M4 currently prepares diagnostic variables and manual type-curve overlay inputs.
"""

from src.rta.models import (
    RTAConfig,
    RTAMatchConfig,
    default_rta_config,
    default_rta_match_config,
)

__all__ = [
    "RTAConfig",
    "RTAMatchConfig",
    "default_rta_config",
    "default_rta_match_config",
]
