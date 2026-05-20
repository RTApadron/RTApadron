"""Save and load RTA reservoir/fluid scenarios as JSON files.

A scenario bundles all the reservoir and fluid inputs required by M4
(pi, phi, h, ct, rw, re, Bo, mu_o, CA) into a single file per well so
the interpreter can resume a session without re-entering every value.

Files are written to  output/<well_id>_rta_scenario.json  by default.
The output/ directory is excluded from version control (.gitignore).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.rta.models import RTAConfig

_DEFAULT_OUTPUT_DIR = Path("output")


def save_rta_scenario(
    config: RTAConfig,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
) -> Path:
    """Persist an RTAConfig to  <output_dir>/<well_id>_rta_scenario.json.

    Creates output_dir if it does not exist. Returns the path written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{config.well_id}_rta_scenario.json"
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_rta_scenario(
    well_id: str,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
) -> RTAConfig | None:
    """Load RTAConfig from  <output_dir>/<well_id>_rta_scenario.json.

    Returns None if the file does not exist or cannot be parsed.
    Corrupted files are silently ignored so the UI never crashes on stale
    on-disk state.
    """
    path = output_dir / f"{well_id}_rta_scenario.json"
    if not path.exists():
        return None
    try:
        return RTAConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None


def scenario_path(
    well_id: str,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
) -> Path:
    """Return the expected path for a well scenario file (may not exist)."""
    return output_dir / f"{well_id}_rta_scenario.json"
