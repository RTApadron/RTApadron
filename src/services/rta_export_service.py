"""Export M4 RTA match results to disk and downloadable bytes.

Two files per export:
    output/<well_id>_rta_match_summary.json  — match params + full config
    output/<well_id>_rta_overlay.png         — overlay plot snapshot
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.rta.models import RTAConfig
from src.services.rta_match_params_service import RTAMatchParams


def build_match_summary(
    *,
    match_params: RTAMatchParams,
    ref_curve_id: str,
    config: RTAConfig,
    curve_status: str = "demo",
) -> dict:
    """Assemble the match summary dict for JSON export."""
    if curve_status == "validated":
        _status = "preliminary"
        _notes = (
            "Curvas tipo analíticas validadas. "
            "Interpretación preliminar — pendiente validación vs software comercial."
        )
    else:
        _status = "demo"
        _notes = (
            "DEMO — curvas tipo no digitalizadas/validadas. "
            "No usar para interpretación técnica final."
        )
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "status": _status,
        "notes": _notes,
        "well_id": config.well_id,
        "method": match_params.method,
        "ref_curve_id": ref_curve_id,
        "match": {
            "x_multiplier": match_params.effective_x_multiplier,
            "y_multiplier": match_params.effective_y_multiplier,
        },
        "results": {
            "kh_md_ft": match_params.kh_md_ft,
            "k_md": match_params.k_md,
            # Volumetric OOIP from config geometry (does not change with joystick)
            "n_vol_stb": match_params.n_vol_stb,
            "n_vol_mm_stb": (
                round(match_params.n_vol_stb / 1e6, 4) if match_params.n_vol_stb else None
            ),
            # Static drainage geometry from config (re_ft / area_acres)
            "re_ft": match_params.re_ft,
            "area_acres": match_params.area_acres,
            "ln_re_rw_term": match_params.ln_re_rw_term,
            # Dynamic match values from joystick position (None if multipliers = 1.0)
            "n_dyn_stb": getattr(match_params, "n_dyn_stb", None),
            "re_dyn_ft": getattr(match_params, "re_dyn_ft", None),
            "a_dyn_acres": getattr(match_params, "a_dyn_acres", None),
        },
        "config": json.loads(config.model_dump_json()),
        "warnings": match_params.warnings,
    }


def save_match_summary(summary: dict, output_dir: Path) -> Path:
    """Write to <output_dir>/<well_id>_rta_<method>_match_summary.json.

    One file per method so Fetkovich and Blasingame saves don't overwrite each other.
    M5 aggregator scans for all method files.
    """
    well_id = str(summary.get("well_id", "unknown"))
    method = str(summary.get("method", "unknown")).replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{well_id}_rta_{method}_match_summary.json"
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def save_overlay_png(
    png_bytes: bytes,
    well_id: str,
    output_dir: Path,
    method: str | None = None,
) -> Path:
    """Write the overlay PNG to disk.

    When *method* is provided the file is named
    ``<well_id>_rta_<method>_overlay.png`` so that Fetkovich and Blasingame
    saves do not overwrite each other.  Without *method* the legacy name
    ``<well_id>_rta_overlay.png`` is used (backward compat).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if method:
        filename = f"{well_id}_rta_{method}_overlay.png"
    else:
        filename = f"{well_id}_rta_overlay.png"
    path = output_dir / filename
    path.write_bytes(png_bytes)
    return path
