"""Manual type-curve overlay preparation for ecoRTA M4.

The functions in this module do not implement final Fetkovich,
Palacio-Blasingame or Agarwal-Gardner interpretation. They transform validated
RTA diagnostic points with user-editable scale multipliers so the interpreter
can start manual log-log matching against validated type curves in the UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.rta.models import RTAMatchConfig


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_manual_match_points(
    diagnostics_df: pd.DataFrame,
    config: RTAMatchConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build manually scaled RTA points from an RTA diagnostic table."""
    required_columns = {config.x_column, config.y_column}
    missing_columns = required_columns.difference(diagnostics_df.columns)
    if missing_columns:
        msg = (
            "La tabla diagnóstica RTA no contiene las columnas requeridas para "
            f"matching manual: {sorted(missing_columns)}"
        )
        raise ValueError(msg)

    df = diagnostics_df.copy()

    x_raw = _to_numeric(df[config.x_column])
    y_raw = _to_numeric(df[config.y_column])
    valid = x_raw.notna() & y_raw.notna() & (x_raw > 0) & (y_raw > 0)

    points = pd.DataFrame(
        {
            "well_id": df.get("well_id", pd.Series([config.well_id] * len(df))),
            "date": df.get("date", pd.Series([pd.NA] * len(df))),
            "method": config.method,
            "match_name": config.match_name,
            "x_column": config.x_column,
            "y_column": config.y_column,
            "x_raw": x_raw,
            "y_raw": y_raw,
            "x_multiplier": config.x_multiplier,
            "y_multiplier": config.y_multiplier,
            "x_match": x_raw * config.x_multiplier,
            "y_match": y_raw * config.y_multiplier,
            "valid_match_point": valid,
            "match_model_version": config.match_model_version,
        }
    )

    points = points[valid].copy().reset_index(drop=True)

    if points.empty:
        msg = (
            "No hay puntos positivos válidos para matching manual. Revisa la "
            "tabla diagnóstica, drawdown y columnas seleccionadas."
        )
        raise ValueError(msg)

    qc_report = {
        "well_id": config.well_id,
        "method": config.method,
        "match_name": config.match_name,
        "match_model_version": config.match_model_version,
        "input_rows": int(len(diagnostics_df)),
        "valid_match_rows": int(len(points)),
        "x_column": config.x_column,
        "y_column": config.y_column,
        "x_multiplier": config.x_multiplier,
        "y_multiplier": config.y_multiplier,
        "x_match_min": float(points["x_match"].min()),
        "x_match_max": float(points["x_match"].max()),
        "y_match_min": float(points["y_match"].min()),
        "y_match_max": float(points["y_match"].max()),
        "notes": [
            "Matching manual preliminar: solo escala puntos diagnósticos.",
            "No se calculan k, skin, volumen contactado ni OOIP en este paso.",
            "Los multiplicadores permiten desplazar puntos sobre ejes log-log.",
        ],
    }

    return points, qc_report


def run_manual_match(
    *,
    diagnostics_csv: Path,
    config: RTAMatchConfig,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Read diagnostics, build manually scaled match points and write artifacts."""
    if not diagnostics_csv.exists():
        msg = f"No existe tabla diagnóstica RTA: {diagnostics_csv}"
        raise FileNotFoundError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_df = pd.read_csv(diagnostics_csv)
    points, qc_report = build_manual_match_points(diagnostics_df, config)

    points_path = output_dir / f"{config.well_id}_rta_manual_match_points.csv"
    qc_path = output_dir / f"{config.well_id}_rta_manual_match_qc_report.json"

    points.to_csv(points_path, index=False)
    qc_path.write_text(
        json.dumps(qc_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return points_path, qc_path
