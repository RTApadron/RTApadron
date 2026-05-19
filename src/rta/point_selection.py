"""Editable RTA point selection utilities for ecoRTA M4.

This module keeps exclusion/selection decisions outside the enriched history.
The interpreter can decide which diagnostic points should be used for RTA
matching while preserving the original M1-M2-M3 data and the full RTA
diagnostic table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SELECTION_COLUMNS = (
    "rta_point_id",
    "well_id",
    "date",
    "elapsed_days",
    "material_balance_time_days",
    "qo_stb_d",
    "pwf_used_psia",
    "delta_p_psia",
    "valid_drawdown",
    "normalized_rate_stb_d_psi",
    "use_for_rta",
    "exclusion_reason",
)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def ensure_rta_point_id(diagnostics_df: pd.DataFrame) -> pd.DataFrame:
    """Return diagnostics with a stable integer RTA point id."""
    result = diagnostics_df.copy()

    if "rta_point_id" in result.columns:
        point_id = _to_numeric(result["rta_point_id"])
        if point_id.notna().all() and point_id.is_unique:
            result["rta_point_id"] = point_id.astype(int)
            return result

    result = result.reset_index(drop=True)
    result["rta_point_id"] = range(1, len(result) + 1)
    return result


def build_default_selection_table(diagnostics_df: pd.DataFrame) -> pd.DataFrame:
    """Build the default editable RTA point selection table."""
    diagnostics = ensure_rta_point_id(diagnostics_df)

    selection = diagnostics.copy()
    selection["use_for_rta"] = True
    selection["exclusion_reason"] = ""

    if "valid_drawdown" in selection.columns:
        invalid_drawdown = selection["valid_drawdown"].astype(str).str.lower().isin(
            {"false", "0", "no"}
        )
        selection.loc[invalid_drawdown, "use_for_rta"] = False
        selection.loc[invalid_drawdown, "exclusion_reason"] = "invalid_drawdown"

    if "qo_stb_d" in selection.columns:
        qo = _to_numeric(selection["qo_stb_d"])
        non_positive_qo = qo.notna() & (qo <= 0)
        selection.loc[non_positive_qo, "use_for_rta"] = False
        selection.loc[non_positive_qo, "exclusion_reason"] = "non_positive_qo"

    if "pwf_used_psia" in selection.columns:
        missing_pwf = _to_numeric(selection["pwf_used_psia"]).isna()
        selection.loc[missing_pwf, "use_for_rta"] = False
        selection.loc[missing_pwf, "exclusion_reason"] = "missing_pwf_used_psia"

    available_columns = [column for column in SELECTION_COLUMNS if column in selection.columns]
    return selection[available_columns].copy()


def merge_existing_selection(
    diagnostics_df: pd.DataFrame,
    existing_selection_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge existing user decisions into the current diagnostic points."""
    default_selection = build_default_selection_table(diagnostics_df)

    if existing_selection_df is None or existing_selection_df.empty:
        return default_selection

    existing = existing_selection_df.copy()
    if "rta_point_id" not in existing.columns:
        return default_selection

    existing["rta_point_id"] = _to_numeric(existing["rta_point_id"])
    existing = existing.dropna(subset=["rta_point_id"]).copy()
    if existing.empty:
        return default_selection

    existing["rta_point_id"] = existing["rta_point_id"].astype(int)

    keep_columns = ["rta_point_id"]
    for column in ("use_for_rta", "exclusion_reason"):
        if column in existing.columns:
            keep_columns.append(column)

    merged = default_selection.drop(
        columns=[column for column in ("use_for_rta", "exclusion_reason") if column in default_selection.columns]
    ).merge(
        existing[keep_columns].drop_duplicates(subset=["rta_point_id"], keep="last"),
        on="rta_point_id",
        how="left",
    )

    if "use_for_rta" not in merged.columns:
        merged["use_for_rta"] = True
    else:
        merged["use_for_rta"] = merged["use_for_rta"].fillna(True).astype(bool)

    if "exclusion_reason" not in merged.columns:
        merged["exclusion_reason"] = ""
    else:
        merged["exclusion_reason"] = merged["exclusion_reason"].fillna("").astype(str)

    available_columns = [column for column in SELECTION_COLUMNS if column in merged.columns]
    return merged[available_columns].copy()


def read_selection_csv(path: Path) -> pd.DataFrame | None:
    """Read a point selection CSV if available."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def apply_rta_point_selection(
    diagnostics_df: pd.DataFrame,
    selection_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter diagnostics using an editable point selection table."""
    diagnostics = ensure_rta_point_id(diagnostics_df)

    if selection_df is None or selection_df.empty:
        return diagnostics, {
            "selection_applied": False,
            "selection_rows": 0,
            "diagnostic_rows_before_selection": int(len(diagnostics)),
            "diagnostic_rows_after_selection": int(len(diagnostics)),
            "excluded_rows": 0,
        }

    if "rta_point_id" not in selection_df.columns or "use_for_rta" not in selection_df.columns:
        return diagnostics, {
            "selection_applied": False,
            "selection_rows": int(len(selection_df)),
            "diagnostic_rows_before_selection": int(len(diagnostics)),
            "diagnostic_rows_after_selection": int(len(diagnostics)),
            "excluded_rows": 0,
            "warning": (
                "La selección RTA no contiene columnas rta_point_id/use_for_rta; "
                "se usaron todos los puntos diagnósticos."
            ),
        }

    selection = selection_df[["rta_point_id", "use_for_rta"]].copy()
    selection["rta_point_id"] = _to_numeric(selection["rta_point_id"])
    selection = selection.dropna(subset=["rta_point_id"]).copy()
    selection["rta_point_id"] = selection["rta_point_id"].astype(int)
    selection["use_for_rta"] = selection["use_for_rta"].astype(bool)
    selection = selection.drop_duplicates(subset=["rta_point_id"], keep="last")

    merged = diagnostics.merge(selection, on="rta_point_id", how="left")
    merged["use_for_rta"] = merged["use_for_rta"].fillna(True).astype(bool)

    filtered = merged[merged["use_for_rta"]].copy()
    filtered = filtered.drop(columns=["use_for_rta"])

    qc = {
        "selection_applied": True,
        "selection_rows": int(len(selection_df)),
        "diagnostic_rows_before_selection": int(len(diagnostics)),
        "diagnostic_rows_after_selection": int(len(filtered)),
        "excluded_rows": int(len(diagnostics) - len(filtered)),
    }

    return filtered.reset_index(drop=True), qc
