"""M1 + M2 integration service.

This service receives normalized well history, optionally enriches it with
well geometry/survey context, completes Pwf when needed, applies the Pwf-used
rule, and joins PVT properties.

The output is ready for DCA/RTA and preserves traceability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.adapters.m1_geometry_adapter import (
    WellGeometryContext,
    apply_geometry_context_to_history,
)
from src.adapters.m1_loader_adapter import apply_pwf_rule
from src.adapters.m2_pvt_adapter import build_pvt_table
from src.adapters.pwf_estimator_adapter import (
    PwfEstimationDefaults,
    estimate_missing_pwf_v1,
)
from src.domain.models import PVTConfig

PWF_TRACE_COLUMNS = [
    "pwf_estimation_method",
    "pwf_estimation_whp_used_psia",
    "pwf_estimation_api_used",
    "pwf_estimation_tvd_used_ft",
    "pwf_estimation_tubing_id_used_in",
    "pwf_estimation_length_used_ft",
    "pwf_estimation_used_default_whp",
    "pwf_estimation_used_default_api",
    "pwf_estimation_used_default_tvd",
    "pwf_estimation_used_default_tubing_id",
    "pwf_estimation_used_default_length",
    "pwf_estimation_force_reestimated",
]

GEOMETRY_TRACE_COLUMNS = [
    "geometry_source",
    "survey_source",
    "tvd_perf_ft",
    "perforation_mid_tvd_ft",
    "perforation_top_md_ft",
    "perforation_bottom_md_ft",
    "sensor_depth_md_ft",
    "pump_depth_md_ft",
    "tubing_id_in",
    "length_ft",
]

ENRICHED_COLUMNS = [
    "well_id",
    "date",
    "qo_stb_d",
    "qg_mscf_d",
    "qw_stb_d",
    "whp_psia",
    "t_wh_f",
    "pwf_measured_psia",
    "pwf_estimated_psia",
    "pwf_used_psia",
    "pwf_source",
    *PWF_TRACE_COLUMNS,
    *GEOMETRY_TRACE_COLUMNS,
    "bo",
    "rs",
    "mu_o_cp",
    "rho_o_lbft3",
    "pb_psia",
    "pvt_model_version",
    "oil_corr",
    "calibrated_flag",
]


@dataclass(frozen=True)
class IntegrationOutput:
    """M1 + M2 integration result."""

    enriched: pd.DataFrame
    qc_report: dict[str, Any]


def integrate_history_with_pvt(
    history_df: pd.DataFrame,
    pvt_cfg: PVTConfig,
    *,
    auto_estimate_missing_pwf: bool = True,
    pwf_defaults: PwfEstimationDefaults | None = None,
    geometry_context: WellGeometryContext | None = None,
) -> IntegrationOutput:
    """Integrate production history with PVT properties."""
    _validate_history(history_df)

    history = history_df.copy()

    history = _add_api_from_pvt_config_if_missing(history, pvt_cfg)
    history = apply_geometry_context_to_history(history, geometry_context)

    if auto_estimate_missing_pwf:
        history = estimate_missing_pwf_v1(
            history,
            defaults=pwf_defaults,
            force_reestimate=geometry_context is not None,
        )

    history = apply_pwf_rule(history)
    pvt = build_pvt_table(history, pvt_cfg)

    enriched = history.merge(
        pvt,
        on=["well_id", "date"],
        how="left",
        validate="one_to_one",
    )

    for column in ENRICHED_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = pd.NA

    enriched = enriched[ENRICHED_COLUMNS].copy()
    enriched = enriched.sort_values(["well_id", "date"]).reset_index(drop=True)

    report = build_qc_report(enriched)

    return IntegrationOutput(enriched=enriched, qc_report=report)


def build_qc_report(enriched: pd.DataFrame) -> dict[str, Any]:
    """Build simple serializable QC report."""
    total_rows = int(len(enriched))

    pwf_source_counts = (
        enriched["pwf_source"].fillna("missing").value_counts().to_dict()
        if "pwf_source" in enriched.columns
        else {}
    )

    pvt_columns = ["bo", "rs", "mu_o_cp", "rho_o_lbft3", "pb_psia"]
    missing_pvt_by_column = {
        column: int(enriched[column].isna().sum())
        for column in pvt_columns
        if column in enriched.columns
    }

    return {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total_rows": total_rows,
        "date_min": _safe_date_min(enriched),
        "date_max": _safe_date_max(enriched),
        "wells": sorted(enriched["well_id"].dropna().astype(str).unique().tolist()),
        "pwf_source_counts": pwf_source_counts,
        "pwf_estimation_trace": _build_pwf_trace_summary(enriched),
        "geometry_trace": _build_geometry_trace_summary(enriched),
        "warnings": _build_warnings(enriched),
        "missing_pvt_by_column": missing_pvt_by_column,
        "has_required_columns": all(
            column in enriched.columns for column in ENRICHED_COLUMNS
        ),
    }


def write_outputs(
    output: IntegrationOutput,
    *,
    well_id: str,
    output_dir: str | Any = "output",
) -> tuple[Any, Any]:
    """Write enriched CSV and QC JSON."""
    import json
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_path = out_dir / f"{well_id}_history_enriched.csv"
    qc_path = out_dir / f"{well_id}_qc_report.json"

    output.enriched.to_csv(enriched_path, index=False)

    with qc_path.open("w", encoding="utf-8") as file:
        json.dump(output.qc_report, file, indent=2, ensure_ascii=False)

    return enriched_path, qc_path


def _add_api_from_pvt_config_if_missing(
    history: pd.DataFrame,
    pvt_cfg: PVTConfig,
) -> pd.DataFrame:
    out = history.copy()

    if "api" not in out.columns:
        api = getattr(pvt_cfg, "api", pd.NA)
        out["api"] = api

    return out


def _build_pwf_trace_summary(enriched: pd.DataFrame) -> dict[str, Any]:
    estimated_v1 = (
        enriched["pwf_source"].astype("string") == "estimated_v1"
        if "pwf_source" in enriched.columns
        else pd.Series(False, index=enriched.index)
    )

    return {
        "estimated_v1_rows": int(estimated_v1.sum()),
        "force_reestimated_rows": _count_true(
            enriched,
            "pwf_estimation_force_reestimated",
        ),
        "used_default_whp_rows": _count_true(
            enriched,
            "pwf_estimation_used_default_whp",
        ),
        "used_default_api_rows": _count_true(
            enriched,
            "pwf_estimation_used_default_api",
        ),
        "used_default_tvd_rows": _count_true(
            enriched,
            "pwf_estimation_used_default_tvd",
        ),
        "used_default_tubing_id_rows": _count_true(
            enriched,
            "pwf_estimation_used_default_tubing_id",
        ),
        "used_default_length_rows": _count_true(
            enriched,
            "pwf_estimation_used_default_length",
        ),
    }


def _build_geometry_trace_summary(enriched: pd.DataFrame) -> dict[str, Any]:
    return {
        "has_geometry_source": _has_non_null(enriched, "geometry_source"),
        "has_survey_source": _has_non_null(enriched, "survey_source"),
        "geometry_sources": _unique_strings(enriched, "geometry_source"),
        "survey_sources": _unique_strings(enriched, "survey_source"),
        "rows_with_tvd_perf_ft": _count_valid_positive(enriched, "tvd_perf_ft"),
        "rows_with_tubing_id_in": _count_valid_positive(enriched, "tubing_id_in"),
        "rows_with_length_ft": _count_valid_positive(enriched, "length_ft"),
    }


def _build_warnings(enriched: pd.DataFrame) -> list[str]:
    warnings: list[str] = []

    estimated_v1_rows = int(
        (enriched["pwf_source"].astype("string") == "estimated_v1").sum()
        if "pwf_source" in enriched.columns
        else 0
    )

    if estimated_v1_rows:
        warnings.append(
            f"{estimated_v1_rows} filas estimaron Pwf con estimate_pwf_v1; "
            "validar contra mediciones de fondo o modelo hidráulico calibrado."
        )

    default_checks = [
        (
            "pwf_estimation_used_default_whp",
            "whp_psia",
            "presión de cabeza",
        ),
        ("pwf_estimation_used_default_api", "api", "API"),
        ("pwf_estimation_used_default_tvd", "tvd_perf_ft", "TVD frente a perforados"),
        (
            "pwf_estimation_used_default_tubing_id",
            "tubing_id_in",
            "ID de tubing",
        ),
        ("pwf_estimation_used_default_length", "length_ft", "longitud de flujo"),
    ]

    for column, field_name, label in default_checks:
        rows = _count_true(enriched, column)
        if rows:
            warnings.append(
                f"{rows} filas estimaron Pwf usando valor por defecto para "
                f"{label} ({field_name})."
            )

    if not _has_non_null(enriched, "geometry_source"):
        warnings.append(
            "No se usó archivo de geometría M1 para estimar Pwf; "
            "se aplicaron datos existentes o defaults."
        )

    return warnings


def _count_true(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0

    values = df[column].map(lambda value: bool(value) if pd.notna(value) else False)
    return int(values.sum())


def _count_valid_positive(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0

    values = pd.to_numeric(df[column], errors="coerce")
    return int((values.notna() & (values > 0)).sum())


def _has_non_null(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and bool(df[column].notna().any())


def _unique_strings(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []

    return sorted(df[column].dropna().astype(str).unique().tolist())


def _validate_history(history_df: pd.DataFrame) -> None:
    required = {
        "well_id",
        "date",
        "qo_stb_d",
        "qg_mscf_d",
        "qw_stb_d",
        "whp_psia",
        "t_wh_f",
        "pwf_measured_psia",
        "pwf_estimated_psia",
    }
    missing = required.difference(history_df.columns)

    if missing:
        msg = f"La historia no cumple contrato M1. Faltan columnas: {sorted(missing)}"
        raise ValueError(msg)

    if history_df.empty:
        msg = "La historia está vacía después de filtros."
        raise ValueError(msg)


def _safe_date_min(df: pd.DataFrame) -> str | None:
    if "date" not in df.columns or df.empty:
        return None

    value = pd.to_datetime(df["date"], errors="coerce").min()
    return None if pd.isna(value) else str(value.date())


def _safe_date_max(df: pd.DataFrame) -> str | None:
    if "date" not in df.columns or df.empty:
        return None

    value = pd.to_datetime(df["date"], errors="coerce").max()
    return None if pd.isna(value) else str(value.date())