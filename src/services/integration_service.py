"""Servicio de integración M1 + M2.

Este servicio recibe historia normalizada del Módulo 1, completa Pwf cuando
faltan mediciones/estimaciones, aplica la regla de Pwf usada y une propiedades
PVT del Módulo 2.

La salida queda lista para DCA/RTA y mantiene trazabilidad de datos medidos,
estimados y calculados.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

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
    """Resultado del servicio de integración M1 + M2."""

    enriched: pd.DataFrame
    qc_report: dict[str, Any]


def integrate_history_with_pvt(
    history_df: pd.DataFrame,
    pvt_cfg: PVTConfig,
    *,
    auto_estimate_missing_pwf: bool = True,
    pwf_defaults: PwfEstimationDefaults | None = None,
) -> IntegrationOutput:
    """Integra historia de producción/presión con propiedades PVT.

    Args:
        history_df: Historia normalizada del pozo.
        pvt_cfg: Configuración PVT del pozo.
        auto_estimate_missing_pwf: Si es True, estima Pwf cuando no existe
            medición ni estimación previa.
        pwf_defaults: Parámetros fallback para el estimador Pwf v1.

    Returns:
        IntegrationOutput con tabla enriquecida y reporte QC.
    """
    _validate_history(history_df)

    history = history_df.copy()

    if auto_estimate_missing_pwf:
        history = estimate_missing_pwf_v1(history, defaults=pwf_defaults)

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
    """Genera reporte QC simple y serializable."""
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
    output_dir: str | Path = "output",
) -> tuple[Path, Path]:
    """Escribe CSV enriquecido y reporte QC JSON."""
    import json

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_path = out_dir / f"{well_id}_history_enriched.csv"
    qc_path = out_dir / f"{well_id}_qc_report.json"

    output.enriched.to_csv(enriched_path, index=False)

    with qc_path.open("w", encoding="utf-8") as file:
        json.dump(output.qc_report, file, indent=2, ensure_ascii=False)

    return enriched_path, qc_path


def _build_pwf_trace_summary(enriched: pd.DataFrame) -> dict[str, Any]:
    estimated_v1 = (
        enriched["pwf_source"].astype("string") == "estimated_v1"
        if "pwf_source" in enriched.columns
        else pd.Series(False, index=enriched.index)
    )

    return {
        "estimated_v1_rows": int(estimated_v1.sum()),
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

    return warnings


def _count_true(df: pd.DataFrame, column: str) -> int:
    """Cuenta valores verdaderos evitando FutureWarning de fillna en object dtype."""
    if column not in df.columns:
        return 0

    values = df[column].map(lambda value: bool(value) if pd.notna(value) else False)
    return int(values.sum())


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