"""Servicio de integración M1 + M2.

Recibe historia normalizada, aplica regla de Pwf y une propiedades PVT.
Este servicio queda listo para ser consumido por CLI, tests o Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.adapters.m1_loader_adapter import apply_pwf_rule
from src.adapters.m2_pvt_adapter import build_pvt_table
from src.domain.models import PVTConfig


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
    """Resultado del servicio de integración."""

    enriched: pd.DataFrame
    qc_report: dict[str, Any]


def integrate_history_with_pvt(
    history_df: pd.DataFrame,
    pvt_cfg: PVTConfig,
) -> IntegrationOutput:
    """Integra historia M1 con propiedades PVT M2."""

    _validate_history(history_df)

    history = apply_pwf_rule(history_df)
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
        "missing_pvt_by_column": missing_pvt_by_column,
        "has_required_columns": all(column in enriched.columns for column in ENRICHED_COLUMNS),
    }


def write_outputs(
    output: IntegrationOutput,
    *,
    well_id: str,
    output_dir: str | Path = "output",
) -> tuple[Path, Path]:
    """Escribe CSV enriquecido y reporte QC JSON."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_path = out_dir / f"{well_id}_history_enriched.csv"
    qc_path = out_dir / f"{well_id}_qc_report.json"

    output.enriched.to_csv(enriched_path, index=False)

    import json

    with qc_path.open("w", encoding="utf-8") as file:
        json.dump(output.qc_report, file, indent=2, ensure_ascii=False)

    return enriched_path, qc_path


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