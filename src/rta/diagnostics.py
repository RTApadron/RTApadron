"""Diagnostic data preparation for ecoRTA M4.

This module intentionally prepares RTA-ready diagnostic variables without
performing final type-curve matching. Matching Fetkovich, Palacio-Blasingame or
Agarwal-Gardner should be implemented on top of this validated table.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.rta.models import RTAConfig


REQUIRED_HISTORY_COLUMNS = ("date", "qo_stb_d", "pwf_used_psia")
OPTIONAL_HISTORY_COLUMNS = (
    "well_id",
    "bo",
    "rs",
    "mu_o_cp",
    "rho_o_lbft3",
    "pb_psia",
    "pvt_model_version",
    "oil_corr",
    "calibrated_flag",
)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _validate_history_columns(history_df: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_HISTORY_COLUMNS if column not in history_df.columns]


def _build_input_qc(history_df: pd.DataFrame) -> dict[str, Any]:
    """Build QC counters from the raw enriched history before filtering."""
    qc: dict[str, Any] = {
        "input_missing_date_rows": None,
        "input_missing_qo_rows": None,
        "input_non_positive_qo_rows": None,
        "input_missing_pwf_used_rows": None,
        "input_duplicate_date_rows": None,
    }

    if "date" in history_df.columns:
        dates = pd.to_datetime(history_df["date"], errors="coerce")
        qc["input_missing_date_rows"] = int(dates.isna().sum())
        qc["input_duplicate_date_rows"] = int(dates[dates.notna()].duplicated(keep=False).sum())

    if "qo_stb_d" in history_df.columns:
        qo = _to_numeric(history_df["qo_stb_d"])
        qc["input_missing_qo_rows"] = int(qo.isna().sum())
        qc["input_non_positive_qo_rows"] = int((qo.notna() & (qo <= 0)).sum())

    if "pwf_used_psia" in history_df.columns:
        pwf = _to_numeric(history_df["pwf_used_psia"])
        qc["input_missing_pwf_used_rows"] = int(pwf.isna().sum())

    return qc


def _build_near_duplicate_mb_time_report(
    diagnostics: pd.DataFrame,
    *,
    log10_bucket_decimals: int = 3,
    min_normalized_rate_ratio: float = 1.05,
) -> dict[str, Any]:
    """Detect repeated or near-repeated material-balance time with different q/dp.

    The grouping uses rounded log10(t_mb), which is appropriate for log-log RTA
    diagnostics. A bucket precision of 3 decimals groups points that are very close
    visually on a log axis without requiring exact equality in floating point values.
    """
    required = {"material_balance_time_days", "normalized_rate_stb_d_psi"}
    if not required.issubset(diagnostics.columns):
        return {
            "near_duplicate_mb_time_group_count": 0,
            "near_duplicate_mb_time_row_count": 0,
            "near_duplicate_mb_time_groups": [],
            "near_duplicate_mb_time_settings": {
                "log10_bucket_decimals": log10_bucket_decimals,
                "min_normalized_rate_ratio": min_normalized_rate_ratio,
            },
        }

    columns = ["material_balance_time_days", "normalized_rate_stb_d_psi"]
    if "date" in diagnostics.columns:
        columns.append("date")

    df = diagnostics[columns].copy()
    df["material_balance_time_days"] = _to_numeric(df["material_balance_time_days"])
    df["normalized_rate_stb_d_psi"] = _to_numeric(df["normalized_rate_stb_d_psi"])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df[
        (df["material_balance_time_days"] > 0)
        & (df["normalized_rate_stb_d_psi"] > 0)
    ].copy()

    if df.empty:
        return {
            "near_duplicate_mb_time_group_count": 0,
            "near_duplicate_mb_time_row_count": 0,
            "near_duplicate_mb_time_groups": [],
            "near_duplicate_mb_time_settings": {
                "log10_bucket_decimals": log10_bucket_decimals,
                "min_normalized_rate_ratio": min_normalized_rate_ratio,
            },
        }

    df["tmb_log10_bucket"] = df["material_balance_time_days"].map(
        lambda value: round(math.log10(float(value)), log10_bucket_decimals)
    )

    groups: list[dict[str, Any]] = []
    row_count = 0

    for bucket, group in df.groupby("tmb_log10_bucket", dropna=True):
        if len(group) < 2:
            continue

        y_min = float(group["normalized_rate_stb_d_psi"].min())
        y_max = float(group["normalized_rate_stb_d_psi"].max())
        if y_min <= 0:
            continue

        rate_ratio = y_max / y_min
        if rate_ratio < min_normalized_rate_ratio:
            continue

        tmb_min = float(group["material_balance_time_days"].min())
        tmb_max = float(group["material_balance_time_days"].max())
        item: dict[str, Any] = {
            "tmb_log10_bucket": float(bucket),
            "row_count": int(len(group)),
            "material_balance_time_days_min": tmb_min,
            "material_balance_time_days_max": tmb_max,
            "normalized_rate_stb_d_psi_min": y_min,
            "normalized_rate_stb_d_psi_max": y_max,
            "normalized_rate_ratio_max_min": rate_ratio,
        }

        if "date" in group.columns and group["date"].notna().any():
            item["date_min"] = group["date"].min().date().isoformat()
            item["date_max"] = group["date"].max().date().isoformat()

        groups.append(item)
        row_count += int(len(group))

    groups = sorted(
        groups,
        key=lambda item: item["normalized_rate_ratio_max_min"],
        reverse=True,
    )

    return {
        "near_duplicate_mb_time_group_count": len(groups),
        "near_duplicate_mb_time_row_count": row_count,
        "near_duplicate_mb_time_groups": groups[:20],
        "near_duplicate_mb_time_settings": {
            "log10_bucket_decimals": log10_bucket_decimals,
            "min_normalized_rate_ratio": min_normalized_rate_ratio,
            "interpretation": (
                "Grupos con t_mb casi igual en eje log y q/dp diferente. "
                "Revisar cambios de drawdown, Pwf, tasa o condiciones operativas."
            ),
        },
    }


def _prepare_history(history_df: pd.DataFrame, config: RTAConfig) -> pd.DataFrame:
    missing = _validate_history_columns(history_df)
    if missing:
        msg = (
            "La historia enriquecida no contiene las columnas mínimas para M4 RTA: "
            f"{missing}"
        )
        raise ValueError(msg)

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["qo_stb_d"] = _to_numeric(df["qo_stb_d"])
    df["pwf_used_psia"] = _to_numeric(df["pwf_used_psia"])

    if "well_id" not in df.columns:
        df["well_id"] = config.well_id
    else:
        df["well_id"] = df["well_id"].fillna(config.well_id).astype(str)

    for column in OPTIONAL_HISTORY_COLUMNS:
        if column in df.columns and column not in ("well_id", "pvt_model_version", "oil_corr"):
            df[column] = _to_numeric(df[column])

    df = df.dropna(subset=["date", "qo_stb_d", "pwf_used_psia"]).sort_values("date")
    df = df[df["qo_stb_d"] > 0].copy()

    return df.reset_index(drop=True)


def build_rta_diagnostics(
    history_df: pd.DataFrame,
    config: RTAConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build RTA diagnostic table from enriched history and editable config."""
    input_qc = _build_input_qc(history_df)
    prepared = _prepare_history(history_df, config)

    if prepared.empty:
        msg = "No hay filas válidas con date, qo_stb_d y pwf_used_psia para M4 RTA."
        raise ValueError(msg)

    first_date = prepared["date"].min()
    prepared["elapsed_days"] = (
        (prepared["date"] - first_date).dt.total_seconds() / 86_400.0
    )

    prepared["delta_p_psia"] = config.pi_psia - prepared["pwf_used_psia"]
    prepared["valid_drawdown"] = prepared["delta_p_psia"] > 0

    prepared["normalized_rate_stb_d_psi"] = pd.NA
    valid_drawdown = prepared["valid_drawdown"]
    prepared.loc[valid_drawdown, "normalized_rate_stb_d_psi"] = (
        prepared.loc[valid_drawdown, "qo_stb_d"]
        / prepared.loc[valid_drawdown, "delta_p_psia"]
    )

    # Trapezoidal cumulative oil integration using calendar days.
    elapsed = prepared["elapsed_days"].astype(float)
    rates = prepared["qo_stb_d"].astype(float)
    delta_days = elapsed.diff().fillna(0.0).clip(lower=0.0)
    avg_rate = (rates + rates.shift(1).fillna(rates)) / 2.0
    prepared["cumulative_oil_stb"] = (avg_rate * delta_days).cumsum()

    prepared["material_balance_time_days"] = pd.NA
    valid_rate = prepared["qo_stb_d"] > 0
    prepared.loc[valid_rate, "material_balance_time_days"] = (
        prepared.loc[valid_rate, "cumulative_oil_stb"]
        / prepared.loc[valid_rate, "qo_stb_d"]
    )

    for column, value in config.as_repeated_columns().items():
        prepared[column] = value

    # Log columns are useful for diagnostic Plotly axes and future type-curve overlays.
    prepared["log_elapsed_days"] = pd.NA
    prepared.loc[prepared["elapsed_days"] > 0, "log_elapsed_days"] = (
        prepared.loc[prepared["elapsed_days"] > 0, "elapsed_days"]
    )

    mb_time = pd.to_numeric(prepared["material_balance_time_days"], errors="coerce")
    prepared["log_material_balance_time_days"] = pd.NA
    prepared.loc[mb_time > 0, "log_material_balance_time_days"] = mb_time.loc[mb_time > 0]

    normalized_rate = pd.to_numeric(
        prepared["normalized_rate_stb_d_psi"],
        errors="coerce",
    )
    prepared["log_normalized_rate_stb_d_psi"] = pd.NA
    prepared.loc[normalized_rate > 0, "log_normalized_rate_stb_d_psi"] = (
        normalized_rate.loc[normalized_rate > 0]
    )

    priority_columns = [
        "well_id",
        "date",
        "elapsed_days",
        "material_balance_time_days",
        "qo_stb_d",
        "pwf_used_psia",
        "pi_psia",
        "delta_p_psia",
        "valid_drawdown",
        "normalized_rate_stb_d_psi",
        "cumulative_oil_stb",
        "bo",
        "rs",
        "mu_o_cp",
        "rho_o_lbft3",
        "pb_psia",
        "ct_1psi",
        "phi_frac",
        "h_ft",
        "rw_ft",
        "area_acres",
        "swi_frac",
        "pvt_model_version",
        "oil_corr",
        "calibrated_flag",
        "rta_model_version",
        "log_elapsed_days",
        "log_material_balance_time_days",
        "log_normalized_rate_stb_d_psi",
    ]
    available_columns = [column for column in priority_columns if column in prepared.columns]
    diagnostics = prepared[available_columns].copy()

    duplicate_date_rows = 0
    if "date" in diagnostics.columns:
        duplicate_date_rows = int(
            diagnostics["date"][diagnostics["date"].notna()]
            .duplicated(keep=False)
            .sum()
        )

    near_duplicate_mb_time_report = _build_near_duplicate_mb_time_report(diagnostics)

    qc_report = {
        "well_id": config.well_id,
        "rta_model_version": config.rta_model_version,
        "input_rows": int(len(history_df)),
        "diagnostic_rows": int(len(diagnostics)),
        **input_qc,
        "duplicate_date_rows_after_filtering": duplicate_date_rows,
        **near_duplicate_mb_time_report,
        "valid_drawdown_rows": int(diagnostics["valid_drawdown"].sum()),
        "invalid_drawdown_rows": int((~diagnostics["valid_drawdown"]).sum()),
        "date_min": diagnostics["date"].min().date().isoformat(),
        "date_max": diagnostics["date"].max().date().isoformat(),
        "pi_psia": config.pi_psia,
        "ct_1psi": config.ct_1psi,
        "phi_frac": config.phi_frac,
        "h_ft": config.h_ft,
        "rw_ft": config.rw_ft,
        "area_acres": config.area_acres,
        "swi_frac": config.swi_frac,
        "notes": [
            "M4.1 prepara variables diagnósticas RTA; no realiza matching final de curvas tipo.",
            "Revisar pi_psia y propiedades de yacimiento antes de interpretar resultados.",
            "delta_p_psia debe ser positivo para usar normalized_rate_stb_d_psi.",
            "Los puntos con t_mb casi repetido y q/dp diferente no se eliminan; se reportan como QC para revisión del intérprete.",
        ],
    }

    return diagnostics, qc_report


def run_rta_diagnostics(
    *,
    history_csv: Path,
    config: RTAConfig,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Read enriched history, build diagnostics and write M4 artifacts."""
    if not history_csv.exists():
        msg = f"No existe historia enriquecida: {history_csv}"
        raise FileNotFoundError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.read_csv(history_csv)
    diagnostics, qc_report = build_rta_diagnostics(history_df, config)

    diagnostics_path = output_dir / f"{config.well_id}_rta_diagnostics.csv"
    qc_path = output_dir / f"{config.well_id}_rta_qc_report.json"

    diagnostics.to_csv(diagnostics_path, index=False)
    qc_path.write_text(
        json.dumps(qc_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return diagnostics_path, qc_path
