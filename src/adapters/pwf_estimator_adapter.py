"""Adaptador conservador para estimar Pwf cuando no viene medida ni estimada.

Este adaptador envuelve el estimador existente en src/well_mod/pwf.py sin
modificarlo. La intención es que el pipeline M1-M2 pueda completar
pwf_estimated_psia cuando falte, manteniendo trazabilidad del origen.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    from src.well_mod.pwf import PwfInputs, estimate_pwf_v1
except ModuleNotFoundError:  # Compatibilidad con ejecuciones antiguas
    from well_mod.pwf import PwfInputs, estimate_pwf_v1  # type: ignore[no-redef]


@dataclass(frozen=True)
class PwfEstimationDefaults:
    """Parámetros por defecto para el estimador Pwf v1.

    Estos valores son un fallback conservador. En iteraciones posteriores deben
    venir desde Módulo 1: well static, survey, estado mecánico y lift.
    """

    api: float = 30.0
    whp_psia: float = 100.0
    tvd_perf_ft: float = 6000.0
    tubing_id_in: float = 2.375
    length_ft: float | None = None
    cf: float = 0.02


def estimate_missing_pwf_v1(
    history_df: pd.DataFrame,
    *,
    defaults: PwfEstimationDefaults | None = None,
) -> pd.DataFrame:
    """Completa pwf_estimated_psia solo cuando no hay Pwf medida ni estimada.

    Reglas:
    - Si pwf_measured_psia es válida, no se toca la fila.
    - Si pwf_estimated_psia ya existe y es válida, no se toca la fila.
    - Si faltan ambas, se llama estimate_pwf_v1().
    - Se agrega pwf_estimation_method para trazabilidad.

    Args:
        history_df: Historia normalizada del pozo.
        defaults: Parámetros fallback para el estimador v1.

    Returns:
        DataFrame con pwf_estimated_psia completada cuando aplique.
    """
    _validate_required_columns(history_df)

    cfg = defaults or PwfEstimationDefaults()
    out = history_df.copy()

    if "pwf_estimation_method" not in out.columns:
        out["pwf_estimation_method"] = pd.NA

    out["pwf_measured_psia"] = pd.to_numeric(
        out["pwf_measured_psia"],
        errors="coerce",
    )
    out["pwf_estimated_psia"] = pd.to_numeric(
        out["pwf_estimated_psia"],
        errors="coerce",
    )

    measured_valid = out["pwf_measured_psia"].notna() & (
        out["pwf_measured_psia"] > 0
    )
    estimated_valid = out["pwf_estimated_psia"].notna() & (
        out["pwf_estimated_psia"] > 0
    )
    needs_estimation = ~measured_valid & ~estimated_valid

    for idx in out.index[needs_estimation]:
        row = out.loc[idx]

        inputs = PwfInputs(
            qo_stb_d=_safe_float(row.get("qo_stb_d"), default=0.0),
            qw_stb_d=_safe_float(row.get("qw_stb_d"), default=0.0),
            api=_safe_float(row.get("api"), default=cfg.api),
            whp_psia=_safe_float(row.get("whp_psia"), default=cfg.whp_psia),
            tvd_perf_ft=_safe_float(row.get("tvd_perf_ft"), default=cfg.tvd_perf_ft),
            tubing_id_in=_safe_float(
                row.get("tubing_id_in"),
                default=cfg.tubing_id_in,
            ),
            length_ft=_safe_optional_float(row.get("length_ft"), default=cfg.length_ft),
            Cf=cfg.cf,
        )

        out.loc[idx, "pwf_estimated_psia"] = float(estimate_pwf_v1(inputs))
        out.loc[idx, "pwf_estimation_method"] = "estimate_pwf_v1"

    existing_estimated = ~measured_valid & estimated_valid
    out.loc[
        existing_estimated & out["pwf_estimation_method"].isna(),
        "pwf_estimation_method",
    ] = "provided_in_history"

    return out


def _validate_required_columns(df: pd.DataFrame) -> None:
    required = {
        "qo_stb_d",
        "qw_stb_d",
        "whp_psia",
        "pwf_measured_psia",
        "pwf_estimated_psia",
    }
    missing = required.difference(df.columns)
    if missing:
        msg = f"Faltan columnas para estimar Pwf: {sorted(missing)}"
        raise ValueError(msg)


def _safe_float(value: object, *, default: float) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return float(default)
    return float(parsed)


def _safe_optional_float(value: object, *, default: float | None) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)