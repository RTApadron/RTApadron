"""Adapter for estimating Pwf when no valid flowing bottomhole pressure exists."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    from src.well_mod.pwf import PwfInputs, estimate_pwf_v1
except ModuleNotFoundError:  # Backward compatibility
    from well_mod.pwf import PwfInputs, estimate_pwf_v1  # type: ignore[no-redef]


@dataclass(frozen=True)
class PwfEstimationDefaults:
    """Fallback parameters for Pwf estimator v1."""

    api: float = 30.0
    whp_psia: float = 100.0
    tvd_perf_ft: float = 6000.0
    tubing_id_in: float = 2.375
    length_ft: float | None = None
    cf: float = 0.02


TRACE_COLUMNS = [
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


def estimate_missing_pwf_v1(
    history_df: pd.DataFrame,
    *,
    defaults: PwfEstimationDefaults | None = None,
    force_reestimate: bool = False,
) -> pd.DataFrame:
    """Complete pwf_estimated_psia using estimate_pwf_v1.

    Rules:
    - Valid measured Pwf is never overwritten.
    - If force_reestimate=False, only rows without measured and without estimated
      Pwf are estimated.
    - If force_reestimate=True, rows without measured Pwf are re-estimated even
      if pwf_estimated_psia already exists. This is used when geometry/survey
      inputs change.
    """
    _validate_required_columns(history_df)

    cfg = defaults or PwfEstimationDefaults()
    out = history_df.copy()

    _ensure_trace_columns(out)

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

    if force_reestimate:
        needs_estimation = ~measured_valid
    else:
        needs_estimation = ~measured_valid & ~estimated_valid

    for idx in out.index[needs_estimation]:
        row = out.loc[idx]

        whp_used, whp_default = _value_or_default(
            row.get("whp_psia"),
            default=cfg.whp_psia,
        )
        api_used, api_default = _value_or_default(
            row.get("api"),
            default=cfg.api,
        )
        tvd_used, tvd_default = _value_or_default(
            row.get("tvd_perf_ft"),
            default=cfg.tvd_perf_ft,
        )
        tubing_id_used, tubing_default = _value_or_default(
            row.get("tubing_id_in"),
            default=cfg.tubing_id_in,
        )
        length_used, length_default = _optional_value_or_default(
            row.get("length_ft"),
            default=cfg.length_ft,
        )

        inputs = PwfInputs(
            qo_stb_d=_safe_float(row.get("qo_stb_d"), default=0.0),
            qw_stb_d=_safe_float(row.get("qw_stb_d"), default=0.0),
            api=api_used,
            whp_psia=whp_used,
            tvd_perf_ft=tvd_used,
            tubing_id_in=tubing_id_used,
            length_ft=length_used,
            Cf=cfg.cf,
        )

        out.loc[idx, "pwf_estimated_psia"] = float(estimate_pwf_v1(inputs))
        out.loc[idx, "pwf_estimation_method"] = "estimate_pwf_v1"
        out.loc[idx, "pwf_estimation_whp_used_psia"] = whp_used
        out.loc[idx, "pwf_estimation_api_used"] = api_used
        out.loc[idx, "pwf_estimation_tvd_used_ft"] = tvd_used
        out.loc[idx, "pwf_estimation_tubing_id_used_in"] = tubing_id_used
        out.loc[idx, "pwf_estimation_length_used_ft"] = length_used
        out.loc[idx, "pwf_estimation_used_default_whp"] = whp_default
        out.loc[idx, "pwf_estimation_used_default_api"] = api_default
        out.loc[idx, "pwf_estimation_used_default_tvd"] = tvd_default
        out.loc[idx, "pwf_estimation_used_default_tubing_id"] = tubing_default
        out.loc[idx, "pwf_estimation_used_default_length"] = length_default
        out.loc[idx, "pwf_estimation_force_reestimated"] = bool(force_reestimate)

    existing_estimated = ~measured_valid & estimated_valid & ~needs_estimation
    out.loc[
        existing_estimated & out["pwf_estimation_method"].isna(),
        "pwf_estimation_method",
    ] = "provided_in_history"
    out.loc[
        existing_estimated & out["pwf_estimation_force_reestimated"].isna(),
        "pwf_estimation_force_reestimated",
    ] = False

    return out


def _ensure_trace_columns(df: pd.DataFrame) -> None:
    for column in TRACE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA


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


def _value_or_default(value: object, *, default: float) -> tuple[float, bool]:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return float(default), True
    return float(parsed), False


def _optional_value_or_default(
    value: object,
    *,
    default: float | None,
) -> tuple[float | None, bool]:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default, default is not None
    return float(parsed), False


def _safe_float(value: object, *, default: float) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return float(default)
    return float(parsed)