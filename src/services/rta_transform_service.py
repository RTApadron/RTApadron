"""Service for computing RTA transformed variables from enriched well history.

This module produces physically meaningful RTA points for each supported method
(Fetkovich, Palacio-Blasingame, Agarwal-Gardner) from an enriched history CSV.

It does NOT perform matching, curve digitization, or reservoir parameter
estimation. It only computes the transformed axes needed to overlay real
well data on type curves.

Key output columns per point:
    well_id               - well identifier
    date                  - production date (ISO string)
    method                - RTATypeCurveMethod value
    x                     - transformed x-axis value (positive, log-log safe)
    y                     - transformed y-axis value (positive, log-log safe)
    x_label               - physical label for x axis
    y_label               - physical label for y axis
    qo_stb_d              - oil rate used (STB/d)
    pwf_used_psia         - flowing bottomhole pressure used (psia)
    delta_p_psia          - pressure drawdown: pi - pwf  (psia)
    normalized_rate       - qo / delta_p  (STB/d/psi)
    material_balance_time - cumulative oil / current rate  (days)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.rta_type_curves.models import RTATypeCurveMethod
from src.rta_type_curves.overlay import RTAOverlayPoint


# ---------------------------------------------------------------------------
# Domain dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RTATransformPoint:
    """One row of RTA-transformed data, ready for overlay on type curves."""

    well_id: str
    date: str
    method: RTATypeCurveMethod
    x: float
    y: float
    x_label: str
    y_label: str
    qo_stb_d: float
    pwf_used_psia: float
    delta_p_psia: float
    normalized_rate: float
    material_balance_time: float

    # Blasingame auxiliary functions (Palacio-Blasingame only; None for other methods)
    # qDdi  = (1/t̄) * ∫₀^t̄ (q/Δp) dt̄          [Palacio-Blasingame Ec. 14]
    # qDdid = |d(qDdi)/d(ln(t̄))|                [Palacio-Blasingame Ec. 15]
    blasingame_integral: float | None = field(default=None)
    blasingame_derivative: float | None = field(default=None)

    # Log-log diagnostic derivative (all methods):
    # log_derivative = -d(ln(q/Δp)) / d(ln(MBT))
    # Slope interpretation: -1 → BDF, -0.5 → linear flow, -1/4 → bilinear flow
    log_derivative: float | None = field(default=None)

    def to_overlay_point(self) -> RTAOverlayPoint:
        """Convert to an RTAOverlayPoint for use with the existing overlay engine."""
        return RTAOverlayPoint(
            x=self.x,
            y=self.y,
            label=self.well_id,
            date=self.date,
        )


# ---------------------------------------------------------------------------
# Required column names in the enriched history CSV
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = {
    "well_id",
    "date",
    "qo_stb_d",
    "pwf_used_psia",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_history(dataframe: pd.DataFrame, source: str) -> None:
    missing = _REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Enriched history from '{source}' is missing required columns: "
            f"{sorted(missing)}. "
            "Run the M1-M2 pipeline to produce a complete enriched history."
        )


def _load_and_validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Enriched history CSV not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Enriched history CSV is empty: {path}")

    _validate_history(df, str(path))
    return df


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy of df with the given columns coerced to float."""
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _compute_base_columns(df: pd.DataFrame, pi_psia: float) -> pd.DataFrame:
    """Add delta_p, normalized_rate, and material_balance_time columns."""
    df = _coerce_numeric(df, ["qo_stb_d", "pwf_used_psia"])

    # Keep only rows where both rate and Pwf are positive
    df = df.dropna(subset=["qo_stb_d", "pwf_used_psia"])
    df = df[(df["qo_stb_d"] > 0) & (df["pwf_used_psia"] > 0)]

    if df.empty:
        raise ValueError(
            "No rows with positive qo_stb_d and pwf_used_psia found after cleaning."
        )

    df = df.copy()

    # Pressure drawdown: pi - pwf
    df["delta_p_psia"] = pi_psia - df["pwf_used_psia"]

    # Guard: if drawdown is zero or negative the normalized rate is undefined
    df = df[df["delta_p_psia"] > 0]

    if df.empty:
        raise ValueError(
            f"No rows with positive pressure drawdown (pi={pi_psia} psia). "
            "Check that pi_psia is greater than all Pwf values."
        )

    # Normalized rate: q / Δp
    df["normalized_rate"] = df["qo_stb_d"] / df["delta_p_psia"]

    # Cumulative oil production (trapezoidal integration over time)
    df = df.sort_values("date").reset_index(drop=True)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"])

    if len(df) < 1:
        raise ValueError("No rows with valid dates after parsing.")

    # Days elapsed from first production date
    df["t_days"] = (df["date_dt"] - df["date_dt"].iloc[0]).dt.total_seconds() / 86400.0

    # Cumulative oil via trapezoidal rule (STB)
    df["Np_stb"] = 0.0
    if len(df) > 1:
        cumulative = [0.0]
        for i in range(1, len(df)):
            dt = df["t_days"].iloc[i] - df["t_days"].iloc[i - 1]
            avg_q = (df["qo_stb_d"].iloc[i] + df["qo_stb_d"].iloc[i - 1]) / 2.0
            cumulative.append(cumulative[-1] + avg_q * dt)
        df["Np_stb"] = cumulative

    # Material balance time: Np / q  (days)
    df["material_balance_time"] = df["Np_stb"] / df["qo_stb_d"]

    # Blasingame auxiliary functions (qDdi, qDdid)
    df = _compute_blasingame_functions(df)

    # Log-log diagnostic derivative: -d(ln(q/Δp)) / d(ln(MBT))
    df = _compute_log_derivative(df)

    return df


# ---------------------------------------------------------------------------
# Blasingame auxiliary functions
# ---------------------------------------------------------------------------

def _compute_blasingame_functions(df: pd.DataFrame) -> pd.DataFrame:
    """Add blasingame_integral and blasingame_derivative columns.

    Both are computed on the *already sorted* df with positive
    material_balance_time and normalized_rate values.

    qDdi  = (1/t̄) · ∫₀^t̄ (q/Δp) dt̄        [Palacio-Blasingame Ec. 14]
    qDdid = |d(qDdi)/d(ln(t̄))|               [Palacio-Blasingame Ec. 15]

    Integral: trapezoidal rule on the MBT axis.
    Derivative: Bourdet-style centered log-space finite differences;
        one-sided at end points.  Negative values (noise artifacts) → NaN.

    Physical interpretation:
        - For declining production qDdi < qDd (integral lags instantaneous).
        - qDdid > 0 always (q/Δp is decreasing → its integral average also
          decreases → negative log-derivative → absolute value is positive).
        - At late BDF all three traces (qDd, qDdi, qDdid) converge to the
          b=1 harmonic stem on the Blasingame chart.
    """
    df = df.copy()
    n = len(df)
    mbt = df["material_balance_time"].to_numpy(dtype=float)
    nr = df["normalized_rate"].to_numpy(dtype=float)

    # --- qDdi: cumulative trapezoidal integral / MBT ---
    # Row 0 may have MBT=0 (natural lower bound); integral starts at 0.
    cumul = np.zeros(n)
    for i in range(1, n):
        dt = mbt[i] - mbt[i - 1]
        if dt > 0:
            cumul[i] = cumul[i - 1] + 0.5 * (nr[i] + nr[i - 1]) * dt
        else:
            cumul[i] = cumul[i - 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        qDdi = np.where(mbt > 0, cumul / mbt, np.nan)

    # --- qDdid: Bourdet log-derivative of qDdi ---
    valid = np.isfinite(qDdi) & (mbt > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_mbt = np.where(valid, np.log(mbt), np.nan)
    qDdid = np.full(n, np.nan)

    for i in range(n):
        if not valid[i]:
            continue
        left = i > 0 and valid[i - 1]
        right = i < n - 1 and valid[i + 1]
        if left and right:
            d_ln = ln_mbt[i + 1] - ln_mbt[i - 1]
            if d_ln > 0:
                qDdid[i] = -(qDdi[i + 1] - qDdi[i - 1]) / d_ln
        elif right:
            d_ln = ln_mbt[i + 1] - ln_mbt[i]
            if d_ln > 0:
                qDdid[i] = -(qDdi[i + 1] - qDdi[i]) / d_ln
        elif left:
            d_ln = ln_mbt[i] - ln_mbt[i - 1]
            if d_ln > 0:
                qDdid[i] = -(qDdi[i] - qDdi[i - 1]) / d_ln

    # Negative qDdid = noise artifact → discard
    qDdid = np.where(qDdid > 0, qDdid, np.nan)

    df["blasingame_integral"] = qDdi
    df["blasingame_derivative"] = qDdid
    return df


# ---------------------------------------------------------------------------
# Log-log diagnostic derivative
# ---------------------------------------------------------------------------

def _compute_log_derivative(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the log-log diagnostic derivative of normalized rate.

    log_derivative = -d(ln(q/Δp)) / d(ln(MBT))

    Uses Bourdet centered finite differences in log space; one-sided at
    end points.  Negative or non-finite values are set to NaN (noise).

    Physical interpretation of the slope on a log-log plot:
        slope ≈ -1    → boundary dominated flow (BDF)
        slope ≈ -0.5  → linear flow
        slope ≈ -0.25 → bilinear flow
        slope ≈  0    → pseudo-steady-state / volumetric depletion
    """
    df = df.copy()
    n = len(df)
    mbt = df["material_balance_time"].to_numpy(dtype=float)
    nr  = df["normalized_rate"].to_numpy(dtype=float)

    valid = (mbt > 0) & (nr > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_mbt = np.where(valid, np.log(mbt), np.nan)
        ln_nr  = np.where(valid, np.log(nr),  np.nan)

    log_deriv = np.full(n, np.nan)
    for i in range(n):
        if not valid[i]:
            continue
        left  = i > 0 and valid[i - 1]
        right = i < n - 1 and valid[i + 1]
        if left and right:
            d_lnm = ln_mbt[i + 1] - ln_mbt[i - 1]
            if d_lnm > 0:
                log_deriv[i] = -(ln_nr[i + 1] - ln_nr[i - 1]) / d_lnm
        elif right:
            d_lnm = ln_mbt[i + 1] - ln_mbt[i]
            if d_lnm > 0:
                log_deriv[i] = -(ln_nr[i + 1] - ln_nr[i]) / d_lnm
        elif left:
            d_lnm = ln_mbt[i] - ln_mbt[i - 1]
            if d_lnm > 0:
                log_deriv[i] = -(ln_nr[i] - ln_nr[i - 1]) / d_lnm

    # Keep only positive values; negative = increasing rate = noise artifact
    log_deriv = np.where(log_deriv > 0, log_deriv, np.nan)
    df["log_derivative"] = log_deriv
    return df


# ---------------------------------------------------------------------------
# Per-method transform functions
# ---------------------------------------------------------------------------

def _transform_fetkovich(df: pd.DataFrame) -> list[RTATransformPoint]:
    """Fetkovich type curve axes.

    X = material balance time  (days)
    Y = normalized rate  qo / Δp  (STB/d/psi)

    Physical basis: Fetkovich (1980) combines transient and boundary-dominated
    decline on log-log axes. The normalized rate vs. MBT plot is the standard
    presentation for field data matching.
    """
    points = []
    for _, row in df.iterrows():
        mbt = float(row["material_balance_time"])
        nr = float(row["normalized_rate"])
        if mbt > 0 and nr > 0:
            raw_ld = row.get("log_derivative")
            ld = float(raw_ld) if (raw_ld is not None and pd.notna(raw_ld) and float(raw_ld) > 0) else None
            points.append(
                RTATransformPoint(
                    well_id=str(row["well_id"]),
                    date=str(row["date"]),
                    method=RTATypeCurveMethod.FETKOVICH,
                    x=mbt,
                    y=nr,
                    x_label="Material balance time (days)",
                    y_label="Normalized rate qo/Δp (STB/d/psi)",
                    qo_stb_d=float(row["qo_stb_d"]),
                    pwf_used_psia=float(row["pwf_used_psia"]),
                    delta_p_psia=float(row["delta_p_psia"]),
                    normalized_rate=nr,
                    material_balance_time=mbt,
                    log_derivative=ld,
                )
            )
    return points


def _transform_palacio_blasingame(df: pd.DataFrame) -> list[RTATransformPoint]:
    """Palacio-Blasingame type curve axes.

    X = material balance time  (days)
    Y = normalized rate  qo / Δp  (STB/d/psi)

    Physical basis: Palacio & Blasingame (1993) showed that variable-rate/
    variable-pressure data collapse onto Fetkovich-style decline curves when
    plotted using material balance time. The axes are the same as Fetkovich;
    the distinction lies in how matching parameters map to reservoir properties.

    The full Blasingame plot includes three traces simultaneously:
        qDd   — normalized rate (primary, this function's x/y)
        qDdi  — normalized rate integral  [Ec. 14]
        qDdid — derivative of integral    [Ec. 15]

    All three are stored on RTATransformPoint so the UI can overlay them.
    """
    points = []
    for _, row in df.iterrows():
        mbt = float(row["material_balance_time"])
        nr = float(row["normalized_rate"])
        if mbt > 0 and nr > 0:
            raw_bi = row.get("blasingame_integral")
            raw_bd = row.get("blasingame_derivative")
            raw_ld = row.get("log_derivative")
            bi = float(raw_bi) if (raw_bi is not None and pd.notna(raw_bi) and float(raw_bi) > 0) else None
            bd = float(raw_bd) if (raw_bd is not None and pd.notna(raw_bd) and float(raw_bd) > 0) else None
            ld = float(raw_ld) if (raw_ld is not None and pd.notna(raw_ld) and float(raw_ld) > 0) else None
            points.append(
                RTATransformPoint(
                    well_id=str(row["well_id"]),
                    date=str(row["date"]),
                    method=RTATypeCurveMethod.PALACIO_BLASINGAME,
                    x=mbt,
                    y=nr,
                    x_label="Material balance time (days)",
                    y_label="Normalized rate qo/Δp (STB/d/psi)",
                    qo_stb_d=float(row["qo_stb_d"]),
                    pwf_used_psia=float(row["pwf_used_psia"]),
                    delta_p_psia=float(row["delta_p_psia"]),
                    normalized_rate=nr,
                    material_balance_time=mbt,
                    blasingame_integral=bi,
                    blasingame_derivative=bd,
                    log_derivative=ld,
                )
            )
    return points


def _transform_agarwal_gardner(df: pd.DataFrame) -> list[RTATransformPoint]:
    """Agarwal-Gardner type curve axes.

    X = material balance time  (days)
    Y = normalized rate  qo / Δp  (STB/d/psi)

    Physical basis: Agarwal, Gardner et al. (1999) extended the Blasingame
    approach with additional auxiliary functions. For the primary rate plot
    the axes are identical; the diagnostic power comes from the derivative
    and integral overlays, which are also deferred to a later sprint.
    """
    points = []
    for _, row in df.iterrows():
        mbt = float(row["material_balance_time"])
        nr = float(row["normalized_rate"])
        if mbt > 0 and nr > 0:
            raw_ld = row.get("log_derivative")
            ld = float(raw_ld) if (raw_ld is not None and pd.notna(raw_ld) and float(raw_ld) > 0) else None
            points.append(
                RTATransformPoint(
                    well_id=str(row["well_id"]),
                    date=str(row["date"]),
                    method=RTATypeCurveMethod.AGARWAL_GARDNER,
                    x=mbt,
                    y=nr,
                    x_label="Material balance time (days)",
                    y_label="Normalized rate qo/Δp (STB/d/psi)",
                    qo_stb_d=float(row["qo_stb_d"]),
                    pwf_used_psia=float(row["pwf_used_psia"]),
                    delta_p_psia=float(row["delta_p_psia"]),
                    normalized_rate=nr,
                    material_balance_time=mbt,
                    log_derivative=ld,
                )
            )
    return points


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_TRANSFORM_DISPATCH = {
    RTATypeCurveMethod.FETKOVICH: _transform_fetkovich,
    RTATypeCurveMethod.PALACIO_BLASINGAME: _transform_palacio_blasingame,
    RTATypeCurveMethod.AGARWAL_GARDNER: _transform_agarwal_gardner,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rta_transforms(
    *,
    dataframe: pd.DataFrame,
    pi_psia: float,
    methods: Sequence[RTATypeCurveMethod] | None = None,
) -> list[RTATransformPoint]:
    """Compute RTA-transformed points from an enriched history DataFrame.

    Args:
        dataframe:  Enriched history from the M1-M2 pipeline. Must contain
                    at minimum: well_id, date, qo_stb_d, pwf_used_psia.
        pi_psia:    Initial reservoir pressure (psia). Used to compute Δp.
        methods:    Which methods to compute. Defaults to all three.

    Returns:
        List of RTATransformPoint, one per valid row per method.
        Empty rows (zero/negative rate or Pwf, or negative drawdown) are
        silently dropped — the caller should check the count.
    """
    if pi_psia <= 0:
        raise ValueError(f"pi_psia must be positive, got {pi_psia}.")

    _validate_history(dataframe, source="<dataframe>")

    active_methods = list(methods) if methods is not None else list(RTATypeCurveMethod)

    base_df = _compute_base_columns(dataframe, pi_psia)

    all_points: list[RTATransformPoint] = []

    for method in active_methods:
        transform_fn = _TRANSFORM_DISPATCH[method]
        method_points = transform_fn(base_df)
        all_points.extend(method_points)

    return all_points


def compute_rta_transforms_from_csv(
    *,
    path: Path | str,
    pi_psia: float,
    methods: Sequence[RTATypeCurveMethod] | None = None,
) -> list[RTATransformPoint]:
    """Load an enriched history CSV and compute RTA transforms.

    Thin wrapper around compute_rta_transforms for filesystem-based workflows.
    """
    df = _load_and_validate(Path(path))
    return compute_rta_transforms(dataframe=df, pi_psia=pi_psia, methods=methods)


def rta_points_to_dataframe(points: list[RTATransformPoint]) -> pd.DataFrame:
    """Convert a list of RTATransformPoint to a tidy DataFrame.

    Useful for inspection, CSV export, and Streamlit display.
    """
    if not points:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "well_id": p.well_id,
                "date": p.date,
                "method": p.method.value,
                "x": p.x,
                "y": p.y,
                "x_label": p.x_label,
                "y_label": p.y_label,
                "qo_stb_d": p.qo_stb_d,
                "pwf_used_psia": p.pwf_used_psia,
                "delta_p_psia": p.delta_p_psia,
                "normalized_rate": p.normalized_rate,
                "material_balance_time": p.material_balance_time,
                "blasingame_integral": p.blasingame_integral,
                "blasingame_derivative": p.blasingame_derivative,
                "log_derivative": p.log_derivative,
            }
            for p in points
        ]
    )
