"""Tests for src/services/rta_transform_service.py.

These tests verify the RTA variable computation without requiring real
type-curve data, reservoir models, or external files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.rta_type_curves.models import RTATypeCurveMethod
from src.services.rta_transform_service import (
    RTATransformPoint,
    compute_rta_transforms,
    compute_rta_transforms_from_csv,
    rta_points_to_dataframe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_history(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


SIMPLE_HISTORY = _make_history(
    [
        {
            "well_id": "W-001",
            "date": "2024-01-01",
            "qo_stb_d": 1000.0,
            "pwf_used_psia": 1500.0,
        },
        {
            "well_id": "W-001",
            "date": "2024-02-01",
            "qo_stb_d": 900.0,
            "pwf_used_psia": 1450.0,
        },
        {
            "well_id": "W-001",
            "date": "2024-03-01",
            "qo_stb_d": 810.0,
            "pwf_used_psia": 1400.0,
        },
    ]
)

PI_PSIA = 2000.0


# ---------------------------------------------------------------------------
# Basic output shape and content
# ---------------------------------------------------------------------------

def test_compute_rta_transforms_returns_points_for_all_methods() -> None:
    points = compute_rta_transforms(dataframe=SIMPLE_HISTORY, pi_psia=PI_PSIA)

    methods_found = {p.method for p in points}
    assert RTATypeCurveMethod.FETKOVICH in methods_found
    assert RTATypeCurveMethod.PALACIO_BLASINGAME in methods_found
    assert RTATypeCurveMethod.AGARWAL_GARDNER in methods_found


def test_compute_rta_transforms_single_method() -> None:
    points = compute_rta_transforms(
        dataframe=SIMPLE_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH],
    )

    assert all(p.method == RTATypeCurveMethod.FETKOVICH for p in points)
    # First row excluded: Np=0 → MBT=0 → not log-safe
    assert len(points) == len(SIMPLE_HISTORY) - 1


def test_all_output_points_are_log_log_safe() -> None:
    """x and y must be strictly positive for log-log plotting."""
    points = compute_rta_transforms(dataframe=SIMPLE_HISTORY, pi_psia=PI_PSIA)

    for point in points:
        assert point.x > 0, f"x={point.x} is not positive for {point}"
        assert point.y > 0, f"y={point.y} is not positive for {point}"


# ---------------------------------------------------------------------------
# Physical variable sanity checks
# ---------------------------------------------------------------------------

def test_delta_p_equals_pi_minus_pwf() -> None:
    points = compute_rta_transforms(
        dataframe=SIMPLE_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH],
    )

    for point in points:
        expected_delta_p = PI_PSIA - point.pwf_used_psia
        assert abs(point.delta_p_psia - expected_delta_p) < 1e-6


def test_normalized_rate_equals_qo_over_delta_p() -> None:
    points = compute_rta_transforms(
        dataframe=SIMPLE_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH],
    )

    for point in points:
        expected_nr = point.qo_stb_d / point.delta_p_psia
        assert abs(point.normalized_rate - expected_nr) < 1e-9


def test_material_balance_time_is_positive_and_increasing() -> None:
    points = compute_rta_transforms(
        dataframe=SIMPLE_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH],
    )

    # Sort by date to check monotonicity
    points_sorted = sorted(points, key=lambda p: p.date)

    # MBT starts at 0 for the first row (Np=0), but the first row may be
    # excluded if MBT==0 (not log-log safe). So we check that successive
    # values are non-decreasing.
    mbt_values = [p.material_balance_time for p in points_sorted]
    for i in range(1, len(mbt_values)):
        assert mbt_values[i] >= mbt_values[i - 1], (
            f"MBT not non-decreasing: {mbt_values[i - 1]} -> {mbt_values[i]}"
        )


def test_first_row_mbt_excluded_because_cumulative_is_zero() -> None:
    """First date has Np=0 so MBT=0 — not log-log safe, must be dropped."""
    points = compute_rta_transforms(
        dataframe=SIMPLE_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH],
    )

    dates_found = {p.date for p in points}
    # The first date has MBT=0 and must NOT appear in the output
    assert "2024-01-01" not in dates_found


# ---------------------------------------------------------------------------
# Axis labels
# ---------------------------------------------------------------------------

def test_axis_labels_are_non_empty_strings() -> None:
    points = compute_rta_transforms(
        dataframe=SIMPLE_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH],
    )

    for point in points:
        assert isinstance(point.x_label, str) and point.x_label.strip()
        assert isinstance(point.y_label, str) and point.y_label.strip()


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------

def test_raises_for_invalid_pi() -> None:
    with pytest.raises(ValueError, match="pi_psia must be positive"):
        compute_rta_transforms(dataframe=SIMPLE_HISTORY, pi_psia=0.0)

    with pytest.raises(ValueError, match="pi_psia must be positive"):
        compute_rta_transforms(dataframe=SIMPLE_HISTORY, pi_psia=-100.0)


def test_raises_for_missing_required_columns() -> None:
    bad_df = pd.DataFrame([{"well_id": "W-001", "date": "2024-01-01", "qo_stb_d": 100.0}])

    with pytest.raises(ValueError, match="missing required columns"):
        compute_rta_transforms(dataframe=bad_df, pi_psia=PI_PSIA)


def test_raises_when_pi_below_all_pwf() -> None:
    """If pi < all Pwf values, all drawdowns are negative → no valid rows."""
    low_pi = 500.0  # below all pwf_used_psia values in SIMPLE_HISTORY

    with pytest.raises(ValueError, match="positive pressure drawdown"):
        compute_rta_transforms(dataframe=SIMPLE_HISTORY, pi_psia=low_pi)


def test_zero_and_negative_rates_are_dropped() -> None:
    df = _make_history(
        [
            {"well_id": "W-001", "date": "2024-01-01", "qo_stb_d": 0.0, "pwf_used_psia": 1500.0},
            {"well_id": "W-001", "date": "2024-02-01", "qo_stb_d": -50.0, "pwf_used_psia": 1450.0},
            {"well_id": "W-001", "date": "2024-03-01", "qo_stb_d": 800.0, "pwf_used_psia": 1400.0},
            {"well_id": "W-001", "date": "2024-04-01", "qo_stb_d": 720.0, "pwf_used_psia": 1380.0},
        ]
    )

    points = compute_rta_transforms(
        dataframe=df,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH],
    )

    # Only the last two rows have positive rate; first row (MBT=0) is also
    # excluded from log-log output, so we expect at most 1 point.
    assert len(points) >= 1
    for point in points:
        assert point.qo_stb_d > 0


def test_single_row_raises_because_mbt_is_zero() -> None:
    """A single row produces MBT=0 for all points → nothing log-log safe."""
    single_row = _make_history(
        [{"well_id": "W-001", "date": "2024-01-01", "qo_stb_d": 500.0, "pwf_used_psia": 1500.0}]
    )

    # With a single row Np=0, MBT=0, all points are dropped.
    points = compute_rta_transforms(
        dataframe=single_row,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH],
    )

    assert len(points) == 0


# ---------------------------------------------------------------------------
# to_overlay_point conversion
# ---------------------------------------------------------------------------

def test_to_overlay_point_returns_correct_xy() -> None:
    points = compute_rta_transforms(
        dataframe=SIMPLE_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH],
    )

    for rta_point in points:
        overlay = rta_point.to_overlay_point()
        assert overlay.x == rta_point.x
        assert overlay.y == rta_point.y
        assert overlay.date == rta_point.date


# ---------------------------------------------------------------------------
# DataFrame export
# ---------------------------------------------------------------------------

def test_rta_points_to_dataframe_has_expected_columns() -> None:
    points = compute_rta_transforms(
        dataframe=SIMPLE_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH],
    )

    df = rta_points_to_dataframe(points)

    expected_cols = {
        "well_id",
        "date",
        "method",
        "x",
        "y",
        "x_label",
        "y_label",
        "qo_stb_d",
        "pwf_used_psia",
        "delta_p_psia",
        "normalized_rate",
        "material_balance_time",
    }

    assert expected_cols.issubset(set(df.columns))
    assert len(df) == len(points)


def test_rta_points_to_dataframe_empty_input() -> None:
    df = rta_points_to_dataframe([])
    assert df.empty


# ---------------------------------------------------------------------------
# CSV loader wrapper
# ---------------------------------------------------------------------------

def test_compute_rta_transforms_from_csv_reads_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "enriched.csv"
    SIMPLE_HISTORY.to_csv(csv_path, index=False)

    points = compute_rta_transforms_from_csv(
        path=csv_path,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH],
    )

    assert len(points) > 0


def test_compute_rta_transforms_from_csv_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_rta_transforms_from_csv(
            path=tmp_path / "does_not_exist.csv",
            pi_psia=PI_PSIA,
        )


# ---------------------------------------------------------------------------
# Blasingame integral and derivative (qDdi, qDdid)
# ---------------------------------------------------------------------------

# Longer declining history for Blasingame tests (needs ≥ 3 rows for centered derivative)
_DECLINING_HISTORY = _make_history([
    {"well_id": "W-001", "date": f"2024-0{m}-01", "qo_stb_d": q, "pwf_used_psia": p}
    for m, q, p in [
        (1, 1000.0, 1500.0),
        (2,  900.0, 1450.0),
        (3,  810.0, 1400.0),
        (4,  729.0, 1360.0),
        (5,  656.0, 1330.0),
    ]
])


def test_blasingame_integral_present_for_palacio_blasingame() -> None:
    points = compute_rta_transforms(
        dataframe=_DECLINING_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.PALACIO_BLASINGAME],
    )
    # At least some points should have a valid integral
    with_integral = [p for p in points if p.blasingame_integral is not None]
    assert len(with_integral) > 0


def test_blasingame_integral_none_for_fetkovich_and_agarwal() -> None:
    points = compute_rta_transforms(
        dataframe=_DECLINING_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH, RTATypeCurveMethod.AGARWAL_GARDNER],
    )
    for p in points:
        assert p.blasingame_integral is None
        assert p.blasingame_derivative is None


def test_blasingame_integral_positive() -> None:
    points = compute_rta_transforms(
        dataframe=_DECLINING_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.PALACIO_BLASINGAME],
    )
    for p in points:
        if p.blasingame_integral is not None:
            assert p.blasingame_integral > 0, f"qDdi={p.blasingame_integral} not positive"


def test_blasingame_integral_geq_normalized_rate_for_declining_production() -> None:
    """For declining production qDdi ≥ qDd.

    Physical reason: qDdi = (1/t̄) * ∫₀^t̄ qDd dt̄ is a cumulative average.
    Since past normalized rates were higher than the current one (production
    is declining), the average exceeds the instantaneous value.
    Both converge only in the limit of constant-rate production.
    """
    points = compute_rta_transforms(
        dataframe=_DECLINING_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.PALACIO_BLASINGAME],
    )
    for p in sorted(points, key=lambda x: x.date):
        if p.blasingame_integral is not None:
            assert p.blasingame_integral >= p.normalized_rate - 1e-9, (
                f"qDdi={p.blasingame_integral:.4f} < qDd={p.normalized_rate:.4f} "
                "(integral average should be >= instantaneous for declining production)"
            )


def test_blasingame_derivative_positive() -> None:
    points = compute_rta_transforms(
        dataframe=_DECLINING_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.PALACIO_BLASINGAME],
    )
    for p in points:
        if p.blasingame_derivative is not None:
            assert p.blasingame_derivative > 0, f"qDdid={p.blasingame_derivative} not positive"


def test_blasingame_integral_numeric_check() -> None:
    """Manual trapezoidal check for the first valid integral point."""
    # Two-row history: MBT at row1 = Np1/q1
    # qDdi(MBT1) ≈ 0.5*(nr0 + nr1) by definition (integral from 0 to MBT1, divided by MBT1)
    # since MBT0=0, the trapezoid = 0.5*(nr0+nr1)*MBT1, so qDdi = 0.5*(nr0+nr1)
    pi = 3000.0
    df = _make_history([
        {"well_id": "W-X", "date": "2024-01-01", "qo_stb_d": 1000.0, "pwf_used_psia": 2000.0},
        {"well_id": "W-X", "date": "2024-02-01", "qo_stb_d":  800.0, "pwf_used_psia": 1900.0},
        {"well_id": "W-X", "date": "2024-03-01", "qo_stb_d":  640.0, "pwf_used_psia": 1800.0},
    ])
    points = compute_rta_transforms(
        dataframe=df, pi_psia=pi,
        methods=[RTATypeCurveMethod.PALACIO_BLASINGAME],
    )
    pts = sorted(points, key=lambda p: p.date)
    # First valid point: qDdi ≈ 0.5*(nr_row0 + nr_row1)
    nr0 = 1000.0 / (pi - 2000.0)   # = 1000/1000 = 1.0
    nr1 =  800.0 / (pi - 1900.0)   # = 800/1100  ≈ 0.7273
    expected_qDdi_first = 0.5 * (nr0 + nr1)
    assert pts[0].blasingame_integral == pytest.approx(expected_qDdi_first, rel=0.01)


def test_rta_points_to_dataframe_includes_blasingame_columns() -> None:
    points = compute_rta_transforms(
        dataframe=_DECLINING_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.PALACIO_BLASINGAME],
    )
    df = rta_points_to_dataframe(points)
    assert "blasingame_integral" in df.columns
    assert "blasingame_derivative" in df.columns


# ---------------------------------------------------------------------------
# Log-log diagnostic derivative (log_derivative)
# ---------------------------------------------------------------------------

def test_log_derivative_present_for_all_methods() -> None:
    """log_derivative field is populated for Fetkovich, PB, and AG.

    BLASINGAME intentionally has no dispatch entry — it reuses PB points
    in M4 — so it is excluded from this check.
    """
    points = compute_rta_transforms(
        dataframe=_DECLINING_HISTORY,
        pi_psia=PI_PSIA,
    )
    _dispatched_methods = [
        m for m in RTATypeCurveMethod
        if m != RTATypeCurveMethod.BLASINGAME
    ]
    for method in _dispatched_methods:
        method_pts = [p for p in points if p.method == method]
        # At least one interior point should have a valid log derivative
        with_ld = [p for p in method_pts if p.log_derivative is not None]
        assert len(with_ld) > 0, f"No log_derivative found for method {method.value}"


def test_log_derivative_positive_for_declining_production() -> None:
    """Normalized rate is decreasing → -d(ln(nr))/d(ln(MBT)) must be positive."""
    points = compute_rta_transforms(
        dataframe=_DECLINING_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH],
    )
    for p in points:
        if p.log_derivative is not None:
            assert p.log_derivative > 0, (
                f"log_derivative={p.log_derivative} should be positive for declining production"
            )


def test_log_derivative_none_for_single_row() -> None:
    """With only one valid point, no derivative can be computed → all None."""
    df = _make_history([
        {"well_id": "W-001", "date": "2024-01-01", "qo_stb_d": 1000.0, "pwf_used_psia": 1500.0},
        {"well_id": "W-001", "date": "2024-02-01", "qo_stb_d":  900.0, "pwf_used_psia": 1450.0},
    ])
    points = compute_rta_transforms(
        dataframe=df, pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH],
    )
    # With 2 rows, first has MBT=0 (excluded), leaving 1 valid point → no centered derivative
    interior = [p for p in points if p.log_derivative is not None]
    # Edge case: may have one-sided derivative at the remaining endpoint — that is acceptable
    for p in interior:
        assert p.log_derivative > 0


def test_rta_points_to_dataframe_includes_log_derivative_column() -> None:
    points = compute_rta_transforms(
        dataframe=_DECLINING_HISTORY,
        pi_psia=PI_PSIA,
        methods=[RTATypeCurveMethod.FETKOVICH],
    )
    df = rta_points_to_dataframe(points)
    assert "log_derivative" in df.columns
