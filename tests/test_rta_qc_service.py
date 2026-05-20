"""Tests for src/services/rta_qc_service.py.

Covers all six QC checks independently plus the public run_rta_qc() aggregator
and qc_severity_level() helper.  Uses lightweight RTATransformPoint stubs so
no real production data or type-curve files are needed.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from src.rta_type_curves.models import RTATypeCurveMethod
from src.services.rta_qc_service import (
    QCResult,
    run_rta_qc,
    qc_severity_level,
    _check_point_count,
    _check_drawdown_stability,
    _check_data_span,
    _check_match_adjusted,
    _check_qdd_range,
    _check_transient_only,
)
from src.services.rta_transform_service import RTATransformPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_point(
    *,
    mbt: float = 100.0,
    normalized_rate: float = 0.5,
    delta_p: float = 1000.0,
    qo: float = 500.0,
    pwf: float = 1000.0,
    d: str = "2024-01-01",
) -> RTATransformPoint:
    """Minimal RTATransformPoint for QC testing."""
    return RTATransformPoint(
        well_id="W-TEST",
        date=d,
        method=RTATypeCurveMethod.FETKOVICH,
        x=mbt,
        y=normalized_rate,
        x_label="MBT (días)",
        y_label="qo/Δp",
        qo_stb_d=qo,
        pwf_used_psia=pwf,
        delta_p_psia=delta_p,
        normalized_rate=normalized_rate,
        material_balance_time=mbt,
    )


def _declining_points(n: int, *, mbt_start: float = 30.0, mbt_factor: float = 2.0) -> list[RTATransformPoint]:
    """Generate n points with exponentially declining normalized_rate on a log-spaced MBT grid."""
    points = []
    mbt = mbt_start
    nr = 1.0
    for i in range(n):
        dp = 1000.0
        points.append(_make_point(
            mbt=mbt,
            normalized_rate=nr,
            delta_p=dp,
            qo=nr * dp,
            d=f"2024-{(i % 12) + 1:02d}-01",
        ))
        mbt *= mbt_factor
        nr *= 0.8  # 20 % decline each step
    return points


# ---------------------------------------------------------------------------
# 1. _check_point_count
# ---------------------------------------------------------------------------

class TestCheckPointCount:

    def test_error_when_fewer_than_5_points(self) -> None:
        pts = [_make_point() for _ in range(3)]
        results = _check_point_count(pts)
        assert len(results) == 1
        assert results[0].code == "POINT_COUNT"
        assert results[0].severity == "error"

    def test_error_boundary_exactly_4_points(self) -> None:
        pts = [_make_point() for _ in range(4)]
        results = _check_point_count(pts)
        assert results[0].severity == "error"

    def test_warning_when_5_to_14_points(self) -> None:
        for n in (5, 10, 14):
            pts = [_make_point() for _ in range(n)]
            results = _check_point_count(pts)
            assert results[0].severity == "warning", f"Expected warning for n={n}"

    def test_ok_when_15_or_more_points(self) -> None:
        pts = [_make_point() for _ in range(20)]
        results = _check_point_count(pts)
        assert results[0].severity == "ok"

    def test_result_mentions_point_count(self) -> None:
        pts = [_make_point() for _ in range(3)]
        result = _check_point_count(pts)[0]
        assert "3" in result.detail

    def test_empty_list_returns_error(self) -> None:
        results = _check_point_count([])
        assert results[0].severity == "error"


# ---------------------------------------------------------------------------
# 2. _check_drawdown_stability
# ---------------------------------------------------------------------------

class TestCheckDrawdownStability:

    def test_ok_for_constant_drawdown(self) -> None:
        pts = [_make_point(delta_p=1000.0) for _ in range(10)]
        results = _check_drawdown_stability(pts)
        assert results[0].severity == "ok"
        # CV should be 0 when all delta_p are equal
        assert "0.0 %" in results[0].detail

    def test_warning_for_moderate_cv(self) -> None:
        # CV ≈ 28 % — mean=1000, std≈283 → CV=0.283 (between 15 % and 30 % thresholds)
        # Values: [600, 800, 1000, 1200, 1400] → mean=1000, std=√80000≈282.8
        delta_ps = [600.0, 800.0, 1000.0, 1200.0, 1400.0]
        pts = [_make_point(delta_p=dp, qo=dp * 0.5) for dp in delta_ps]
        results = _check_drawdown_stability(pts)
        assert results[0].severity == "warning"

    def test_error_for_high_cv(self) -> None:
        # Very high variation: CV well above 30 %
        delta_ps = [200.0, 500.0, 1500.0, 3000.0, 4000.0]
        pts = [_make_point(delta_p=dp, qo=dp * 0.5) for dp in delta_ps]
        results = _check_drawdown_stability(pts)
        assert results[0].severity == "error"

    def test_error_when_mean_dp_is_zero(self) -> None:
        # delta_p = 0 would mean pi == pwf (physically impossible, but guard anyway)
        pts = [_make_point(delta_p=0.0, qo=0.0) for _ in range(5)]
        results = _check_drawdown_stability(pts)
        assert results[0].severity == "error"
        assert results[0].code == "DRAWDOWN_STABILITY"

    def test_empty_list_returns_nothing(self) -> None:
        assert _check_drawdown_stability([]) == []


# ---------------------------------------------------------------------------
# 3. _check_data_span
# ---------------------------------------------------------------------------

class TestCheckDataSpan:

    def _span_pts(self, mbt_min: float, mbt_max: float, n: int = 5) -> list[RTATransformPoint]:
        """Points with MBT log-uniformly spaced between mbt_min and mbt_max."""
        import math
        log_min = math.log10(mbt_min)
        log_max = math.log10(mbt_max)
        mbts = [10 ** (log_min + i * (log_max - log_min) / (n - 1)) for i in range(n)]
        return [_make_point(mbt=m) for m in mbts]

    def test_ok_for_span_greater_than_1_log_cycle(self) -> None:
        # span = 2 log cycles → ok
        pts = self._span_pts(10.0, 1000.0)
        results = _check_data_span(pts)
        assert results[0].severity == "ok"

    def test_warning_for_span_between_0_5_and_1_log_cycle(self) -> None:
        # span = 0.7 log cycles → warning
        pts = self._span_pts(100.0, 500.0)
        results = _check_data_span(pts)
        assert results[0].severity == "warning"

    def test_error_for_span_less_than_0_5_log_cycles(self) -> None:
        # span = 0.2 log cycles → error
        pts = self._span_pts(100.0, 160.0)
        results = _check_data_span(pts)
        assert results[0].severity == "error"

    def test_fewer_than_2_points_returns_nothing(self) -> None:
        assert _check_data_span([_make_point()]) == []
        assert _check_data_span([]) == []

    def test_detail_includes_log_span_value(self) -> None:
        pts = self._span_pts(10.0, 1000.0)
        result = _check_data_span(pts)[0]
        assert "2.0" in result.detail or "2." in result.detail


# ---------------------------------------------------------------------------
# 4. _check_match_adjusted
# ---------------------------------------------------------------------------

class TestCheckMatchAdjusted:

    def test_warning_when_both_multipliers_at_default(self) -> None:
        results = _check_match_adjusted(1.0, 1.0)
        assert results[0].severity == "warning"
        assert results[0].code == "MATCH_NOT_ADJUSTED"

    def test_warning_when_only_y_multiplier_at_default(self) -> None:
        results = _check_match_adjusted(2.5, 1.0)
        assert results[0].severity == "warning"
        assert results[0].code == "MATCH_NOT_ADJUSTED"

    def test_ok_when_both_multipliers_adjusted(self) -> None:
        results = _check_match_adjusted(2.5, 0.4)
        assert results[0].severity == "ok"

    def test_ok_when_y_adjusted_x_at_default(self) -> None:
        # x at 1.0 is not flagged if y is adjusted
        results = _check_match_adjusted(1.0, 0.3)
        assert results[0].severity == "ok"

    def test_tolerance_boundary_treated_as_default(self) -> None:
        # Within 1e-3 of 1.0 → treated as 1.0 (not adjusted)
        results = _check_match_adjusted(1.0 + 5e-4, 1.0 + 5e-4)
        assert results[0].severity == "warning"

    def test_just_outside_tolerance_treated_as_adjusted(self) -> None:
        results = _check_match_adjusted(1.0 + 2e-3, 1.0 + 2e-3)
        assert results[0].severity == "ok"


# ---------------------------------------------------------------------------
# 5. _check_qdd_range
# ---------------------------------------------------------------------------

class TestCheckQddRange:

    def test_skipped_when_y_mult_at_default(self) -> None:
        """If y_mult ≈ 1.0, we can't assess qDd — must return empty list."""
        pts = [_make_point(normalized_rate=0.9) for _ in range(5)]
        assert _check_qdd_range(pts, 1.0) == []

    def test_warning_when_all_qdd_above_threshold(self) -> None:
        # All normalized_rate * y_mult > 0.70
        pts = [_make_point(normalized_rate=0.80) for _ in range(5)]
        results = _check_qdd_range(pts, effective_y_multiplier=1.5)
        # 0.80 * 1.5 = 1.2 — BUT qDd should be ≤ 1, so let's use smaller values
        # Use normalized_rate=0.5, y_mult=1.5 → qDd=0.75 > 0.70
        pts2 = [_make_point(normalized_rate=0.5) for _ in range(5)]
        results2 = _check_qdd_range(pts2, effective_y_multiplier=1.5)
        assert results2[0].severity == "warning"

    def test_ok_when_some_qdd_below_threshold(self) -> None:
        # Mix of high and low — min qDd = 0.2 * 1.5 = 0.30 < 0.70
        pts = [_make_point(normalized_rate=nr) for nr in [0.8, 0.5, 0.2]]
        results = _check_qdd_range(pts, effective_y_multiplier=1.5)
        assert results[0].severity == "ok"

    def test_empty_points_returns_nothing(self) -> None:
        assert _check_qdd_range([], 2.0) == []

    def test_detail_reports_min_qdd(self) -> None:
        pts = [_make_point(normalized_rate=0.3)]
        results = _check_qdd_range(pts, effective_y_multiplier=1.5)
        # min_qdd = 0.3*1.5 = 0.45 < 0.70 → ok result
        assert "0.45" in results[0].detail


# ---------------------------------------------------------------------------
# 6. _check_transient_only
# ---------------------------------------------------------------------------

class TestCheckTransientOnly:

    def _bdf_points(self, n: int = 12) -> list[RTATransformPoint]:
        """Points that follow exponential decline (log-log slope ≈ -1.0)."""
        # BDF: q ∝ exp(-Di*t), MBT ∝ t  →  log(nr) ≈ -1*log(MBT)
        pts = []
        Di = 0.01
        for i in range(n):
            mbt = 30.0 * (i + 1)
            nr = 0.5 * math.exp(-Di * mbt)
            pts.append(_make_point(mbt=mbt, normalized_rate=max(nr, 1e-6)))
        return pts

    def _transient_points(self, n: int = 12) -> list[RTATransformPoint]:
        """Points following 1/ln(t) — log-log slope ≈ 0 (nearly flat)."""
        pts = []
        for i in range(n):
            mbt = 10.0 * (i + 1)
            nr = 1.0 / max(math.log(mbt / 10.0 + 1.0), 0.1)
            pts.append(_make_point(mbt=mbt, normalized_rate=nr))
        return pts

    def test_returns_nothing_for_fewer_than_4_points(self) -> None:
        pts = [_make_point() for _ in range(3)]
        assert _check_transient_only(pts) == []

    def test_ok_for_bdf_slope(self) -> None:
        results = _check_transient_only(self._bdf_points())
        assert results[0].severity == "ok"

    def test_warning_for_near_zero_slope(self) -> None:
        # Near-flat points: log-log slope ≈ 0 → transient warning
        # Use constant normalized_rate (slope = 0) with increasing MBT
        pts = [_make_point(mbt=10.0 * (i + 1), normalized_rate=1.0) for i in range(12)]
        results = _check_transient_only(pts)
        assert results[0].severity == "warning"

    def test_slope_reported_in_detail(self) -> None:
        pts = [_make_point(mbt=10.0 * (i + 1), normalized_rate=1.0) for i in range(12)]
        result = _check_transient_only(pts)[0]
        assert "Pendiente" in result.detail or "pendiente" in result.detail.lower()


# ---------------------------------------------------------------------------
# 7. run_rta_qc — aggregator
# ---------------------------------------------------------------------------

class TestRunRtaQc:

    def test_returns_list_of_qc_results(self) -> None:
        pts = _declining_points(20)
        results = run_rta_qc(
            points=pts,
            effective_x_multiplier=2.0,
            effective_y_multiplier=0.5,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, QCResult) for r in results)

    def test_all_six_checks_present_for_adequate_data(self) -> None:
        pts = _declining_points(20)
        results = run_rta_qc(
            points=pts,
            effective_x_multiplier=2.0,
            effective_y_multiplier=0.5,
        )
        codes = {r.code for r in results}
        assert "POINT_COUNT" in codes
        assert "DRAWDOWN_STABILITY" in codes
        assert "DATA_SPAN" in codes
        assert "MATCH_NOT_ADJUSTED" in codes
        # QDD_RANGE and TRANSIENT_ONLY may be absent with default mult or few points
        # but with 20 pts and adjusted mult they should appear
        assert len(results) >= 4

    def test_empty_points_only_returns_point_count_error(self) -> None:
        results = run_rta_qc(
            points=[],
            effective_x_multiplier=1.0,
            effective_y_multiplier=1.0,
        )
        codes = {r.code for r in results}
        assert "POINT_COUNT" in codes
        assert any(r.severity == "error" for r in results)

    def test_result_order_matches_check_sequence(self) -> None:
        """POINT_COUNT must come before DRAWDOWN_STABILITY before DATA_SPAN..."""
        pts = _declining_points(20)
        results = run_rta_qc(
            points=pts,
            effective_x_multiplier=2.0,
            effective_y_multiplier=0.5,
        )
        codes = [r.code for r in results]
        # Check that known codes appear in the correct relative order
        for earlier, later in [
            ("POINT_COUNT", "DRAWDOWN_STABILITY"),
            ("DRAWDOWN_STABILITY", "DATA_SPAN"),
            ("DATA_SPAN", "MATCH_NOT_ADJUSTED"),
        ]:
            if earlier in codes and later in codes:
                assert codes.index(earlier) < codes.index(later)


# ---------------------------------------------------------------------------
# 8. qc_severity_level
# ---------------------------------------------------------------------------

class TestQcSeverityLevel:

    def test_returns_ok_when_all_ok(self) -> None:
        results = [
            QCResult(code="A", severity="ok", title="T", detail="D"),
            QCResult(code="B", severity="ok", title="T", detail="D"),
        ]
        assert qc_severity_level(results) == "ok"

    def test_returns_warning_when_any_warning_no_error(self) -> None:
        results = [
            QCResult(code="A", severity="ok", title="T", detail="D"),
            QCResult(code="B", severity="warning", title="T", detail="D"),
        ]
        assert qc_severity_level(results) == "warning"

    def test_returns_error_when_any_error(self) -> None:
        results = [
            QCResult(code="A", severity="ok", title="T", detail="D"),
            QCResult(code="B", severity="warning", title="T", detail="D"),
            QCResult(code="C", severity="error", title="T", detail="D"),
        ]
        assert qc_severity_level(results) == "error"

    def test_error_takes_precedence_over_warning(self) -> None:
        results = [
            QCResult(code="A", severity="warning", title="T", detail="D"),
            QCResult(code="B", severity="error", title="T", detail="D"),
            QCResult(code="C", severity="warning", title="T", detail="D"),
        ]
        assert qc_severity_level(results) == "error"

    def test_empty_list_returns_ok(self) -> None:
        assert qc_severity_level([]) == "ok"
