"""Tests for src/services/well_mech_qc_service.py.

Covers all nine QC check functions individually plus the run_mech_qc()
aggregator and mech_severity_level() helper.  No UI or Streamlit code is
exercised — pure logic only.
"""

from __future__ import annotations

import pytest

from src.services.well_mech_qc_service import (
    CasingString,
    MechQCResult,
    TubingString,
    WellMechConfig,
    _check_casing_concentricity,
    _check_casing_shoe_order,
    _check_casing_wall_integrity,
    _check_esp,
    _check_perf_order,
    _check_perfs_within_casing,
    _check_tubing_fits_casing,
    _check_tubing_vs_perfs,
    _check_tubing_wall_integrity,
    mech_severity_level,
    run_mech_qc,
)


# ---------------------------------------------------------------------------
# Fixtures — realistic CPO-9 well
# ---------------------------------------------------------------------------

@pytest.fixture
def surface_casing() -> CasingString:
    return CasingString(name="Superficie", od_in=13.375, id_in=12.415, shoe_depth_ft=2_000.0)


@pytest.fixture
def production_casing() -> CasingString:
    return CasingString(name="Producción", od_in=9.625, id_in=8.681, shoe_depth_ft=8_500.0)


@pytest.fixture
def tubing() -> TubingString:
    return TubingString(od_in=2.875, id_in=2.441, set_depth_ft=7_200.0)


@pytest.fixture
def valid_config(surface_casing, production_casing, tubing) -> WellMechConfig:
    return WellMechConfig(
        well_id="W-001",
        casings=[surface_casing, production_casing],
        tubing=tubing,
        perfs_top_ft=7_400.0,
        perfs_bottom_ft=7_600.0,
        has_esp=True,
        esp_intake_depth_ft=7_000.0,
    )


# ---------------------------------------------------------------------------
# 1. PERF_ORDER
# ---------------------------------------------------------------------------

class TestPerfOrder:

    def test_ok_when_bottom_greater_than_top(self) -> None:
        results = _check_perf_order(7_400.0, 7_600.0)
        assert results[0].severity == "ok"
        assert results[0].code == "PERF_ORDER"

    def test_error_when_bottom_equals_top(self) -> None:
        results = _check_perf_order(7_400.0, 7_400.0)
        assert results[0].severity == "error"

    def test_error_when_bottom_less_than_top(self) -> None:
        results = _check_perf_order(7_600.0, 7_400.0)
        assert results[0].severity == "error"

    def test_detail_mentions_both_depths(self) -> None:
        results = _check_perf_order(8_000.0, 7_000.0)
        assert "8000" in results[0].detail
        assert "7000" in results[0].detail


# ---------------------------------------------------------------------------
# 2. CASING_WALL
# ---------------------------------------------------------------------------

class TestCasingWall:

    def test_ok_for_valid_casing(self, production_casing) -> None:
        results = _check_casing_wall_integrity([production_casing])
        assert results[0].severity == "ok"

    def test_error_when_id_equals_od(self) -> None:
        bad = CasingString("X", od_in=9.625, id_in=9.625, shoe_depth_ft=8_000.0)
        results = _check_casing_wall_integrity([bad])
        assert results[0].severity == "error"

    def test_error_when_id_greater_than_od(self) -> None:
        bad = CasingString("X", od_in=9.0, id_in=9.5, shoe_depth_ft=8_000.0)
        results = _check_casing_wall_integrity([bad])
        assert results[0].severity == "error"

    def test_error_when_id_is_zero(self) -> None:
        bad = CasingString("X", od_in=9.625, id_in=0.0, shoe_depth_ft=8_000.0)
        results = _check_casing_wall_integrity([bad])
        assert results[0].severity == "error"

    def test_returns_one_result_per_casing(self, surface_casing, production_casing) -> None:
        results = _check_casing_wall_integrity([surface_casing, production_casing])
        assert len(results) == 2


# ---------------------------------------------------------------------------
# 3. TUBING_WALL
# ---------------------------------------------------------------------------

class TestTubingWall:

    def test_ok_for_valid_tubing(self, tubing) -> None:
        results = _check_tubing_wall_integrity(tubing)
        assert results[0].severity == "ok"

    def test_error_when_tubing_id_equals_od(self) -> None:
        bad = TubingString(od_in=2.875, id_in=2.875, set_depth_ft=7_200.0)
        results = _check_tubing_wall_integrity(bad)
        assert results[0].severity == "error"

    def test_error_when_tubing_id_is_zero(self) -> None:
        bad = TubingString(od_in=2.875, id_in=0.0, set_depth_ft=7_200.0)
        results = _check_tubing_wall_integrity(bad)
        assert results[0].severity == "error"


# ---------------------------------------------------------------------------
# 4. CASING_SHOE_ORDER
# ---------------------------------------------------------------------------

class TestCasingShoeOrder:

    def test_ok_when_inner_deeper_than_outer(
        self, surface_casing, production_casing
    ) -> None:
        results = _check_casing_shoe_order([surface_casing, production_casing])
        assert results[0].severity == "ok"

    def test_error_when_inner_shoe_shallower_than_outer(self) -> None:
        outer = CasingString("Superficie", od_in=13.375, id_in=12.415, shoe_depth_ft=5_000.0)
        inner = CasingString("Produccion", od_in=9.625, id_in=8.681, shoe_depth_ft=4_000.0)
        results = _check_casing_shoe_order([outer, inner])
        assert results[0].severity == "error"

    def test_error_when_inner_shoe_equals_outer(self) -> None:
        outer = CasingString("Superficie", od_in=13.375, id_in=12.415, shoe_depth_ft=5_000.0)
        inner = CasingString("Produccion", od_in=9.625, id_in=8.681, shoe_depth_ft=5_000.0)
        results = _check_casing_shoe_order([outer, inner])
        assert results[0].severity == "error"

    def test_single_casing_returns_no_results(self, surface_casing) -> None:
        results = _check_casing_shoe_order([surface_casing])
        assert results == []


# ---------------------------------------------------------------------------
# 5. CASING_CONCENTRICITY
# ---------------------------------------------------------------------------

class TestCasingConcentricity:

    def test_ok_when_inner_od_clearly_smaller_than_outer_id(
        self, surface_casing, production_casing
    ) -> None:
        # Surface ID=12.415", Production OD=9.625" → clearance 2.79" >> 0.25"
        results = _check_casing_concentricity([surface_casing, production_casing])
        assert results[0].severity == "ok"

    def test_error_when_inner_od_equals_outer_id(self) -> None:
        outer = CasingString("Sup", od_in=13.375, id_in=9.625, shoe_depth_ft=2_000.0)
        inner = CasingString("Pro", od_in=9.625, id_in=8.681, shoe_depth_ft=8_000.0)
        results = _check_casing_concentricity([outer, inner])
        assert results[0].severity == "error"

    def test_error_when_inner_od_greater_than_outer_id(self) -> None:
        outer = CasingString("Sup", od_in=13.375, id_in=9.0, shoe_depth_ft=2_000.0)
        inner = CasingString("Pro", od_in=9.625, id_in=8.681, shoe_depth_ft=8_000.0)
        results = _check_casing_concentricity([outer, inner])
        assert results[0].severity == "error"

    def test_warning_when_clearance_less_than_threshold(self) -> None:
        # outer ID = 9.625, inner OD = 9.500 → clearance = 0.125 < 0.25" threshold
        outer = CasingString("Sup", od_in=11.0, id_in=9.625, shoe_depth_ft=2_000.0)
        inner = CasingString("Pro", od_in=9.500, id_in=8.681, shoe_depth_ft=8_000.0)
        results = _check_casing_concentricity([outer, inner])
        assert results[0].severity == "warning"

    def test_single_casing_returns_empty(self, surface_casing) -> None:
        results = _check_casing_concentricity([surface_casing])
        assert results == []


# ---------------------------------------------------------------------------
# 6. TUBING_FITS_CASING
# ---------------------------------------------------------------------------

class TestTubingFitsCasing:

    def test_ok_when_tubing_clearly_fits(self, tubing, production_casing) -> None:
        # Tubing OD 2.875" << Prod casing ID 8.681"
        results = _check_tubing_fits_casing(tubing, production_casing)
        assert results[0].severity == "ok"

    def test_error_when_tubing_od_equals_casing_id(self, production_casing) -> None:
        big_tubing = TubingString(od_in=8.681, id_in=7.0, set_depth_ft=7_200.0)
        results = _check_tubing_fits_casing(big_tubing, production_casing)
        assert results[0].severity == "error"

    def test_error_when_tubing_od_greater_than_casing_id(self, production_casing) -> None:
        big_tubing = TubingString(od_in=9.0, id_in=8.0, set_depth_ft=7_200.0)
        results = _check_tubing_fits_casing(big_tubing, production_casing)
        assert results[0].severity == "error"

    def test_warning_when_clearance_below_threshold(self, production_casing) -> None:
        # Casing ID = 8.681", tubing OD = 8.625" → clearance = 0.056" < 0.125"
        tight_tubing = TubingString(od_in=8.625, id_in=7.5, set_depth_ft=7_200.0)
        results = _check_tubing_fits_casing(tight_tubing, production_casing)
        assert results[0].severity == "warning"

    def test_returns_empty_when_no_casing(self, tubing) -> None:
        results = _check_tubing_fits_casing(tubing, None)
        assert results == []


# ---------------------------------------------------------------------------
# 7. PERFS_WITHIN_CASING
# ---------------------------------------------------------------------------

class TestPerfsWithinCasing:

    def test_ok_when_perfs_well_within_casing(self, production_casing) -> None:
        # Shoe at 8500, perfs at 7400-7600 → 900 ft above shoe
        results = _check_perfs_within_casing(7_400.0, 7_600.0, production_casing)
        assert results[0].severity == "ok"

    def test_error_when_perfs_bottom_below_shoe(self, production_casing) -> None:
        # Shoe at 8500, perfs bottom at 8700 → error
        results = _check_perfs_within_casing(8_400.0, 8_700.0, production_casing)
        assert results[0].severity == "error"

    def test_warning_when_perfs_bottom_near_shoe(self, production_casing) -> None:
        # Shoe at 8500, perfs bottom at 8450 → within 100 ft → warning
        results = _check_perfs_within_casing(8_300.0, 8_450.0, production_casing)
        assert results[0].severity == "warning"

    def test_returns_empty_when_no_casing(self) -> None:
        results = _check_perfs_within_casing(7_000.0, 7_200.0, None)
        assert results == []


# ---------------------------------------------------------------------------
# 8. TUBING_VS_PERFS
# ---------------------------------------------------------------------------

class TestTubingVsPerfs:

    def test_ok_when_tubing_just_above_perfs(self, tubing) -> None:
        # Tubing shoe 7200, perfs_top 7400 → 200 ft above → ok
        results = _check_tubing_vs_perfs(tubing, 7_400.0, 7_600.0)
        assert results[0].severity == "ok"

    def test_warning_when_tubing_too_far_above_perfs(self, tubing) -> None:
        # Tubing shoe 7200, perfs_top 8000 → 800 ft above > 500 ft threshold
        results = _check_tubing_vs_perfs(tubing, 8_000.0, 8_200.0)
        assert results[0].severity == "warning"

    def test_warning_when_tubing_set_inside_perf_interval(self, tubing) -> None:
        # Tubing shoe 7200, perfs 7000-7400 → tubing inside perfs
        results = _check_tubing_vs_perfs(tubing, 7_000.0, 7_400.0)
        assert results[0].severity == "warning"

    def test_warning_when_tubing_set_below_all_perfs(self, tubing) -> None:
        # Tubing shoe 7200, perfs 6000-6500 → tubing below perfs
        results = _check_tubing_vs_perfs(tubing, 6_000.0, 6_500.0)
        assert results[0].severity == "warning"


# ---------------------------------------------------------------------------
# 9. ESP_CONFIG
# ---------------------------------------------------------------------------

class TestEspConfig:

    def test_returns_empty_when_no_esp(self, tubing) -> None:
        results = _check_esp(False, None, tubing.set_depth_ft, 7_400.0)
        assert results == []

    def test_warning_when_esp_enabled_but_no_depth(self, tubing) -> None:
        results = _check_esp(True, None, tubing.set_depth_ft, 7_400.0)
        assert results[0].severity == "warning"

    def test_error_when_esp_intake_below_tubing_shoe(self, tubing) -> None:
        # Tubing shoe 7200, ESP intake 7500 → error
        results = _check_esp(True, 7_500.0, tubing.set_depth_ft, 7_400.0)
        assert results[0].severity == "error"

    def test_warning_when_esp_intake_in_perf_interval(self, tubing) -> None:
        # Tubing shoe 7200, ESP intake 7050, perfs_top 7000 → intake >= perfs_top
        results = _check_esp(True, 7_050.0, tubing.set_depth_ft, 7_000.0)
        assert results[0].severity == "warning"

    def test_ok_when_esp_above_perfs_inside_tubing(self, tubing) -> None:
        # Tubing shoe 7200, ESP intake 7000, perfs_top 7400 → ok
        results = _check_esp(True, 7_000.0, tubing.set_depth_ft, 7_400.0)
        assert results[0].severity == "ok"


# ---------------------------------------------------------------------------
# 10. run_mech_qc — aggregator
# ---------------------------------------------------------------------------

class TestRunMechQc:

    def test_all_ok_for_valid_configuration(self, valid_config) -> None:
        results = run_mech_qc(valid_config)
        for r in results:
            assert r.severity in ("ok", "warning", "error")
        assert all(r.severity == "ok" for r in results), (
            "\n".join(f"{r.code}: {r.severity} — {r.title}" for r in results
                      if r.severity != "ok")
        )

    def test_returns_list_of_mech_qc_result(self, valid_config) -> None:
        results = run_mech_qc(valid_config)
        assert isinstance(results, list)
        assert all(isinstance(r, MechQCResult) for r in results)

    def test_error_detected_for_tubing_wider_than_casing(
        self, surface_casing, production_casing
    ) -> None:
        fat_tubing = TubingString(od_in=9.625, id_in=8.5, set_depth_ft=7_200.0)
        cfg = WellMechConfig(
            well_id="BAD",
            casings=[surface_casing, production_casing],
            tubing=fat_tubing,
            perfs_top_ft=7_400.0,
            perfs_bottom_ft=7_600.0,
        )
        results = run_mech_qc(cfg)
        codes = {r.code for r in results if r.severity == "error"}
        assert "TUBING_FITS_CASING" in codes

    def test_error_detected_for_perfs_below_shoe(
        self, surface_casing, production_casing, tubing
    ) -> None:
        cfg = WellMechConfig(
            well_id="BAD",
            casings=[surface_casing, production_casing],
            tubing=tubing,
            perfs_top_ft=8_600.0,   # below production casing shoe at 8500
            perfs_bottom_ft=8_800.0,
        )
        results = run_mech_qc(cfg)
        codes = {r.code for r in results if r.severity == "error"}
        assert "PERFS_WITHIN_CASING" in codes

    def test_error_detected_for_inverted_perf_interval(
        self, surface_casing, production_casing, tubing
    ) -> None:
        cfg = WellMechConfig(
            well_id="BAD",
            casings=[surface_casing, production_casing],
            tubing=tubing,
            perfs_top_ft=7_600.0,
            perfs_bottom_ft=7_400.0,   # inverted!
        )
        results = run_mech_qc(cfg)
        assert any(r.code == "PERF_ORDER" and r.severity == "error" for r in results)


# ---------------------------------------------------------------------------
# 11. mech_severity_level
# ---------------------------------------------------------------------------

class TestMechSeverityLevel:

    def test_ok_when_all_ok(self) -> None:
        r = [MechQCResult("A", "ok", "T", "D"), MechQCResult("B", "ok", "T", "D")]
        assert mech_severity_level(r) == "ok"

    def test_warning_when_any_warning_no_error(self) -> None:
        r = [
            MechQCResult("A", "ok", "T", "D"),
            MechQCResult("B", "warning", "T", "D"),
        ]
        assert mech_severity_level(r) == "warning"

    def test_error_takes_precedence(self) -> None:
        r = [
            MechQCResult("A", "warning", "T", "D"),
            MechQCResult("B", "error", "T", "D"),
        ]
        assert mech_severity_level(r) == "error"

    def test_empty_list_returns_ok(self) -> None:
        assert mech_severity_level([]) == "ok"


# ---------------------------------------------------------------------------
# 12. WellMechConfig helpers
# ---------------------------------------------------------------------------

class TestWellMechConfigHelpers:

    def test_innermost_casing_returns_last(
        self, surface_casing, production_casing, tubing
    ) -> None:
        cfg = WellMechConfig(
            well_id="W",
            casings=[surface_casing, production_casing],
            tubing=tubing,
            perfs_top_ft=7_400.0,
            perfs_bottom_ft=7_600.0,
        )
        assert cfg.innermost_casing is production_casing

    def test_innermost_casing_returns_none_when_no_casings(self, tubing) -> None:
        cfg = WellMechConfig(
            well_id="W",
            casings=[],
            tubing=tubing,
            perfs_top_ft=7_400.0,
            perfs_bottom_ft=7_600.0,
        )
        assert cfg.innermost_casing is None

    def test_effective_total_depth_uses_perfs_bottom_plus_margin(
        self, surface_casing, production_casing, tubing
    ) -> None:
        cfg = WellMechConfig(
            well_id="W",
            casings=[surface_casing, production_casing],
            tubing=tubing,
            perfs_top_ft=7_400.0,
            perfs_bottom_ft=7_600.0,
        )
        td = cfg.effective_total_depth
        assert td >= 7_600.0
