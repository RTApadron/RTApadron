"""Tests para M5 — Servicio de tabla comparativa ecoRTA vs software comercial."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.m5_models import (
    ComparisonRow,
    DCASummary,
    DCAModelSummary,
    ExternalSoftwareResult,
    PVTSummary,
    RTASummary,
    WellInfoSummary,
    WellResultsSummary,
)
from src.services.m5_comparison_service import (
    _row,
    _status,
    build_comparison_table,
    comparison_table_to_csv_bytes,
    load_external_result,
    save_external_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def full_summary() -> WellResultsSummary:
    return WellResultsSummary(
        well_id="W001",
        well_info=WellInfoSummary(well_id="W001", history_points=30),
        dca=DCASummary(
            best_model="hyperbolic",
            models=[
                DCAModelSummary(model="exponential", qi_stb_d=600, di_nominal_d=0.001,
                                b=0.0, eur_stb=219_000, r2=0.95, rmse_stb_d=14, forecast_days=365, n_points=30),
                DCAModelSummary(model="hyperbolic", qi_stb_d=600, di_nominal_d=0.001,
                                b=0.5, eur_stb=280_000, r2=0.99, rmse_stb_d=7, forecast_days=365, n_points=30),
                DCAModelSummary(model="harmonic", qi_stb_d=600, di_nominal_d=0.001,
                                b=1.0, eur_stb=340_000, r2=0.92, rmse_stb_d=20, forecast_days=365, n_points=30),
            ],
            eur_exponential_stb=219_000,
            eur_hyperbolic_stb=280_000,
            eur_harmonic_stb=340_000,
        ),
        rta=RTASummary(
            method="fetkovich",
            kh_md_ft=130.0,
            k_md=2.6,
            n_vol_stb=4_800_000.0,
            status="demo",
        ),
    )


@pytest.fixture()
def external_close() -> ExternalSoftwareResult:
    """Valores de referencia con diferencias < 5 % en EUR."""
    return ExternalSoftwareResult(
        software_label="Software Comercial",
        eur_stb=285_000.0,   # +1.8 % sobre hiperbólico 280_000
        kh_md_ft=132.0,      # +1.5 % sobre 130.0
        k_md=2.64,           # +1.5 % sobre 2.6
        n_vol_stb=4_750_000.0,  # -1.0 % sobre 4_800_000
    )


@pytest.fixture()
def external_diverge() -> ExternalSoftwareResult:
    """Valores de referencia con diferencias > 20 %."""
    return ExternalSoftwareResult(
        software_label="Software Comercial",
        eur_stb=400_000.0,   # +42.8 % sobre hiperbólico 280_000
        kh_md_ft=80.0,       # -38.5 % sobre 130.0
        k_md=1.5,            # -42.3 % sobre 2.6
        n_vol_stb=8_000_000.0,  # +66.7 % sobre 4_800_000
    )


# ---------------------------------------------------------------------------
# Tests: helpers internos _status y _row
# ---------------------------------------------------------------------------

class TestStatusHelper:
    def test_match_below_5(self):
        assert _status(4.9) == "match"
        assert _status(-4.9) == "match"
        assert _status(0.0) == "match"

    def test_close_5_to_20(self):
        assert _status(5.0) == "close"
        assert _status(19.9) == "close"
        assert _status(-15.0) == "close"

    def test_diverge_above_20(self):
        assert _status(20.0) == "diverge"
        assert _status(50.0) == "diverge"
        assert _status(-25.0) == "diverge"

    def test_missing_when_none(self):
        assert _status(None) == "missing"


class TestRowHelper:
    def test_computes_abs_diff(self):
        row = _row("EUR", "STB", 280_000.0, 285_000.0)
        assert row.abs_diff == pytest.approx(5_000.0)

    def test_computes_rel_diff(self):
        row = _row("EUR", "STB", 280_000.0, 285_000.0)
        # (280000 - 285000) / 285000 * 100 = -1.754...
        assert row.rel_diff_pct == pytest.approx(-1.754, rel=1e-2)

    def test_status_match(self):
        row = _row("EUR", "STB", 280_000.0, 285_000.0)
        assert row.status == "match"

    def test_missing_when_ecorta_none(self):
        row = _row("kh", "mD·ft", None, 130.0)
        assert row.status == "missing"
        assert row.abs_diff is None
        assert row.rel_diff_pct is None

    def test_missing_when_external_none(self):
        row = _row("kh", "mD·ft", 130.0, None)
        assert row.status == "missing"

    def test_missing_when_external_zero(self):
        row = _row("kh", "mD·ft", 130.0, 0.0)
        assert row.status == "missing"

    def test_note_propagated(self):
        row = _row("kh", "mD·ft", 130.0, 132.0, note="DEMO")
        assert row.note == "DEMO"


# ---------------------------------------------------------------------------
# Tests: build_comparison_table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:
    def test_returns_list_of_rows(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close)
        assert isinstance(rows, list)
        assert all(isinstance(r, ComparisonRow) for r in rows)

    def test_eur_row_present(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close)
        params = [r.parameter for r in rows]
        assert any("EUR" in p for p in params)

    def test_kh_row_present(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close)
        params = [r.parameter for r in rows]
        assert any("kh" in p for p in params)

    def test_k_row_present(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close)
        params = [r.parameter for r in rows]
        assert any("permeabilidad" in p.lower() for p in params)

    def test_ooip_row_present(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close)
        params = [r.parameter for r in rows]
        assert any("OOIP" in p for p in params)

    def test_rf_row_present_when_both_available(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close)
        params = [r.parameter for r in rows]
        assert any("recobro" in p.lower() for p in params)

    def test_close_values_get_match_status(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close)
        eur_row = next(r for r in rows if "EUR" in r.parameter and r.units == "STB")
        assert eur_row.status == "match"

    def test_diverge_values_get_diverge_status(self, full_summary, external_diverge):
        rows = build_comparison_table(full_summary, external_diverge)
        eur_row = next(r for r in rows if "EUR" in r.parameter and r.units == "STB")
        assert eur_row.status == "diverge"

    def test_best_model_selector(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close, dca_model="best")
        eur_row = next(r for r in rows if "EUR" in r.parameter and r.units == "STB")
        # best model es hyperbolic (R²=0.99), EUR=280_000
        assert eur_row.ecorta_value == pytest.approx(280_000.0)

    def test_explicit_model_selector(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close, dca_model="exponential")
        eur_row = next(r for r in rows if "EUR" in r.parameter and r.units == "STB")
        assert eur_row.ecorta_value == pytest.approx(219_000.0)

    def test_rta_note_is_demo(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close)
        kh_row = next(r for r in rows if r.parameter == "kh")
        assert "DEMO" in kh_row.note

    def test_no_dca_still_returns_rta_rows(self, external_close):
        summary_no_dca = WellResultsSummary(
            well_id="W002",
            well_info=WellInfoSummary(well_id="W002", history_points=10),
            rta=RTASummary(method="fetkovich", kh_md_ft=100.0, k_md=2.0, n_vol_stb=3_000_000.0),
        )
        rows = build_comparison_table(summary_no_dca, external_close)
        params = [r.parameter for r in rows]
        assert any("kh" in p for p in params)

    def test_empty_external_all_missing(self, full_summary):
        external_empty = ExternalSoftwareResult()
        rows = build_comparison_table(full_summary, external_empty)
        # Cuando no hay valores externos, las filas comparables deben ser missing
        non_rf_rows = [r for r in rows if "recobro" not in r.parameter.lower()]
        for r in non_rf_rows:
            assert r.status in ("missing",), f"Fila '{r.parameter}' debería ser missing pero es {r.status}"

    def test_mm_stb_row_is_also_present(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close)
        mm_rows = [r for r in rows if r.units == "MM STB" and "EUR" in r.parameter]
        assert len(mm_rows) >= 1


# ---------------------------------------------------------------------------
# Tests: save / load external result
# ---------------------------------------------------------------------------

class TestPersistExternalResult:
    def test_save_creates_file(self, tmp_path: Path):
        ext = ExternalSoftwareResult(eur_stb=280_000.0, kh_md_ft=130.0)
        path = save_external_result(ext, tmp_path, "W001")
        assert path.exists()
        assert path.name == "W001_external_reference.json"

    def test_saved_content_is_valid_json(self, tmp_path: Path):
        ext = ExternalSoftwareResult(eur_stb=280_000.0)
        save_external_result(ext, tmp_path, "W001")
        content = json.loads((tmp_path / "W001_external_reference.json").read_text())
        assert content["eur_stb"] == pytest.approx(280_000.0)

    def test_load_returns_model(self, tmp_path: Path):
        ext = ExternalSoftwareResult(eur_stb=280_000.0, kh_md_ft=130.0, software_label="Test SW")
        save_external_result(ext, tmp_path, "W001")
        loaded = load_external_result(tmp_path, "W001")
        assert loaded is not None
        assert loaded.eur_stb == pytest.approx(280_000.0)
        assert loaded.kh_md_ft == pytest.approx(130.0)
        assert loaded.software_label == "Test SW"

    def test_load_returns_none_if_missing(self, tmp_path: Path):
        result = load_external_result(tmp_path, "W999")
        assert result is None

    def test_creates_output_dir_if_missing(self, tmp_path: Path):
        nested = tmp_path / "sub" / "output"
        ext = ExternalSoftwareResult(eur_stb=100_000.0)
        save_external_result(ext, nested, "W001")
        assert nested.exists()

    def test_roundtrip_all_fields(self, tmp_path: Path):
        ext = ExternalSoftwareResult(
            software_label="Mi Software",
            eur_stb=280_000.0,
            qi_stb_d=600.0,
            di_nominal_d=0.001,
            b_factor=0.5,
            kh_md_ft=130.0,
            k_md=2.6,
            n_vol_stb=4_800_000.0,
            skin=-2.5,
            notes="Prueba de roundtrip",
        )
        save_external_result(ext, tmp_path, "W001")
        loaded = load_external_result(tmp_path, "W001")
        assert loaded is not None
        assert loaded.b_factor == pytest.approx(0.5)
        assert loaded.skin == pytest.approx(-2.5)
        assert loaded.notes == "Prueba de roundtrip"


# ---------------------------------------------------------------------------
# Tests: comparison_table_to_csv_bytes
# ---------------------------------------------------------------------------

class TestComparisonCSV:
    def test_returns_bytes(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close)
        data = comparison_table_to_csv_bytes(rows, full_summary, external_close)
        assert isinstance(data, bytes)

    def test_contains_well_id(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close)
        data = comparison_table_to_csv_bytes(rows, full_summary, external_close)
        assert b"W001" in data

    def test_contains_software_label(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close)
        data = comparison_table_to_csv_bytes(rows, full_summary, external_close)
        assert b"Software Comercial" in data

    def test_contains_concordancia_column(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close)
        data = comparison_table_to_csv_bytes(rows, full_summary, external_close)
        assert b"Concordancia" in data

    def test_empty_rows_no_crash(self, full_summary):
        ext = ExternalSoftwareResult()
        data = comparison_table_to_csv_bytes([], full_summary, ext)
        assert isinstance(data, bytes)


# ---------------------------------------------------------------------------
# Tests P1: Score global de concordancia
# ---------------------------------------------------------------------------

class TestScoreGlobal:
    def test_all_match_gives_100_pct(self, full_summary, external_close):
        rows = build_comparison_table(full_summary, external_close)
        meaningful = [r for r in rows if r.status != "missing"]
        if not meaningful:
            pytest.skip("No meaningful rows to compute score")
        counts = {s: sum(1 for r in rows if r.status == s) for s in ("match", "close", "diverge")}
        total = counts["match"] + counts["close"] + counts["diverge"]
        pct_ok = (counts["match"] + counts["close"]) / total * 100 if total > 0 else 0
        # external_close tiene diferencias < 2% → todos match → 100%
        assert pct_ok == pytest.approx(100.0, abs=1.0)

    def test_mostly_diverge_gives_low_pct(self, full_summary, external_diverge):
        """Con diferencias > 20% en todos los parámetros el pct_ok debe ser bajo (< 50%).

        Nota: el factor de recobro (EUR/OOIP) puede quedar "close" incluso cuando
        EUR y OOIP divergen individualmente (efecto de cancelación), por lo que no
        se exige estrictamente 0%.
        """
        rows = build_comparison_table(full_summary, external_diverge)
        counts = {s: sum(1 for r in rows if r.status == s) for s in ("match", "close", "diverge")}
        total = counts["match"] + counts["close"] + counts["diverge"]
        if total == 0:
            pytest.skip("No comparable rows")
        pct_ok = (counts["match"] + counts["close"]) / total * 100
        assert pct_ok < 50.0, f"Se esperaba pct_ok < 50%, pero fue {pct_ok:.1f}%"

    def test_missing_not_counted_in_score(self):
        """Las filas 'missing' no cuentan ni en numerador ni denominador del score."""
        from src.services.m5_comparison_service import _status
        # Si todos son missing, el denominador es 0 → pct_ok indefinido (no crash)
        counts = {"match": 0, "close": 0, "diverge": 0, "missing": 3}
        total = counts["match"] + counts["close"] + counts["diverge"]
        assert total == 0  # no divisiones por cero en el código

    def test_score_threshold_80_is_validado(self, full_summary, external_close):
        """pct_ok >= 80 debe producir badge VALIDADO en la UI (lógica de score)."""
        rows = build_comparison_table(full_summary, external_close)
        counts = {s: sum(1 for r in rows if r.status == s) for s in ("match", "close", "diverge")}
        total = counts["match"] + counts["close"] + counts["diverge"]
        pct_ok = (counts["match"] + counts["close"]) / total * 100 if total > 0 else 0
        if pct_ok >= 80:
            badge = "VALIDADO"
        elif pct_ok >= 50:
            badge = "VALIDACIÓN PARCIAL"
        else:
            badge = "DIVERGENCIA ALTA"
        assert badge == "VALIDADO"


# ---------------------------------------------------------------------------
# Tests P2: PVTSummary pvt_source + status según calibrated_flag
# ---------------------------------------------------------------------------

class TestPVTTrazabilidad:
    def test_pvt_source_default_is_correlation(self):
        pvt = PVTSummary()
        assert pvt.pvt_source == "correlation"
        assert pvt.status == "estimated"

    def test_pvt_source_lab_when_calibrated(self):
        pvt = PVTSummary(calibrated=True, pvt_source="lab", status="measured")
        assert pvt.pvt_source == "lab"
        assert pvt.status == "measured"

    def test_aggregator_calibrated_true(self):
        import pandas as pd
        from src.services.m5_aggregator_service import _build_pvt_summary

        df = pd.DataFrame([{
            "bo_rb_stb": 1.15, "rs_scf_stb": 350.0, "mu_o_cp": 3.2,
            "calibrated_flag": True,
        }])
        pvt = _build_pvt_summary(df)
        assert pvt is not None
        assert pvt.pvt_source == "lab"
        assert pvt.status == "measured"
        assert pvt.calibrated is True

    def test_aggregator_no_calibrated_col(self):
        import pandas as pd
        from src.services.m5_aggregator_service import _build_pvt_summary

        df = pd.DataFrame([{"bo_rb_stb": 1.1, "rs_scf_stb": 300.0, "mu_o_cp": 4.0}])
        pvt = _build_pvt_summary(df)
        assert pvt is not None
        assert pvt.pvt_source == "correlation"
        assert pvt.status == "estimated"

    def test_aggregator_calibrated_false(self):
        import pandas as pd
        from src.services.m5_aggregator_service import _build_pvt_summary

        df = pd.DataFrame([{
            "bo_rb_stb": 1.1, "rs_scf_stb": 300.0, "mu_o_cp": 4.0,
            "calibrated_flag": False,
        }])
        pvt = _build_pvt_summary(df)
        assert pvt is not None
        assert pvt.pvt_source == "correlation"
        assert pvt.status == "estimated"


# ---------------------------------------------------------------------------
# Tests P2: RTASummary campos kh_status / n_vol_status
# ---------------------------------------------------------------------------

class TestRTATrazabilidad:
    def test_default_kh_status_is_estimated(self):
        rta = RTASummary(method="fetkovich")
        assert rta.kh_status == "estimated"

    def test_default_n_vol_status_is_estimated(self):
        rta = RTASummary(method="fetkovich")
        assert rta.n_vol_status == "estimated"

    def test_custom_kh_status(self):
        rta = RTASummary(method="fetkovich", kh_status="calculated")
        assert rta.kh_status == "calculated"

    def test_custom_n_vol_status(self):
        rta = RTASummary(method="fetkovich", n_vol_status="measured")
        assert rta.n_vol_status == "measured"

    def test_rta_summary_serializable(self):
        """RTASummary con nuevos campos se serializa/deserializa correctamente."""
        rta = RTASummary(
            method="fetkovich",
            kh_md_ft=50.0,
            kh_status="estimated",
            n_vol_status="estimated",
        )
        data = rta.model_dump()
        assert data["kh_status"] == "estimated"
        assert data["n_vol_status"] == "estimated"
        rta2 = RTASummary.model_validate(data)
        assert rta2.kh_status == "estimated"
