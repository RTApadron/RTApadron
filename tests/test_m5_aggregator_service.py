"""Tests para M5 — Agregador de resultados y modelo unificado.

Valida que build_well_results() construya WellResultsSummary correctamente
a partir de artefactos de output/ reales o sintéticos.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.domain.m5_models import (
    DCASummary,
    DCAModelSummary,
    PVTSummary,
    RTASummary,
    WellInfoSummary,
    WellResultsSummary,
)
from src.services.m5_aggregator_service import (
    build_well_results,
    _build_dca_summary,
    _build_pvt_summary,
    _build_rta_summary,
    _build_well_info,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_history_df() -> pd.DataFrame:
    """Historia mínima enriquecida con M1+M2."""
    return pd.DataFrame({
        "well_id": ["W001"] * 5,
        "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"]),
        "qo_stb_d": [500.0, 480.0, 460.0, 440.0, 420.0],
        "pwf_used_psia": [2800.0, 2750.0, 2700.0, 2650.0, 2600.0],
        "pwf_source": ["measured", "measured", "estimated", "estimated", "estimated"],
        "bo": [1.20, 1.20, 1.21, 1.21, 1.21],
        "rs": [250.0, 248.0, 246.0, 244.0, 242.0],
        "mu_o_cp": [2.0, 2.0, 2.01, 2.01, 2.01],
        "rho_o_lbft3": [52.0, 52.0, 52.1, 52.1, 52.1],
        "pb_psia": [3200.0] * 5,
        "oil_corr": ["standing"] * 5,
        "calibrated_flag": [False] * 5,
    })


@pytest.fixture()
def minimal_dca_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"well_id": "W001", "model": "exponential", "qi_stb_d": 500.0,
         "forecast_qi_stb_d": 500.0, "di_nominal_d": 0.001, "b": 0.0,
         "rmse_stb_d": 12.0, "r2": 0.97, "eur_stb": 182500.0,
         "forecast_days": 365, "n_points": 5},
        {"well_id": "W001", "model": "hyperbolic", "qi_stb_d": 500.0,
         "forecast_qi_stb_d": 500.0, "di_nominal_d": 0.001, "b": 0.5,
         "rmse_stb_d": 8.0, "r2": 0.99, "eur_stb": 210000.0,
         "forecast_days": 365, "n_points": 5},
        {"well_id": "W001", "model": "harmonic", "qi_stb_d": 500.0,
         "forecast_qi_stb_d": 500.0, "di_nominal_d": 0.001, "b": 1.0,
         "rmse_stb_d": 15.0, "r2": 0.94, "eur_stb": 250000.0,
         "forecast_days": 365, "n_points": 5},
    ])


@pytest.fixture()
def minimal_rta_json() -> dict:
    return {
        "method": "fetkovich",
        "match_params": {
            "method": "fetkovich",
            "kh_md_ft": 125.5,
            "k_md": 2.51,
            "n_vol_stb": 5_000_000.0,
            "re_ft": 1320.0,
            "area_acres": 125.66,
            "effective_x_multiplier": 0.85,
            "effective_y_multiplier": 1.15,
        },
        "qc_warnings": ["Drawdown inestable — CV(Δp) > 15%"],
    }


@pytest.fixture()
def output_dir_with_files(
    tmp_path: Path,
    minimal_history_df: pd.DataFrame,
    minimal_dca_df: pd.DataFrame,
    minimal_rta_json: dict,
) -> Path:
    """Directorio temporal con todos los artefactos de M1-M4."""
    minimal_history_df.to_csv(tmp_path / "W001_history_enriched.csv", index=False)
    minimal_dca_df.to_csv(tmp_path / "W001_dca_fit_results.csv", index=False)
    with open(tmp_path / "W001_rta_match_summary.json", "w") as f:
        json.dump(minimal_rta_json, f)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: modelos de dominio M5
# ---------------------------------------------------------------------------

class TestWellResultsSummaryModel:
    def test_completeness_flags_all_present(self):
        summary = WellResultsSummary(
            well_id="W001",
            well_info=WellInfoSummary(well_id="W001", history_points=10),
            pvt=PVTSummary(oil_corr="standing"),
            dca=DCASummary(models=[DCAModelSummary(
                model="exponential", qi_stb_d=500, di_nominal_d=0.001,
                b=0, eur_stb=182500, r2=0.97, rmse_stb_d=12, forecast_days=365, n_points=5,
            )]),
            rta=RTASummary(method="fetkovich"),
        )
        flags = summary.completeness_flags()
        assert flags["M1_historia"] is True
        assert flags["M2_PVT"] is True
        assert flags["M3_DCA"] is True
        assert flags["M4_RTA"] is True

    def test_completeness_flags_empty(self):
        summary = WellResultsSummary(
            well_id="W001",
            well_info=WellInfoSummary(well_id="W001", history_points=0),
        )
        flags = summary.completeness_flags()
        assert flags["M1_historia"] is False
        assert flags["M2_PVT"] is False
        assert flags["M3_DCA"] is False
        assert flags["M4_RTA"] is False

    def test_dca_best_eur(self):
        dca = DCASummary(models=[
            DCAModelSummary(model="exponential", qi_stb_d=500, di_nominal_d=0.001,
                            b=0, eur_stb=100_000, r2=0.90, rmse_stb_d=15, forecast_days=365, n_points=5),
            DCAModelSummary(model="hyperbolic", qi_stb_d=500, di_nominal_d=0.001,
                            b=0.5, eur_stb=200_000, r2=0.99, rmse_stb_d=5, forecast_days=365, n_points=5),
        ])
        assert dca.best_eur_stb() == 200_000

    def test_dca_best_eur_none_when_empty(self):
        dca = DCASummary()
        assert dca.best_eur_stb() is None


# ---------------------------------------------------------------------------
# Tests: _build_well_info
# ---------------------------------------------------------------------------

class TestBuildWellInfo:
    def test_basic_fields(self, minimal_history_df: pd.DataFrame):
        info = _build_well_info(minimal_history_df, {}, "W001")
        assert info.well_id == "W001"
        assert info.history_points == 5
        assert info.date_from == date(2023, 1, 1)
        assert info.date_to == date(2023, 5, 1)

    def test_qo_stats(self, minimal_history_df: pd.DataFrame):
        info = _build_well_info(minimal_history_df, {}, "W001")
        assert info.qo_avg_stb_d == pytest.approx(460.0, rel=1e-3)
        assert info.qo_max_stb_d == pytest.approx(500.0, rel=1e-3)

    def test_pwf_source_counts(self, minimal_history_df: pd.DataFrame):
        info = _build_well_info(minimal_history_df, {}, "W001")
        assert info.pwf_source_counts.get("measured", 0) == 2
        assert info.pwf_source_counts.get("estimated", 0) == 3

    def test_empty_df(self):
        info = _build_well_info(pd.DataFrame(), {}, "W001")
        assert info.history_points == 0
        assert info.date_from is None

    def test_warnings_from_qc_report(self, minimal_history_df: pd.DataFrame):
        qc = {"warnings": ["Test warning 1", "Test warning 2"]}
        info = _build_well_info(minimal_history_df, qc, "W001")
        assert len(info.qc_warnings) == 2
        assert "Test warning 1" in info.qc_warnings

    def test_api_and_temp_from_qc_report(self, minimal_history_df: pd.DataFrame):
        qc = {"api_gravity": 25.5, "temp_f": 185.0}
        info = _build_well_info(minimal_history_df, qc, "W001")
        assert info.api_gravity == pytest.approx(25.5)
        assert info.temp_f == pytest.approx(185.0)


# ---------------------------------------------------------------------------
# Tests: _build_pvt_summary
# ---------------------------------------------------------------------------

class TestBuildPVTSummary:
    def test_extracts_pvt_values(self, minimal_history_df: pd.DataFrame):
        pvt = _build_pvt_summary(minimal_history_df)
        assert pvt is not None
        assert pvt.avg_bo_rb_stb == pytest.approx(1.206, rel=1e-2)
        assert pvt.avg_rs_scf_stb == pytest.approx(246.0, rel=1e-1)
        assert pvt.avg_mu_o_cp == pytest.approx(2.004, rel=1e-2)

    def test_oil_corr_detected(self, minimal_history_df: pd.DataFrame):
        pvt = _build_pvt_summary(minimal_history_df)
        assert pvt is not None
        assert pvt.oil_corr == "standing"

    def test_returns_none_if_no_pvt_cols(self):
        df = pd.DataFrame({"well_id": ["W001"], "date": ["2023-01-01"], "qo_stb_d": [500.0]})
        pvt = _build_pvt_summary(df)
        assert pvt is None

    def test_returns_none_for_empty_df(self):
        pvt = _build_pvt_summary(pd.DataFrame())
        assert pvt is None

    def test_pb_median(self, minimal_history_df: pd.DataFrame):
        pvt = _build_pvt_summary(minimal_history_df)
        assert pvt is not None
        assert pvt.pb_psia == pytest.approx(3200.0)


# ---------------------------------------------------------------------------
# Tests: _build_dca_summary
# ---------------------------------------------------------------------------

class TestBuildDCASummary:
    def test_three_models(self, minimal_dca_df: pd.DataFrame):
        dca = _build_dca_summary(minimal_dca_df)
        assert dca is not None
        assert len(dca.models) == 3

    def test_best_model_highest_r2(self, minimal_dca_df: pd.DataFrame):
        dca = _build_dca_summary(minimal_dca_df)
        assert dca is not None
        assert dca.best_model == "hyperbolic"  # R² 0.99

    def test_eur_by_model(self, minimal_dca_df: pd.DataFrame):
        dca = _build_dca_summary(minimal_dca_df)
        assert dca is not None
        assert dca.eur_exponential_stb == pytest.approx(182500.0)
        assert dca.eur_hyperbolic_stb == pytest.approx(210000.0)
        assert dca.eur_harmonic_stb == pytest.approx(250000.0)

    def test_returns_none_for_empty_df(self):
        dca = _build_dca_summary(pd.DataFrame())
        assert dca is None


# ---------------------------------------------------------------------------
# Tests: _build_rta_summary
# ---------------------------------------------------------------------------

class TestBuildRTASummary:
    def test_extracts_match_params(self, minimal_rta_json: dict):
        rta = _build_rta_summary(minimal_rta_json)
        assert rta is not None
        assert rta.method == "fetkovich"
        assert rta.kh_md_ft == pytest.approx(125.5)
        assert rta.k_md == pytest.approx(2.51)
        assert rta.n_vol_stb == pytest.approx(5_000_000.0)
        assert rta.re_ft == pytest.approx(1320.0)

    def test_multipliers_extracted(self, minimal_rta_json: dict):
        rta = _build_rta_summary(minimal_rta_json)
        assert rta is not None
        assert rta.x_multiplier == pytest.approx(0.85)
        assert rta.y_multiplier == pytest.approx(1.15)

    def test_status_is_demo(self, minimal_rta_json: dict):
        rta = _build_rta_summary(minimal_rta_json)
        assert rta is not None
        assert rta.status == "demo"

    def test_qc_warnings_list(self, minimal_rta_json: dict):
        rta = _build_rta_summary(minimal_rta_json)
        assert rta is not None
        assert len(rta.qc_warnings) == 1
        assert "Drawdown" in rta.qc_warnings[0]

    def test_returns_none_for_empty_dict(self):
        rta = _build_rta_summary({})
        assert rta is None


# ---------------------------------------------------------------------------
# Tests: build_well_results (integración)
# ---------------------------------------------------------------------------

class TestBuildWellResultsIntegration:
    def test_full_integration(self, output_dir_with_files: Path):
        summary = build_well_results(well_id="W001", output_dir=output_dir_with_files)
        assert summary.well_id == "W001"
        assert summary.well_info.history_points == 5
        assert summary.pvt is not None
        assert summary.dca is not None
        assert len(summary.dca.models) == 3
        assert summary.rta is not None
        assert summary.rta.method == "fetkovich"

    def test_missing_dca_file_adds_warning(self, tmp_path: Path, minimal_history_df: pd.DataFrame):
        minimal_history_df.to_csv(tmp_path / "W001_history_enriched.csv", index=False)
        summary = build_well_results(well_id="W001", output_dir=tmp_path)
        assert summary.dca is None
        assert any("dca_fit_results" in w for w in summary.consolidated_warnings)

    def test_missing_rta_file_adds_warning(self, tmp_path: Path, minimal_history_df: pd.DataFrame):
        minimal_history_df.to_csv(tmp_path / "W001_history_enriched.csv", index=False)
        summary = build_well_results(well_id="W001", output_dir=tmp_path)
        assert summary.rta is None
        assert any("rta_match_summary" in w for w in summary.consolidated_warnings)

    def test_missing_history_file_adds_warning(self, tmp_path: Path):
        summary = build_well_results(well_id="W001", output_dir=tmp_path)
        assert summary.well_info.history_points == 0
        assert any("history_enriched" in w for w in summary.consolidated_warnings)

    def test_summary_is_serializable(self, output_dir_with_files: Path):
        """WellResultsSummary debe serializar a JSON sin errores."""
        summary = build_well_results(well_id="W001", output_dir=output_dir_with_files)
        data = summary.model_dump(mode="json")
        assert data["well_id"] == "W001"
        assert "dca" in data
        assert "rta" in data
