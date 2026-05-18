from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.pipeline.run_full_workflow import run_m1_m2_step, run_m3_dca_step


def test_full_workflow_steps_generate_expected_outputs(tmp_path: Path) -> None:
    history_path = tmp_path / "history_WELL-001.csv"
    pvt_path = tmp_path / "pvt_config_WELL-001.json"
    output_dir = tmp_path / "output"

    _write_history(history_path)
    _write_pvt_config(pvt_path)

    enriched_path, well_qc_path = run_m1_m2_step(
        well_id="WELL-001",
        history_csv=history_path,
        pvt_config_json=pvt_path,
        output_dir=output_dir,
        from_date=None,
        to_date=None,
        auto_estimate_missing_pwf=True,
    )

    assert enriched_path.exists()
    assert well_qc_path.exists()

    (
        fit_path,
        forecast_path,
        dca_qc_path,
        rate_plot_path,
        forecast_plot_path,
    ) = run_m3_dca_step(
        well_id="WELL-001",
        history_enriched_csv=enriched_path,
        output_dir=output_dir,
        rate_column="qo_stb_d",
        forecast_days=365,
        abandonment_rate=50.0,
        fit_from_date=None,
        fit_to_date=None,
        exclude_first_n=0,
        make_plot=True,
    )

    assert fit_path.exists()
    assert forecast_path.exists()
    assert dca_qc_path.exists()

    assert rate_plot_path is not None
    assert rate_plot_path.exists()
    assert rate_plot_path.stat().st_size > 0

    assert forecast_plot_path is not None
    assert forecast_plot_path.exists()
    assert forecast_plot_path.stat().st_size > 0

    fit_df = pd.read_csv(fit_path)
    assert set(fit_df["model"]) == {"exponential", "harmonic", "hyperbolic"}

    qc = json.loads(dca_qc_path.read_text(encoding="utf-8"))
    assert qc["module"] == "M3_DCA"
    assert qc["well_id"] == "WELL-001"
    assert "eur_comparison" in qc
    assert "eur_summary" in qc
    assert "models_with_similar_rmse" in qc


def test_full_workflow_dca_step_can_skip_plots(tmp_path: Path) -> None:
    history_path = tmp_path / "history_WELL-001.csv"
    pvt_path = tmp_path / "pvt_config_WELL-001.json"
    output_dir = tmp_path / "output"

    _write_history(history_path)
    _write_pvt_config(pvt_path)

    enriched_path, _ = run_m1_m2_step(
        well_id="WELL-001",
        history_csv=history_path,
        pvt_config_json=pvt_path,
        output_dir=output_dir,
        from_date=None,
        to_date=None,
        auto_estimate_missing_pwf=True,
    )

    (
        fit_path,
        forecast_path,
        dca_qc_path,
        rate_plot_path,
        forecast_plot_path,
    ) = run_m3_dca_step(
        well_id="WELL-001",
        history_enriched_csv=enriched_path,
        output_dir=output_dir,
        rate_column="qo_stb_d",
        forecast_days=365,
        abandonment_rate=50.0,
        fit_from_date=None,
        fit_to_date=None,
        exclude_first_n=0,
        make_plot=False,
    )

    assert fit_path.exists()
    assert forecast_path.exists()
    assert dca_qc_path.exists()
    assert rate_plot_path is None
    assert forecast_plot_path is None


def _write_history(path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=12, freq="30D")
    rates = [
        1000.0,
        955.0,
        910.0,
        870.0,
        835.0,
        800.0,
        770.0,
        740.0,
        715.0,
        690.0,
        665.0,
        640.0,
    ]

    pd.DataFrame(
        {
            "well_id": ["WELL-001"] * len(dates),
            "date": dates,
            "qo_stb_d": rates,
            "qg_mscf_d": [100.0] * len(dates),
            "qw_stb_d": [50.0] * len(dates),
            "whp_psia": [300.0] * len(dates),
            "t_wh_f": [120.0] * len(dates),
            "pwf_measured_psia": [None] * len(dates),
            "pwf_estimated_psia": [None] * len(dates),
            "api": [30.0] * len(dates),
            "tvd_perf_ft": [6000.0] * len(dates),
            "tubing_id_in": [2.375] * len(dates),
            "length_ft": [6000.0] * len(dates),
        }
    ).to_csv(path, index=False)


def _write_pvt_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "well_id": "WELL-001",
                "api": 30,
                "gamma_g": 0.75,
                "temp_f": 180,
                "pb_psia": 1200,
                "bo_rb_stb": 1.25,
                "rs_scf_stb": 300,
                "mu_o_cp": 2.5,
                "rho_o_lbft3": 51.5,
            }
        ),
        encoding="utf-8",
    )