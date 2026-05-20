import json
from pathlib import Path

import pandas as pd

from src.pipeline.run_full_workflow import run_m1_m2_step, run_m3_dca_step


def _write_history(path: Path) -> None:
    """Create a small synthetic production history for workflow tests."""
    dataframe = pd.DataFrame(
        [
            {
                "well_id": "WELL-001",
                "date": "2024-01-01",
                "qo_stb_d": 1000.0,
                "qg_mscf_d": 500.0,
                "qw_stb_d": 50.0,
                "whp_psia": 500.0,
                "t_wh_f": 180.0,
                "pwf_measured_psia": 1800.0,
                "pwf_estimated_psia": None,
            },
            {
                "well_id": "WELL-001",
                "date": "2024-02-01",
                "qo_stb_d": 930.0,
                "qg_mscf_d": 480.0,
                "qw_stb_d": 55.0,
                "whp_psia": 490.0,
                "t_wh_f": 181.0,
                "pwf_measured_psia": None,
                "pwf_estimated_psia": 1700.0,
            },
            {
                "well_id": "WELL-001",
                "date": "2024-03-01",
                "qo_stb_d": 870.0,
                "qg_mscf_d": 460.0,
                "qw_stb_d": 60.0,
                "whp_psia": 480.0,
                "t_wh_f": 182.0,
                "pwf_measured_psia": None,
                "pwf_estimated_psia": None,
            },
            {
                "well_id": "WELL-001",
                "date": "2024-04-01",
                "qo_stb_d": 810.0,
                "qg_mscf_d": 440.0,
                "qw_stb_d": 65.0,
                "whp_psia": 470.0,
                "t_wh_f": 183.0,
                "pwf_measured_psia": None,
                "pwf_estimated_psia": 1550.0,
            },
        ]
    )
    dataframe.to_csv(path, index=False)


def _write_pvt_config(path: Path) -> None:
    """Create a simple PVT configuration for workflow tests."""
    config = {
        "oil_corr": "standing",
        "pvt_model_version": "m2_pvt_test_v1",
        "calibrate": False,
        "bo": 1.22,
        "rs": 480.0,
        "mu_o_cp": 1.35,
        "rho_o_lbft3": 51.5,
        "pb_psia": 1850.0,
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


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

    enriched_df = pd.read_csv(enriched_path)

    expected_columns = {
        "well_id",
        "date",
        "qo_stb_d",
        "qg_mscf_d",
        "qw_stb_d",
        "whp_psia",
        "t_wh_f",
        "pwf_measured_psia",
        "pwf_estimated_psia",
        "pwf_used_psia",
        "pwf_source",
        "bo",
        "rs",
        "mu_o_cp",
        "rho_o_lbft3",
        "pb_psia",
        "pvt_model_version",
        "oil_corr",
        "calibrated_flag",
    }

    assert expected_columns.issubset(set(enriched_df.columns))
    assert len(enriched_df) == 4
    assert enriched_df.loc[0, "pwf_source"] == "measured"
    assert enriched_df.loc[1, "pwf_source"] == "estimated"

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
        forecast_start_rate_mode="fitted",
        forecast_start_rate=None,
        make_plot=False,
    )

    assert fit_path.exists()
    assert forecast_path.exists()
    assert dca_qc_path.exists()
    assert rate_plot_path is None
    assert forecast_plot_path is None

    fit_df = pd.read_csv(fit_path)
    forecast_df = pd.read_csv(forecast_path)

    assert not fit_df.empty
    assert not forecast_df.empty
    assert len(forecast_df) == 365

    assert "forecast_start_rate_mode" in fit_df.columns
    assert set(fit_df["forecast_start_rate_mode"]) == {"fitted"}

    qc = json.loads(dca_qc_path.read_text(encoding="utf-8"))
    assert qc["well_id"] == "WELL-001"
    assert qc["rate_column"] == "qo_stb_d"
    assert qc["forecast_days"] == 365


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
        forecast_start_rate_mode="fitted",
        forecast_start_rate=None,
        make_plot=False,
    )

    assert fit_path.exists()
    assert forecast_path.exists()
    assert dca_qc_path.exists()
    assert rate_plot_path is None
    assert forecast_plot_path is None


def test_full_workflow_dca_step_supports_last_window_rate_mode(
    tmp_path: Path,
) -> None:
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

    fit_path, _, dca_qc_path, _, _ = run_m3_dca_step(
        well_id="WELL-001",
        history_enriched_csv=enriched_path,
        output_dir=output_dir,
        rate_column="qo_stb_d",
        forecast_days=365,
        abandonment_rate=50.0,
        fit_from_date=None,
        fit_to_date=None,
        exclude_first_n=0,
        forecast_start_rate_mode="last-window-rate",
        forecast_start_rate=None,
        make_plot=False,
    )

    fit_df = pd.read_csv(fit_path)

    assert "forecast_start_rate_mode" in fit_df.columns
    assert set(fit_df["forecast_start_rate_mode"]) == {"last-window-rate"}

    qc = json.loads(dca_qc_path.read_text(encoding="utf-8"))

    assert qc["well_id"] == "WELL-001"
    assert qc["forecast_days"] == 365

    forecast_start_rate = qc.get("forecast_start_rate")

    if isinstance(forecast_start_rate, dict):
        assert forecast_start_rate["mode"] == "last-window-rate"
        assert forecast_start_rate["value"] > 0
    else:
        assert float(forecast_start_rate) > 0