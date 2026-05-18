from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.adapters.m1_loader_adapter import load_history_csv
from src.adapters.m2_pvt_adapter import load_pvt_config
from src.services.integration_service import integrate_history_with_pvt


def test_measured_pwf_has_priority_over_estimated(tmp_path: Path) -> None:
    history_path = tmp_path / "history.csv"
    pvt_path = tmp_path / "pvt.json"

    pd.DataFrame(
        [
            {
                "well_id": "WELL-001",
                "date": "2024-01-01",
                "qo_stb_d": 100,
                "qg_mscf_d": 50,
                "qw_stb_d": 10,
                "whp_psia": 300,
                "t_wh_f": 120,
                "pwf_measured_psia": 1500,
                "pwf_estimated_psia": 1400,
            }
        ]
    ).to_csv(history_path, index=False)

    _write_pvt_config(pvt_path)

    history = load_history_csv(history_path, well_id="WELL-001")
    cfg = load_pvt_config(pvt_path)

    output = integrate_history_with_pvt(history, cfg)
    row = output.enriched.iloc[0]

    assert row["pwf_used_psia"] == 1500
    assert row["pwf_source"] == "measured"


def test_estimated_pwf_is_used_when_measured_is_missing(tmp_path: Path) -> None:
    history_path = tmp_path / "history.csv"
    pvt_path = tmp_path / "pvt.json"

    pd.DataFrame(
        [
            {
                "well_id": "WELL-001",
                "date": "2024-01-02",
                "qo_stb_d": 90,
                "qg_mscf_d": 45,
                "qw_stb_d": 12,
                "whp_psia": 280,
                "t_wh_f": 118,
                "pwf_measured_psia": None,
                "pwf_estimated_psia": 1350,
            }
        ]
    ).to_csv(history_path, index=False)

    _write_pvt_config(pvt_path)

    history = load_history_csv(history_path, well_id="WELL-001")
    cfg = load_pvt_config(pvt_path)

    output = integrate_history_with_pvt(history, cfg)
    row = output.enriched.iloc[0]

    assert row["pwf_used_psia"] == 1350
    assert row["pwf_source"] == "estimated_from_history"


def test_auto_pwf_estimation_runs_when_measured_and_estimated_are_missing(
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "history.csv"
    pvt_path = tmp_path / "pvt.json"

    pd.DataFrame(
        [
            {
                "well_id": "WELL-001",
                "date": "2024-01-03",
                "qo_stb_d": 100,
                "qg_mscf_d": 50,
                "qw_stb_d": 10,
                "whp_psia": 300,
                "t_wh_f": 120,
                "pwf_measured_psia": None,
                "pwf_estimated_psia": None,
                "api": 30,
                "tvd_perf_ft": 6000,
                "tubing_id_in": 2.375,
                "length_ft": 6000,
            }
        ]
    ).to_csv(history_path, index=False)

    _write_pvt_config(pvt_path)

    history = load_history_csv(history_path, well_id="WELL-001")
    cfg = load_pvt_config(pvt_path)

    output = integrate_history_with_pvt(history, cfg)
    row = output.enriched.iloc[0]

    assert row["pwf_estimated_psia"] > 0
    assert row["pwf_used_psia"] == row["pwf_estimated_psia"]
    assert row["pwf_source"] == "estimated_v1"
    assert row["pwf_estimation_method"] == "estimate_pwf_v1"


def test_auto_pwf_estimation_can_be_disabled(tmp_path: Path) -> None:
    history_path = tmp_path / "history.csv"
    pvt_path = tmp_path / "pvt.json"

    pd.DataFrame(
        [
            {
                "well_id": "WELL-001",
                "date": "2024-01-03",
                "qo_stb_d": 100,
                "qg_mscf_d": 50,
                "qw_stb_d": 10,
                "whp_psia": 300,
                "t_wh_f": 120,
                "pwf_measured_psia": None,
                "pwf_estimated_psia": None,
            }
        ]
    ).to_csv(history_path, index=False)

    _write_pvt_config(pvt_path)

    history = load_history_csv(history_path, well_id="WELL-001")
    cfg = load_pvt_config(pvt_path)

    output = integrate_history_with_pvt(
        history,
        cfg,
        auto_estimate_missing_pwf=False,
    )
    row = output.enriched.iloc[0]

    assert pd.isna(row["pwf_used_psia"])
    assert row["pwf_source"] == "missing"


def test_row_count_is_preserved_and_pvt_columns_exist(tmp_path: Path) -> None:
    history_path = tmp_path / "history.csv"
    pvt_path = tmp_path / "pvt.json"

    pd.DataFrame(
        [
            {
                "well_id": "WELL-001",
                "date": "2024-01-01",
                "qo_stb_d": 100,
                "qg_mscf_d": 50,
                "qw_stb_d": 10,
                "whp_psia": 300,
                "t_wh_f": 120,
                "pwf_measured_psia": 1500,
                "pwf_estimated_psia": 1400,
            },
            {
                "well_id": "WELL-001",
                "date": "2024-01-02",
                "qo_stb_d": 90,
                "qg_mscf_d": 45,
                "qw_stb_d": 12,
                "whp_psia": 280,
                "t_wh_f": 118,
                "pwf_measured_psia": None,
                "pwf_estimated_psia": 1350,
            },
        ]
    ).to_csv(history_path, index=False)

    _write_pvt_config(pvt_path)

    history = load_history_csv(history_path, well_id="WELL-001")
    cfg = load_pvt_config(pvt_path)

    output = integrate_history_with_pvt(history, cfg)

    assert len(output.enriched) == 2
    for column in ["bo", "rs", "mu_o_cp", "rho_o_lbft3", "pb_psia"]:
        assert column in output.enriched.columns
        assert output.enriched[column].notna().all()


def test_lab_calibration_overrides_config_values(tmp_path: Path) -> None:
    history_path = tmp_path / "history.csv"
    pvt_path = tmp_path / "pvt.json"

    pd.DataFrame(
        [
            {
                "well_id": "WELL-001",
                "date": "2024-01-01",
                "qo_stb_d": 100,
                "qg_mscf_d": 50,
                "qw_stb_d": 10,
                "whp_psia": 300,
                "t_wh_f": 120,
                "pwf_measured_psia": 1500,
                "pwf_estimated_psia": 1400,
            }
        ]
    ).to_csv(history_path, index=False)

    pvt_path.write_text(
        json.dumps(
            {
                "well_id": "WELL-001",
                "api": 30,
                "gamma_g": 0.75,
                "temp_f": 180,
                "pb_psia": 1200,
                "bo_rb_stb": 1.20,
                "rs_scf_stb": 250,
                "mu_o_cp": 2.0,
                "rho_o_lbft3": 52.0,
                "calibrate": True,
                "lab_bo_rb_stb": 1.33,
                "lab_rs_scf_stb": 310,
                "lab_mu_o_cp": 1.85,
                "lab_rho_o_lbft3": 50.8,
                "lab_pb_psia": 1450,
            }
        ),
        encoding="utf-8",
    )

    history = load_history_csv(history_path, well_id="WELL-001")
    cfg = load_pvt_config(pvt_path)

    output = integrate_history_with_pvt(history, cfg)
    row = output.enriched.iloc[0]

    assert row["bo"] == 1.33
    assert row["rs"] == 310
    assert row["mu_o_cp"] == 1.85
    assert row["rho_o_lbft3"] == 50.8
    assert row["pb_psia"] == 1450
    assert bool(row["calibrated_flag"]) is True


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