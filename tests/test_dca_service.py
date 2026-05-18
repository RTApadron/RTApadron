from __future__ import annotations

import math

import pandas as pd

from src.domain.dca_models import DCAForecastConfig
from src.services.dca_service import arps_rate, run_dca_analysis


def test_arps_exponential_rate_declines() -> None:
    days = pd.Series([0.0, 10.0, 20.0]).to_numpy()
    rates = arps_rate(days, qi=1000.0, di=0.01, b=0.0)

    assert rates[0] == 1000.0
    assert rates[1] < rates[0]
    assert rates[2] < rates[1]


def test_dca_runs_all_models_and_preserves_model_names() -> None:
    history = _synthetic_decline_history()

    output = run_dca_analysis(
        history,
        well_id="WELL-001",
        config=DCAForecastConfig(forecast_days=365, rate_column="qo_stb_d"),
    )

    models = {fit.model for fit in output.fit_results}

    assert models == {"exponential", "harmonic", "hyperbolic"}
    assert len(output.fit_results) == 3
    assert output.qc_report["module"] == "M3_DCA"
    assert output.qc_report["well_id"] == "WELL-001"


def test_dca_fit_results_are_positive_and_serializable() -> None:
    history = _synthetic_decline_history()

    output = run_dca_analysis(
        history,
        well_id="WELL-001",
        config=DCAForecastConfig(forecast_days=365, rate_column="qo_stb_d"),
    )

    for fit in output.fit_results:
        row = fit.to_dict()

        assert row["qi_stb_d"] > 0
        assert row["di_nominal_d"] >= 0
        assert row["eur_stb"] >= 0
        assert row["n_points"] == len(history)
        assert math.isfinite(row["rmse_stb_d"])
        assert math.isfinite(row["r2"])


def test_dca_forecast_rows_are_generated_for_each_model() -> None:
    history = _synthetic_decline_history()

    output = run_dca_analysis(
        history,
        well_id="WELL-001",
        config=DCAForecastConfig(forecast_days=30, rate_column="qo_stb_d"),
    )

    forecast = pd.DataFrame(output.forecast_rows)

    assert not forecast.empty
    assert set(forecast["model"].unique()) == {"exponential", "harmonic", "hyperbolic"}
    assert forecast["qo_forecast_stb_d"].notna().all()
    assert forecast["cumulative_oil_stb"].notna().all()


def test_dca_rejects_missing_rate_column() -> None:
    history = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "bad_column": [1, 2, 3, 4],
        }
    )

    try:
        run_dca_analysis(
            history,
            well_id="WELL-001",
            config=DCAForecastConfig(rate_column="qo_stb_d"),
        )
    except ValueError as exc:
        assert "Faltan columnas para DCA" in str(exc)
    else:
        raise AssertionError("Se esperaba ValueError por columna faltante.")


def test_dca_supports_fit_window_filters() -> None:
    history = _synthetic_decline_history()

    output = run_dca_analysis(
        history,
        well_id="WELL-001",
        config=DCAForecastConfig(
            forecast_days=365,
            rate_column="qo_stb_d",
            fit_from_date="2024-03-01",
            fit_to_date="2024-10-01",
            exclude_first_n=1,
        ),
    )

    window = output.qc_report["fit_window"]

    assert output.qc_report["input_rows_used"] < len(history)
    assert window["fit_from_date_requested"] == "2024-03-01"
    assert window["fit_to_date_requested"] == "2024-10-01"
    assert window["exclude_first_n"] == 1
    assert window["fit_date_min"] >= "2024-03-01"
    assert window["fit_date_max"] <= "2024-10-01"


def test_dca_rejects_negative_exclude_first_n() -> None:
    history = _synthetic_decline_history()

    try:
        run_dca_analysis(
            history,
            well_id="WELL-001",
            config=DCAForecastConfig(
                forecast_days=365,
                rate_column="qo_stb_d",
                exclude_first_n=-1,
            ),
        )
    except ValueError as exc:
        assert "exclude_first_n no puede ser negativo" in str(exc)
    else:
        raise AssertionError("Se esperaba ValueError por exclude_first_n negativo.")


def test_dca_rejects_window_with_too_few_points() -> None:
    history = _synthetic_decline_history()

    try:
        run_dca_analysis(
            history,
            well_id="WELL-001",
            config=DCAForecastConfig(
                forecast_days=365,
                rate_column="qo_stb_d",
                fit_from_date="2025-01-01",
                fit_to_date="2025-01-15",
            ),
        )
    except ValueError as exc:
        assert "DCA requiere al menos 3 puntos" in str(exc)
    else:
        raise AssertionError("Se esperaba ValueError por ventana con pocos puntos.")


def _synthetic_decline_history() -> pd.DataFrame:
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

    return pd.DataFrame(
        {
            "well_id": ["WELL-001"] * len(dates),
            "date": dates,
            "qo_stb_d": rates,
            "qg_mscf_d": [100.0] * len(dates),
            "qw_stb_d": [50.0] * len(dates),
            "pwf_used_psia": [2500.0] * len(dates),
        }
    )