from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.domain.dca_models import DCAForecastConfig
from src.pipeline.run_dca import plot_dca_rate_fit
from src.services.dca_service import run_dca_analysis


def test_plot_dca_rate_fit_generates_png(tmp_path: Path) -> None:
    history = _synthetic_decline_history()

    output = run_dca_analysis(
        history,
        well_id="WELL-001",
        config=DCAForecastConfig(
            forecast_days=365,
            rate_column="qo_stb_d",
        ),
    )

    plot_path = tmp_path / "WELL-001_dca_rate_fit.png"

    generated = plot_dca_rate_fit(
        history,
        output,
        plot_path=plot_path,
        rate_column="qo_stb_d",
    )

    assert generated.exists()
    assert generated.suffix.lower() == ".png"
    assert generated.stat().st_size > 0


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