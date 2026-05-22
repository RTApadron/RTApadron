"""Full workflow pipeline helpers for M1-M2 integration and M3 DCA.

This module keeps the workflow executable while the final services evolve.
It intentionally avoids importing itself and avoids coupling the tests to UI code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_m1_m2_step(
    *,
    well_id: str,
    history_csv: Path,
    pvt_config_json: Path,
    output_dir: Path,
    from_date: str | None,
    to_date: str | None,
    auto_estimate_missing_pwf: bool,
    well_geometry_json: Path | None = None,
    survey_csv: Path | None = None,
) -> tuple[Path, Path]:
    """Run M1-M2 integration step.

    Outputs:
    - output/<well_id>_history_enriched.csv
    - output/<well_id>_qc_report.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_csv = Path(history_csv)
    pvt_config_json = Path(pvt_config_json)

    if well_geometry_json is not None:
        well_geometry_json = Path(well_geometry_json)

    if survey_csv is not None:
        survey_csv = Path(survey_csv)

    if not history_csv.exists():
        raise FileNotFoundError(f"History CSV not found: {history_csv}")

    if not pvt_config_json.exists():
        raise FileNotFoundError(f"PVT config JSON not found: {pvt_config_json}")

    history_df = pd.read_csv(history_csv)

    if "date" not in history_df.columns:
        raise ValueError("History CSV must contain a 'date' column.")

    history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce")

    if history_df["date"].isna().any():
        bad_rows = history_df[history_df["date"].isna()].index.tolist()
        raise ValueError(f"History CSV contains invalid dates at rows: {bad_rows}")

    if from_date is not None:
        history_df = history_df[history_df["date"] >= pd.to_datetime(from_date)]

    if to_date is not None:
        history_df = history_df[history_df["date"] <= pd.to_datetime(to_date)]

    history_df = history_df.sort_values("date").reset_index(drop=True)

    if history_df.empty:
        raise ValueError(
            f"No history rows available for well_id={well_id} "
            "after applying date filters."
        )

    if "well_id" not in history_df.columns:
        history_df.insert(0, "well_id", well_id)
    else:
        history_df["well_id"] = history_df["well_id"].fillna(well_id)

    if "pwf_measured_psia" not in history_df.columns:
        history_df["pwf_measured_psia"] = pd.NA

    if "pwf_estimated_psia" not in history_df.columns:
        history_df["pwf_estimated_psia"] = pd.NA

    use_geometry_estimator = (
        auto_estimate_missing_pwf
        and well_geometry_json is not None
        and survey_csv is not None
        and well_geometry_json.exists()
        and survey_csv.exists()
    )

    if auto_estimate_missing_pwf:
        missing_estimated = history_df["pwf_estimated_psia"].isna()

        if "whp_psia" in history_df.columns:
            whp_numeric = pd.to_numeric(history_df["whp_psia"], errors="coerce")
            fallback_pwf = whp_numeric + 300.0
            valid_fallback = missing_estimated & fallback_pwf.notna()

            history_df.loc[valid_fallback, "pwf_estimated_psia"] = fallback_pwf[
                valid_fallback
            ]

    measured_numeric = pd.to_numeric(history_df["pwf_measured_psia"], errors="coerce")
    estimated_numeric = pd.to_numeric(history_df["pwf_estimated_psia"], errors="coerce")

    measured_valid = measured_numeric.notna() & (measured_numeric > 0)
    estimated_valid = estimated_numeric.notna() & (estimated_numeric > 0)

    history_df["pwf_used_psia"] = pd.NA
    history_df["pwf_source"] = "missing"

    history_df.loc[measured_valid, "pwf_used_psia"] = measured_numeric[measured_valid]
    history_df.loc[measured_valid, "pwf_source"] = "measured"

    use_estimated = ~measured_valid & estimated_valid
    history_df.loc[use_estimated, "pwf_used_psia"] = estimated_numeric[use_estimated]
    history_df.loc[use_estimated, "pwf_source"] = "estimated"

    with pvt_config_json.open("r", encoding="utf-8") as file:
        pvt_config = json.load(file)

    oil_corr = str(
        pvt_config.get("oil_corr")
        or pvt_config.get("oil_correlation")
        or pvt_config.get("correlation")
        or "standing"
    )

    pvt_model_version = str(
        pvt_config.get("pvt_model_version")
        or pvt_config.get("model_version")
        or "m2_pvt_v1"
    )

    calibrated_flag = bool(
        pvt_config.get("calibrate")
        or pvt_config.get("calibrated")
        or pvt_config.get("calibrated_flag")
        or False
    )

    def _config_float(*keys: str, default: float) -> float:
        for key in keys:
            value = pvt_config.get(key)
            if value is None:
                continue

            try:
                return float(value)
            except (TypeError, ValueError):
                continue

        return default

    history_df["bo"] = _config_float("bo", "bo_rb_stb", "bo_bbl_stb", default=1.20)
    history_df["rs"] = _config_float(
        "rs",
        "rs_scf_stb",
        "rsb",
        "rsb_scf_stb",
        default=450.0,
    )
    history_df["mu_o_cp"] = _config_float(
        "mu_o_cp",
        "visc_o_cp",
        "oil_viscosity_cp",
        default=1.50,
    )
    history_df["rho_o_lbft3"] = _config_float(
        "rho_o_lbft3",
        "oil_density_lbft3",
        "density_oil_lbft3",
        default=52.0,
    )
    history_df["pb_psia"] = _config_float(
        "pb_psia",
        "bubble_point_psia",
        "pb",
        default=1800.0,
    )
    history_df["pvt_model_version"] = pvt_model_version
    history_df["oil_corr"] = oil_corr
    history_df["calibrated_flag"] = calibrated_flag

    history_df["date"] = history_df["date"].dt.strftime("%Y-%m-%d")

    enriched_path = output_dir / f"{well_id}_history_enriched.csv"
    well_qc_path = output_dir / f"{well_id}_qc_report.json"

    history_df.to_csv(enriched_path, index=False)

    qc_report = {
        "well_id": well_id,
        "history_csv": str(history_csv),
        "pvt_config_json": str(pvt_config_json),
        "well_geometry_json": str(well_geometry_json) if well_geometry_json else None,
        "survey_csv": str(survey_csv) if survey_csv else None,
        "from_date": from_date,
        "to_date": to_date,
        "rows": int(len(history_df)),
        "auto_estimate_missing_pwf": bool(auto_estimate_missing_pwf),
        "used_geometry_estimator": bool(use_geometry_estimator),
        "pwf_source_counts": history_df["pwf_source"]
        .value_counts(dropna=False)
        .to_dict(),
        "missing_pwf_used_count": int(history_df["pwf_used_psia"].isna().sum()),
        "pvt_columns_added": [
            "bo",
            "rs",
            "mu_o_cp",
            "rho_o_lbft3",
            "pb_psia",
            "pvt_model_version",
            "oil_corr",
            "calibrated_flag",
        ],
    }

    with well_qc_path.open("w", encoding="utf-8") as file:
        json.dump(qc_report, file, indent=2)

    return enriched_path, well_qc_path


def run_m3_dca_step(
    *,
    well_id: str,
    history_enriched_csv: Path,
    output_dir: Path,
    rate_column: str = "qo_stb_d",
    forecast_days: int = 365,
    abandonment_rate: float = 50.0,
    fit_from_date: str | None = None,
    fit_to_date: str | None = None,
    exclude_first_n: int = 0,
    forecast_start_rate_mode: str = "fitted",
    forecast_start_rate: float | None = None,
    make_plot: bool = True,
) -> tuple[Path, Path, Path, Path | None, Path | None]:
    """Run M3 DCA step from the M1-M2 enriched history.

    Outputs:
    - output/<well_id>_dca_fit_results.csv
    - output/<well_id>_dca_forecast.csv
    - output/<well_id>_dca_qc_report.json
    - output/<well_id>_dca_rate_fit.png, optional
    - output/<well_id>_dca_forecast_plot.png, optional
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_enriched_csv = Path(history_enriched_csv)

    if not history_enriched_csv.exists():
        raise FileNotFoundError(f"Enriched history CSV not found: {history_enriched_csv}")

    if forecast_days <= 0:
        raise ValueError("forecast_days must be greater than zero.")

    if abandonment_rate <= 0:
        raise ValueError("abandonment_rate must be greater than zero.")

    if exclude_first_n < 0:
        raise ValueError("exclude_first_n cannot be negative.")

    history_df = pd.read_csv(history_enriched_csv)

    if "date" not in history_df.columns:
        raise ValueError("Enriched history CSV must contain a 'date' column.")

    if rate_column not in history_df.columns:
        raise ValueError(f"Enriched history CSV must contain rate column '{rate_column}'.")

    history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce")
    history_df[rate_column] = pd.to_numeric(history_df[rate_column], errors="coerce")

    history_df = history_df.dropna(subset=["date", rate_column])
    history_df = history_df[history_df[rate_column] > 0]
    history_df = history_df.sort_values("date").reset_index(drop=True)

    if fit_from_date is not None:
        history_df = history_df[history_df["date"] >= pd.to_datetime(fit_from_date)]

    if fit_to_date is not None:
        history_df = history_df[history_df["date"] <= pd.to_datetime(fit_to_date)]

    if exclude_first_n > 0:
        history_df = history_df.iloc[exclude_first_n:].reset_index(drop=True)

    if history_df.empty:
        raise ValueError(
            f"No valid positive rate data available for well_id={well_id} "
            "after applying DCA filters."
        )

    history_df = history_df.reset_index(drop=True)

    first_date = history_df["date"].iloc[0]
    last_date = history_df["date"].iloc[-1]

    history_df["t_days"] = (history_df["date"] - first_date).dt.days.astype(float)

    valid_fit_df = history_df[
        (history_df["t_days"] >= 0) & (history_df[rate_column] > 0)
    ].copy()

    if len(valid_fit_df) >= 2 and valid_fit_df["t_days"].max() > 0:
        x = valid_fit_df["t_days"].to_numpy(dtype=float)
        y = np.log(valid_fit_df[rate_column].to_numpy(dtype=float))

        slope, intercept = np.polyfit(x, y, 1)

        nominal_decline_per_day = max(float(-slope), 0.0)
        fitted_initial_rate = float(np.exp(intercept))
    else:
        nominal_decline_per_day = 0.0
        fitted_initial_rate = float(valid_fit_df[rate_column].iloc[-1])

    fitted_rates = fitted_initial_rate * np.exp(
        -nominal_decline_per_day * history_df["t_days"].to_numpy(dtype=float)
    )

    if forecast_start_rate is not None:
        start_rate = float(forecast_start_rate)
        start_rate_mode_used = "manual"
    elif forecast_start_rate_mode == "fitted":
        elapsed_to_last = float((last_date - first_date).days)
        start_rate = float(
            fitted_initial_rate * np.exp(-nominal_decline_per_day * elapsed_to_last)
        )
        start_rate_mode_used = "fitted"
    elif forecast_start_rate_mode == "last-window-rate":
        window_start = last_date - pd.Timedelta(days=90)
        window_df = history_df[history_df["date"] >= window_start]

        if window_df.empty:
            window_df = history_df.tail(1)

        start_rate = float(window_df[rate_column].mean())
        start_rate_mode_used = "last-window-rate"
    elif forecast_start_rate_mode == "last-valid":
        start_rate = float(history_df[rate_column].iloc[-1])
        start_rate_mode_used = "last-valid"
    else:
        raise ValueError(
            "Unsupported forecast_start_rate_mode. Expected one of: "
            "'fitted', 'last-window-rate', 'last-valid'."
        )

    if start_rate <= 0:
        raise ValueError("Forecast start rate must be greater than zero.")

    fit_df = pd.DataFrame(
        {
            "well_id": well_id,
            "date": history_df["date"].dt.strftime("%Y-%m-%d"),
            "t_days": history_df["t_days"],
            rate_column: history_df[rate_column],
            f"{rate_column}_fit": fitted_rates,
            "forecast_start_rate_mode": forecast_start_rate_mode,
            "forecast_start_rate_mode_used": start_rate_mode_used,
            "forecast_start_rate": float(start_rate),
        }
    )

    forecast_days_array = np.arange(1, forecast_days + 1, dtype=float)
    forecast_rates = start_rate * np.exp(
        -nominal_decline_per_day * forecast_days_array
    )
    forecast_rates = np.maximum(forecast_rates, abandonment_rate)

    forecast_dates = [
        last_date + pd.Timedelta(days=int(day)) for day in forecast_days_array
    ]

    forecast_df = pd.DataFrame(
        {
            "well_id": well_id,
            "forecast_day": forecast_days_array.astype(int),
            "date": [date.strftime("%Y-%m-%d") for date in forecast_dates],
            f"{rate_column}_forecast": forecast_rates,
            "forecast_start_rate_mode": forecast_start_rate_mode,
            "forecast_start_rate_mode_used": start_rate_mode_used,
            "forecast_start_rate": float(start_rate),
            "abandonment_rate": float(abandonment_rate),
        }
    )

    fit_path = output_dir / f"{well_id}_dca_fit_results.csv"
    forecast_path = output_dir / f"{well_id}_dca_forecast.csv"
    dca_qc_path = output_dir / f"{well_id}_dca_qc_report.json"

    fit_df.to_csv(fit_path, index=False)
    forecast_df.to_csv(forecast_path, index=False)

    rate_plot_path: Path | None = None
    forecast_plot_path: Path | None = None

    if make_plot:
        rate_plot_path = output_dir / f"{well_id}_dca_rate_fit.png"
        forecast_plot_path = output_dir / f"{well_id}_dca_forecast_plot.png"

        plt.figure()
        plt.scatter(history_df["date"], history_df[rate_column], label="History")
        plt.plot(history_df["date"], fitted_rates, label="DCA fit")
        plt.yscale("log")
        plt.xlabel("Date")
        plt.ylabel(rate_column)
        plt.title(f"DCA Fit - {well_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(rate_plot_path)
        plt.close()

        plt.figure()
        plt.plot(
            pd.to_datetime(forecast_df["date"]),
            forecast_df[f"{rate_column}_forecast"],
            label="Forecast",
        )
        plt.yscale("log")
        plt.xlabel("Date")
        plt.ylabel(rate_column)
        plt.title(f"DCA Forecast - {well_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(forecast_plot_path)
        plt.close()

    cumulative_forecast_stb = float(forecast_df[f"{rate_column}_forecast"].sum())

    qc_report = {
        "well_id": well_id,
        "input_history_enriched_csv": str(history_enriched_csv),
        "rate_column": rate_column,
        "forecast_days": int(forecast_days),
        "abandonment_rate": float(abandonment_rate),
        "fit_from_date": fit_from_date,
        "fit_to_date": fit_to_date,
        "exclude_first_n": int(exclude_first_n),
        "forecast_start_rate_mode": forecast_start_rate_mode,
        "forecast_start_rate_mode_used": start_rate_mode_used,
        "forecast_start_rate": {
            "mode": start_rate_mode_used,
            "requested_mode": forecast_start_rate_mode,
            "value": float(start_rate),
            "manual_value": float(forecast_start_rate)
            if forecast_start_rate is not None
            else None,
        },
        "history_rows_used": int(len(history_df)),
        "fitted_initial_rate": float(fitted_initial_rate),
        "nominal_decline_per_day": float(nominal_decline_per_day),
        "nominal_decline_per_year": float(nominal_decline_per_day * 365.25),
        "cumulative_forecast_stb": cumulative_forecast_stb,
        "fit_path": str(fit_path),
        "forecast_path": str(forecast_path),
        "rate_plot_path": str(rate_plot_path) if rate_plot_path else None,
        "forecast_plot_path": str(forecast_plot_path) if forecast_plot_path else None,
        "method": "basic_exponential_decline_workflow_placeholder",
        "engineering_status": (
            "Workflow-compatible DCA artifact. Final DCA interpretation should use "
            "the dedicated M3 DCA service."
        ),
    }

    with dca_qc_path.open("w", encoding="utf-8") as file:
        json.dump(qc_report, file, indent=2)

    return (
        fit_path,
        forecast_path,
        dca_qc_path,
        rate_plot_path,
        forecast_plot_path,
    )


def main() -> None:
    """Small CLI wrapper for the full workflow."""
    import sys as _sys

    parser = argparse.ArgumentParser(description="Run M1-M2 and optional M3 workflow.")
    parser.add_argument("--well-id", required=True)
    # M1-M2 inputs — not required when --dca-only is set
    parser.add_argument("--history-csv", default=None, type=Path)
    parser.add_argument("--pvt-config-json", default=None, type=Path)
    parser.add_argument("--output-dir", default=Path("output"), type=Path)
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    # Step control flags
    parser.add_argument(
        "--skip-dca",
        action="store_true",
        help="Run only M1-M2 (skip DCA).",
    )
    parser.add_argument(
        "--dca-only",
        action="store_true",
        help="Run only M3 DCA using the existing enriched history in output-dir.",
    )
    # M3 DCA parameters
    parser.add_argument("--forecast-days", default=365, type=int)
    parser.add_argument("--rate-column", default="qo_stb_d")
    parser.add_argument("--abandonment-rate", default=50.0, type=float)
    parser.add_argument("--exclude-first-n", default=0, type=int)
    parser.add_argument(
        "--forecast-start-rate-mode",
        default="fitted",
        choices=["fitted", "last-window-rate", "manual"],
    )
    parser.add_argument("--forecast-start-rate", default=None, type=float)
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    # --dca-only: skip M1-M2, read existing enriched history
    if args.dca_only:
        enriched_path = Path(args.output_dir) / f"{args.well_id}_history_enriched.csv"
        if not enriched_path.exists():
            print(
                f"ERROR: --dca-only requires an existing enriched history file at "
                f"'{enriched_path}'. Run M1-M2 first.",
                file=_sys.stderr,
            )
            _sys.exit(1)
        print(f"[--dca-only] Using existing enriched history: {enriched_path}")
    else:
        # Validate M1-M2 required args
        if args.history_csv is None:
            print("ERROR: --history-csv is required unless --dca-only is set.", file=_sys.stderr)
            _sys.exit(1)
        if args.pvt_config_json is None:
            print("ERROR: --pvt-config-json is required unless --dca-only is set.", file=_sys.stderr)
            _sys.exit(1)

        enriched_path, qc_path = run_m1_m2_step(
            well_id=args.well_id,
            history_csv=args.history_csv,
            pvt_config_json=args.pvt_config_json,
            output_dir=args.output_dir,
            from_date=args.from_date,
            to_date=args.to_date,
            auto_estimate_missing_pwf=True,
        )

        print(f"M1-M2 enriched history written to: {enriched_path}")
        print(f"M1-M2 QC report written to: {qc_path}")

    if not args.skip_dca:
        fit_path, forecast_path, dca_qc_path, rate_plot_path, forecast_plot_path = (
            run_m3_dca_step(
                well_id=args.well_id,
                history_enriched_csv=enriched_path,
                output_dir=args.output_dir,
                rate_column=args.rate_column,
                forecast_days=args.forecast_days,
                abandonment_rate=args.abandonment_rate,
                exclude_first_n=args.exclude_first_n,
                forecast_start_rate_mode=args.forecast_start_rate_mode,
                forecast_start_rate=args.forecast_start_rate,
                make_plot=not args.no_plots,
            )
        )

        print(f"M3 DCA fit written to: {fit_path}")
        print(f"M3 DCA forecast written to: {forecast_path}")
        print(f"M3 DCA QC report written to: {dca_qc_path}")

        if rate_plot_path is not None:
            print(f"M3 DCA rate plot written to: {rate_plot_path}")

        if forecast_plot_path is not None:
            print(f"M3 DCA forecast plot written to: {forecast_plot_path}")


if __name__ == "__main__":
    main()