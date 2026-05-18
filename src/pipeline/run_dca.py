"""Pipeline CLI para Módulo 3 - DCA / Arps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.domain.dca_models import DCAForecastConfig, DCAOutput
from src.services.dca_service import arps_rate, prepare_rate_history, run_dca_analysis

DEFAULT_OUTPUT_DIR = Path("output")


def main() -> int:
    """Ejecuta DCA desde línea de comandos."""
    args = _parse_args()

    input_csv = _resolve_input_csv(args.input_csv, well_id=args.well_id)
    output_dir = args.output_dir
    rate_plot_path = _resolve_rate_plot_path(
        args.plot_file,
        well_id=args.well_id,
        output_dir=output_dir,
    )
    forecast_plot_path = _resolve_forecast_plot_path(
        args.forecast_plot_file,
        well_id=args.well_id,
        output_dir=output_dir,
    )

    try:
        if not input_csv.exists():
            msg = f"No existe archivo de entrada DCA: {input_csv}"
            raise FileNotFoundError(msg)

        history = pd.read_csv(input_csv)

        config = DCAForecastConfig(
            forecast_days=args.forecast_days,
            abandonment_rate_stb_d=args.abandonment_rate,
            rate_column=args.rate_column,
            fit_from_date=args.fit_from_date,
            fit_to_date=args.fit_to_date,
            exclude_first_n=args.exclude_first_n,
            forecast_start_rate_mode=args.forecast_start_rate_mode,
            forecast_start_rate_stb_d=args.forecast_start_rate,
        )

        output = run_dca_analysis(
            history,
            well_id=args.well_id,
            config=config,
        )

        fit_path, forecast_path, qc_path = write_dca_outputs(
            output,
            well_id=args.well_id,
            output_dir=output_dir,
        )

        generated_rate_plot: Path | None = None
        generated_forecast_plot: Path | None = None

        if not args.no_plot:
            generated_rate_plot = plot_dca_rate_fit(
                history,
                output,
                plot_path=rate_plot_path,
                rate_column=args.rate_column,
            )
            generated_forecast_plot = plot_dca_forecast(
                history,
                output,
                plot_path=forecast_plot_path,
                rate_column=args.rate_column,
            )

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("[OK] Módulo 3 DCA completado.")
    print(f"[OK] Historia enriquecida usada: {input_csv}")
    print(f"[OK] Resultados de ajuste: {fit_path}")
    print(f"[OK] Pronóstico DCA: {forecast_path}")
    print(f"[OK] Reporte QC DCA: {qc_path}")
    if generated_rate_plot is not None:
        print(f"[OK] Curva automática DCA: {generated_rate_plot}")
    if generated_forecast_plot is not None:
        print(f"[OK] Gráfica forecast DCA: {generated_forecast_plot}")
    print(f"[OK] Modelos ajustados: {len(output.fit_results)}")
    print(f"[OK] Mejor modelo por RMSE: {output.qc_report['best_model_by_rmse']}")
    return 0


def write_dca_outputs(
    output: DCAOutput,
    *,
    well_id: str,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Escribe resultados DCA a CSV/JSON."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_path = out_dir / f"{well_id}_dca_fit_results.csv"
    forecast_path = out_dir / f"{well_id}_dca_forecast.csv"
    qc_path = out_dir / f"{well_id}_dca_qc_report.json"

    pd.DataFrame([fit.to_dict() for fit in output.fit_results]).to_csv(
        fit_path,
        index=False,
    )
    pd.DataFrame(output.forecast_rows).to_csv(
        forecast_path,
        index=False,
    )

    with qc_path.open("w", encoding="utf-8") as file:
        json.dump(output.qc_report, file, indent=2, ensure_ascii=False)

    return fit_path, forecast_path, qc_path


def plot_dca_rate_fit(
    history_df: pd.DataFrame,
    output: DCAOutput,
    *,
    plot_path: str | Path,
    rate_column: str,
) -> Path:
    """Genera gráfica automática de historia real vs curvas ajustadas."""
    fit_window = output.qc_report.get("fit_window", {})

    clean = prepare_rate_history(
        history_df,
        rate_column=rate_column,
        fit_from_date=fit_window.get("fit_from_date_requested"),
        fit_to_date=fit_window.get("fit_to_date_requested"),
        exclude_first_n=int(fit_window.get("exclude_first_n", 0)),
    )

    all_history = _prepare_plot_history(history_df, rate_column=rate_column)
    out_path = Path(plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_model = str(output.qc_report.get("best_model_by_rmse", "unknown"))

    fig, ax = plt.subplots(figsize=(10, 6))

    if not all_history.empty:
        full_start = pd.Timestamp(all_history["date"].min())
        all_days = (all_history["date"] - full_start).dt.total_seconds() / 86400.0

        ax.plot(
            all_days,
            all_history[rate_column],
            marker="o",
            linestyle="",
            alpha=0.25,
            label="Historia completa",
        )

        fit_days_for_plot = (
            clean["date"] - full_start
        ).dt.total_seconds() / 86400.0
    else:
        fit_days_for_plot = clean["days"]

    ax.plot(
        fit_days_for_plot,
        clean[rate_column],
        marker="o",
        linestyle="",
        label="Ventana de ajuste",
    )

    days_dense_fit = np.linspace(
        float(clean["days"].min()),
        float(clean["days"].max()),
        300,
    )

    if not all_history.empty:
        days_dense_plot = (
            pd.Timestamp(clean["date"].min()) - full_start
        ).days + days_dense_fit
    else:
        days_dense_plot = days_dense_fit

    for fit in output.fit_results:
        rates_fit = arps_rate(
            days_dense_fit,
            qi=fit.qi_stb_d,
            di=fit.di_nominal_d,
            b=fit.b,
        )
        label = (
            f"{fit.model} | "
            f"qi={fit.qi_stb_d:.2f}, "
            f"Di={fit.di_nominal_d:.6f}, "
            f"b={fit.b:.2f}, "
            f"RMSE={fit.rmse_stb_d:.2f}, "
            f"R²={fit.r2:.3f}"
        )
        ax.plot(days_dense_plot, rates_fit, label=label)

    ax.set_title(
        "DCA / Arps - Ajuste automático de tasa\n"
        f"Mejor modelo por RMSE: {best_model}"
    )
    ax.set_xlabel("Tiempo desde inicio de historia [días]")
    ax.set_ylabel(f"Tasa {rate_column} [STB/d]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_dca_forecast(
    history_df: pd.DataFrame,
    output: DCAOutput,
    *,
    plot_path: str | Path,
    rate_column: str,
) -> Path:
    """Genera gráfica de forecast DCA incluyendo historia previa.

    Incluye:
    - historia de producción previa al inicio del DCA/forecast;
    - línea vertical punteada entre historia y predicción;
    - acumulado y EUR en MMSTB.
    """
    forecast = pd.DataFrame(output.forecast_rows)
    if forecast.empty:
        msg = "No hay filas de forecast para graficar."
        raise ValueError(msg)

    required = {"model", "days", "qo_forecast_stb_d", "cumulative_oil_stb"}
    missing = required.difference(forecast.columns)
    if missing:
        msg = f"Forecast incompleto para graficar: faltan {sorted(missing)}"
        raise ValueError(msg)

    history = _prepare_plot_history(history_df, rate_column=rate_column)
    if history.empty:
        msg = "No hay historia válida para incluir en la gráfica de forecast."
        raise ValueError(msg)

    fit_window = output.qc_report.get("fit_window", {})
    fit_start_date = _resolve_fit_start_date(history, fit_window)
    full_start = pd.Timestamp(history["date"].min())

    history["days_since_start"] = (
        history["date"] - full_start
    ).dt.total_seconds() / 86400.0

    pre_history = history[history["date"] < fit_start_date].copy()
    if pre_history.empty:
        pre_history = history.iloc[[0]].copy()

    pre_days = pre_history["days_since_start"].to_numpy(dtype=float)
    pre_rates = pre_history[rate_column].to_numpy(dtype=float)
    pre_cum_stb = _cumulative_from_days_and_rates(pre_days, pre_rates)

    forecast_start_day = float(
        (fit_start_date - full_start).total_seconds() / 86400.0
    )
    history_cum_at_start_stb = (
        float(pre_cum_stb[-1]) if len(pre_cum_stb) > 0 else 0.0
    )

    out_path = Path(plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_model = str(output.qc_report.get("best_model_by_rmse", "unknown"))
    eur_summary = output.qc_report.get("eur_summary", {})

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(
        pre_days,
        pre_rates,
        marker="o",
        linewidth=1.5,
        label="Historia previa",
    )

    for model, group in forecast.groupby("model"):
        group = group.sort_values("days")
        shifted_days = forecast_start_day + group["days"].to_numpy(dtype=float)
        axes[0].plot(
            shifted_days,
            group["qo_forecast_stb_d"],
            label=str(model),
        )

    axes[1].plot(
        pre_days,
        pre_cum_stb / 1_000_000.0,
        linewidth=1.8,
        label="Historia previa",
    )

    for model, group in forecast.groupby("model"):
        group = group.sort_values("days")
        shifted_days = forecast_start_day + group["days"].to_numpy(dtype=float)
        cum_mmstb = (
            history_cum_at_start_stb + group["cumulative_oil_stb"].to_numpy(dtype=float)
        ) / 1_000_000.0
        axes[1].plot(
            shifted_days,
            cum_mmstb,
            label=str(model),
        )

    for ax in axes:
        ax.axvline(
            x=forecast_start_day,
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
        )

    axes[0].text(
        forecast_start_day,
        axes[0].get_ylim()[1] * 0.98,
        "Inicio DCA / predicción",
        fontsize=8,
        ha="left",
        va="top",
    )

    axes[0].set_title(
        "DCA / Arps - Pronóstico de tasa y EUR\n"
        f"Mejor modelo por RMSE: {best_model}"
    )
    axes[0].set_ylabel("Tasa [STB/d]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Tiempo desde inicio de historia [días]")
    axes[1].set_ylabel("Aceite acumulado [MMSTB]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    text = _format_eur_summary_text_mmstb(eur_summary)
    if text:
        axes[1].text(
            0.01,
            0.02,
            text,
            transform=axes[1].transAxes,
            fontsize=8,
            verticalalignment="bottom",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def _format_eur_summary_text_mmstb(summary: object) -> str:
    if not isinstance(summary, dict):
        return ""

    base_model = summary.get("base_case_model_by_rmse")
    base_eur = summary.get("base_case_eur_stb")
    low_model = summary.get("low_case_model_by_eur")
    low_eur = summary.get("low_case_eur_stb")
    high_model = summary.get("high_case_model_by_eur")
    high_eur = summary.get("high_case_eur_stb")

    if base_model is None or base_eur is None:
        return ""

    return (
        f"Base: {base_model} = {float(base_eur) / 1_000_000.0:.3f} MMSTB\n"
        f"Bajo: {low_model} = {float(low_eur) / 1_000_000.0:.3f} MMSTB\n"
        f"Alto: {high_model} = {float(high_eur) / 1_000_000.0:.3f} MMSTB"
    )


def _prepare_plot_history(history_df: pd.DataFrame, *, rate_column: str) -> pd.DataFrame:
    """Prepara historia completa solo para graficar."""
    if "date" not in history_df.columns or rate_column not in history_df.columns:
        return pd.DataFrame()

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[rate_column] = pd.to_numeric(df[rate_column], errors="coerce")

    df = df[df["date"].notna()].copy()
    df = df[df[rate_column].notna()].copy()
    df = df[df[rate_column] > 0].copy()
    df = df.sort_values("date").reset_index(drop=True)

    return df[["date", rate_column]].copy()


def _resolve_fit_start_date(
    history: pd.DataFrame,
    fit_window: dict[str, object],
) -> pd.Timestamp:
    """Resuelve la fecha de inicio del DCA/forecast."""
    fit_date_min = fit_window.get("fit_date_min")
    if isinstance(fit_date_min, str) and fit_date_min:
        return pd.Timestamp(fit_date_min)

    return pd.Timestamp(history["date"].min())


def _cumulative_from_days_and_rates(
    days: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """Calcula acumulado por trapecios para una historia irregular."""
    days = np.asarray(days, dtype=float)
    rates = np.asarray(rates, dtype=float)

    cumulative = np.zeros_like(days, dtype=float)
    if len(days) <= 1:
        return cumulative

    increments = 0.5 * (rates[1:] + rates[:-1]) * np.diff(days)
    cumulative[1:] = np.cumsum(increments)
    return cumulative


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta análisis DCA / Arps sobre historia enriquecida M1-M2.",
    )

    parser.add_argument("--well-id", required=True)
    parser.add_argument(
        "--input-csv",
        default=None,
        type=Path,
        help="CSV enriquecido. Por defecto: output/<well_id>_history_enriched.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directorio de salida. Por defecto: output",
    )
    parser.add_argument(
        "--rate-column",
        default="qo_stb_d",
        help="Columna de tasa a ajustar. Por defecto: qo_stb_d",
    )
    parser.add_argument(
        "--forecast-days",
        default=3650,
        type=int,
        help="Días de pronóstico. Por defecto: 3650",
    )
    parser.add_argument(
        "--abandonment-rate",
        default=None,
        type=float,
        help="Tasa de abandono STB/d. Si se omite, integra hasta forecast-days.",
    )
    parser.add_argument(
        "--fit-from-date",
        default=None,
        help="Fecha inicial de ajuste DCA, formato YYYY-MM-DD.",
    )
    parser.add_argument(
        "--fit-to-date",
        default=None,
        help="Fecha final de ajuste DCA, formato YYYY-MM-DD.",
    )
    parser.add_argument(
        "--exclude-first-n",
        default=0,
        type=int,
        help="Excluye los primeros N puntos después de filtrar por fecha.",
    )
    parser.add_argument(
        "--forecast-start-rate-mode",
        default="fitted",
        choices=["fitted", "last-window-rate", "manual"],
        help=(
            "Modo de tasa inicial para forecast: fitted, last-window-rate o manual. "
            "Por defecto: fitted."
        ),
    )
    parser.add_argument(
        "--forecast-start-rate",
        default=None,
        type=float,
        help="Tasa inicial manual STB/d si --forecast-start-rate-mode manual.",
    )
    parser.add_argument(
        "--plot-file",
        default=None,
        type=Path,
        help="Ruta opcional del PNG. Por defecto: output/<well_id>_dca_rate_fit.png",
    )
    parser.add_argument(
        "--forecast-plot-file",
        default=None,
        type=Path,
        help=(
            "Ruta opcional del PNG de forecast. "
            "Por defecto: output/<well_id>_dca_forecast_plot.png"
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="No generar gráficas automáticas.",
    )

    return parser.parse_args()


def _resolve_input_csv(path: Path | None, *, well_id: str) -> Path:
    if path is not None:
        return path

    return DEFAULT_OUTPUT_DIR / f"{well_id}_history_enriched.csv"


def _resolve_rate_plot_path(
    path: Path | None,
    *,
    well_id: str,
    output_dir: Path,
) -> Path:
    if path is not None:
        return path

    return output_dir / f"{well_id}_dca_rate_fit.png"


def _resolve_forecast_plot_path(
    path: Path | None,
    *,
    well_id: str,
    output_dir: Path,
) -> Path:
    if path is not None:
        return path

    return output_dir / f"{well_id}_dca_forecast_plot.png"


if __name__ == "__main__":
    raise SystemExit(main())