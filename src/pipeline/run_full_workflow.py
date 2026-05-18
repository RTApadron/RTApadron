"""Pipeline maestro M1 + M2 + M3.

Ejecuta el flujo completo actual:

1. Integración M1 + M2:
   historia de pozo + Pwf usada + propiedades PVT.

2. Módulo 3 DCA:
   ajuste Arps, forecast, QC y gráfica automática.

Uso corto:
    python -m src.pipeline.run_full_workflow --well-id W001

Uso completo:
    python -m src.pipeline.run_full_workflow \
        --well-id W001 \
        --history-csv data/history_W001.csv \
        --pvt-config-json data/pvt_config_W001.json \
        --fit-from-date 2024-07-01 \
        --fit-to-date 2025-11-30 \
        --abandonment-rate 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.adapters.m1_loader_adapter import load_history_csv
from src.adapters.m2_pvt_adapter import load_pvt_config
from src.domain.dca_models import DCAForecastConfig
from src.pipeline.run_dca import plot_dca_rate_fit, write_dca_outputs
from src.services.dca_service import run_dca_analysis
from src.services.integration_service import integrate_history_with_pvt, write_outputs

DEFAULT_OUTPUT_DIR = Path("output")


def main() -> int:
    """Ejecuta el workflow completo desde línea de comandos."""
    args = _parse_args()

    history_csv = _resolve_history_csv(args.history_csv, well_id=args.well_id)
    pvt_config_json = _resolve_pvt_config_json(
        args.pvt_config_json,
        well_id=args.well_id,
    )
    output_dir = args.output_dir

    try:
        history_enriched_path, well_qc_path = run_m1_m2_step(
            well_id=args.well_id,
            history_csv=history_csv,
            pvt_config_json=pvt_config_json,
            output_dir=output_dir,
            from_date=args.from_date,
            to_date=args.to_date,
            auto_estimate_missing_pwf=not args.no_auto_pwf_estimation,
        )

        dca_fit_path, dca_forecast_path, dca_qc_path, dca_plot_path = run_m3_dca_step(
            well_id=args.well_id,
            history_enriched_csv=history_enriched_path,
            output_dir=output_dir,
            rate_column=args.rate_column,
            forecast_days=args.forecast_days,
            abandonment_rate=args.abandonment_rate,
            fit_from_date=args.fit_from_date,
            fit_to_date=args.fit_to_date,
            exclude_first_n=args.exclude_first_n,
            make_plot=not args.no_plot,
        )

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("[OK] Workflow M1 + M2 + M3 completado.")
    print(f"[OK] Historia fuente: {history_csv}")
    print(f"[OK] Configuración PVT: {pvt_config_json}")
    print(f"[OK] Historia enriquecida: {history_enriched_path}")
    print(f"[OK] QC M1+M2: {well_qc_path}")
    print(f"[OK] DCA fit results: {dca_fit_path}")
    print(f"[OK] DCA forecast: {dca_forecast_path}")
    print(f"[OK] QC DCA: {dca_qc_path}")
    if dca_plot_path is not None:
        print(f"[OK] Gráfica DCA: {dca_plot_path}")

    return 0


def run_m1_m2_step(
    *,
    well_id: str,
    history_csv: Path,
    pvt_config_json: Path,
    output_dir: Path,
    from_date: str | None,
    to_date: str | None,
    auto_estimate_missing_pwf: bool,
) -> tuple[Path, Path]:
    """Ejecuta integración M1 + M2 y escribe salidas."""
    if not history_csv.exists():
        msg = f"No existe archivo de historia: {history_csv}"
        raise FileNotFoundError(msg)

    if not pvt_config_json.exists():
        msg = f"No existe archivo de configuración PVT: {pvt_config_json}"
        raise FileNotFoundError(msg)

    history = load_history_csv(
        history_csv,
        well_id=well_id,
        from_date=from_date,
        to_date=to_date,
    )
    pvt_cfg = load_pvt_config(pvt_config_json)

    if pvt_cfg.well_id != well_id:
        msg = (
            "El well_id del JSON PVT no coincide con --well-id: "
            f"{pvt_cfg.well_id!r} != {well_id!r}"
        )
        raise ValueError(msg)

    output = integrate_history_with_pvt(
        history,
        pvt_cfg,
        auto_estimate_missing_pwf=auto_estimate_missing_pwf,
    )

    return write_outputs(
        output,
        well_id=well_id,
        output_dir=output_dir,
    )


def run_m3_dca_step(
    *,
    well_id: str,
    history_enriched_csv: Path,
    output_dir: Path,
    rate_column: str,
    forecast_days: int,
    abandonment_rate: float | None,
    fit_from_date: str | None,
    fit_to_date: str | None,
    exclude_first_n: int,
    make_plot: bool,
) -> tuple[Path, Path, Path, Path | None]:
    """Ejecuta Módulo 3 DCA sobre historia enriquecida."""
    if not history_enriched_csv.exists():
        msg = f"No existe historia enriquecida para DCA: {history_enriched_csv}"
        raise FileNotFoundError(msg)

    history_enriched = pd.read_csv(history_enriched_csv)

    config = DCAForecastConfig(
        forecast_days=forecast_days,
        abandonment_rate_stb_d=abandonment_rate,
        rate_column=rate_column,
        fit_from_date=fit_from_date,
        fit_to_date=fit_to_date,
        exclude_first_n=exclude_first_n,
    )

    dca_output = run_dca_analysis(
        history_enriched,
        well_id=well_id,
        config=config,
    )

    fit_path, forecast_path, qc_path = write_dca_outputs(
        dca_output,
        well_id=well_id,
        output_dir=output_dir,
    )

    plot_path: Path | None = None
    if make_plot:
        plot_path = output_dir / f"{well_id}_dca_rate_fit.png"
        plot_dca_rate_fit(
            history_enriched,
            dca_output,
            plot_path=plot_path,
            rate_column=rate_column,
        )

    return fit_path, forecast_path, qc_path, plot_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta workflow completo M1 + M2 + M3.",
    )

    parser.add_argument("--well-id", required=True)

    parser.add_argument(
        "--history-csv",
        default=None,
        type=Path,
        help=(
            "CSV de historia. Por defecto intenta data/history_<well_id>.csv "
            "y luego data/history.csv."
        ),
    )
    parser.add_argument(
        "--pvt-config-json",
        default=None,
        type=Path,
        help="JSON PVT. Por defecto: data/pvt_config_<well_id>.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directorio de salida. Por defecto: output.",
    )

    parser.add_argument(
        "--from-date",
        default=None,
        help="Fecha inicial para integrar historia M1+M2, formato YYYY-MM-DD.",
    )
    parser.add_argument(
        "--to-date",
        default=None,
        help="Fecha final para integrar historia M1+M2, formato YYYY-MM-DD.",
    )
    parser.add_argument(
        "--no-auto-pwf-estimation",
        action="store_true",
        help="No estimar Pwf automáticamente cuando falte medida y estimada.",
    )

    parser.add_argument(
        "--rate-column",
        default="qo_stb_d",
        help="Columna de tasa para DCA. Por defecto: qo_stb_d.",
    )
    parser.add_argument(
        "--forecast-days",
        default=3650,
        type=int,
        help="Días de forecast DCA. Por defecto: 3650.",
    )
    parser.add_argument(
        "--abandonment-rate",
        default=None,
        type=float,
        help="Tasa de abandono STB/d para cortar el forecast.",
    )
    parser.add_argument(
        "--fit-from-date",
        default=None,
        help="Fecha inicial de ventana DCA, formato YYYY-MM-DD.",
    )
    parser.add_argument(
        "--fit-to-date",
        default=None,
        help="Fecha final de ventana DCA, formato YYYY-MM-DD.",
    )
    parser.add_argument(
        "--exclude-first-n",
        default=0,
        type=int,
        help="Excluye los primeros N puntos después del filtro de fecha DCA.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="No generar gráfica DCA.",
    )

    return parser.parse_args()


def _resolve_history_csv(path: Path | None, *, well_id: str) -> Path:
    """Resuelve historia con convención específica por pozo y fallback general."""
    if path is not None:
        return path

    candidate = Path("data") / f"history_{well_id}.csv"
    if candidate.exists():
        return candidate

    return Path("data") / "history.csv"


def _resolve_pvt_config_json(path: Path | None, *, well_id: str) -> Path:
    """Resuelve configuración PVT por convención."""
    if path is not None:
        return path

    return Path("data") / f"pvt_config_{well_id}.json"


if __name__ == "__main__":
    raise SystemExit(main())