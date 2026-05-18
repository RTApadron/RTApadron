"""Pipeline CLI para Módulo 3 - DCA / Arps.

Uso corto:
    python -m src.pipeline.run_dca --well-id W-001

Uso completo:
    python -m src.pipeline.run_dca \
        --well-id W-001 \
        --input-csv output/W-001_history_enriched.csv \
        --rate-column qo_stb_d \
        --forecast-days 3650
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.domain.dca_models import DCAForecastConfig
from src.services.dca_service import run_dca_analysis

DEFAULT_OUTPUT_DIR = Path("output")


def main() -> int:
    args = _parse_args()

    input_csv = _resolve_input_csv(args.input_csv, well_id=args.well_id)
    output_dir = args.output_dir

    try:
        if not input_csv.exists():
            msg = f"No existe archivo de entrada DCA: {input_csv}"
            raise FileNotFoundError(msg)

        history = pd.read_csv(input_csv)

        config = DCAForecastConfig(
            forecast_days=args.forecast_days,
            abandonment_rate_stb_d=args.abandonment_rate,
            rate_column=args.rate_column,
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

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("[OK] Módulo 3 DCA completado.")
    print(f"[OK] Historia enriquecida usada: {input_csv}")
    print(f"[OK] Resultados de ajuste: {fit_path}")
    print(f"[OK] Pronóstico DCA: {forecast_path}")
    print(f"[OK] Reporte QC DCA: {qc_path}")
    print(f"[OK] Modelos ajustados: {len(output.fit_results)}")
    print(f"[OK] Mejor modelo por RMSE: {output.qc_report['best_model_by_rmse']}")
    return 0


def write_dca_outputs(
    output,
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

    return parser.parse_args()


def _resolve_input_csv(path: Path | None, *, well_id: str) -> Path:
    if path is not None:
        return path

    return DEFAULT_OUTPUT_DIR / f"{well_id}_history_enriched.csv"


if __name__ == "__main__":
    raise SystemExit(main())