# src/pipeline/run_well.py
"""Pipeline CLI para integrar Módulo 1 y Módulo 2.

Uso completo:
    python -m src.pipeline.run_well \
        --well-id W-001 \
        --history-csv data/history.csv \
        --pvt-config-json data/pvt_config_W-001.json

Uso corto:
    python -m src.pipeline.run_well --well-id W-001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.adapters.m1_loader_adapter import load_history_csv
from src.adapters.m2_pvt_adapter import load_pvt_config
from src.services.integration_service import integrate_history_with_pvt, write_outputs


DEFAULT_HISTORY_CSV = Path("data/history.csv")
DEFAULT_OUTPUT_DIR = Path("output")


def main() -> int:
    """Ejecuta integración M1 + M2 desde línea de comandos."""
    args = _parse_args()

    history_csv = _resolve_history_csv(args.history_csv)
    pvt_config_json = _resolve_pvt_config_json(
        args.pvt_config_json,
        well_id=args.well_id,
    )

    try:
        history = load_history_csv(
            history_csv,
            well_id=args.well_id,
            from_date=args.from_date,
            to_date=args.to_date,
        )

        pvt_cfg = load_pvt_config(pvt_config_json)

        if pvt_cfg.well_id != args.well_id:
            msg = (
                "El well_id del JSON PVT no coincide con --well-id: "
                f"{pvt_cfg.well_id!r} != {args.well_id!r}"
            )
            raise ValueError(msg)

        output = integrate_history_with_pvt(history, pvt_cfg)

        enriched_path, qc_path = write_outputs(
            output,
            well_id=args.well_id,
            output_dir=args.output_dir,
        )

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("[OK] Integración M1 + M2 completada.")
    print(f"[OK] Historia usada: {history_csv}")
    print(f"[OK] Configuración PVT usada: {pvt_config_json}")
    print(f"[OK] Historia enriquecida: {enriched_path}")
    print(f"[OK] Reporte QC: {qc_path}")
    print(f"[OK] Filas integradas: {len(output.enriched)}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integra historia de pozo M1 con propiedades PVT M2.",
    )

    parser.add_argument("--well-id", required=True)
    parser.add_argument(
        "--history-csv",
        default=None,
        type=Path,
        help="CSV de historia. Por defecto: data/history.csv",
    )
    parser.add_argument(
        "--pvt-config-json",
        default=None,
        type=Path,
        help="JSON PVT. Por defecto: data/pvt_config_<well_id>.json",
    )
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)

    return parser.parse_args()


def _resolve_history_csv(path: Path | None) -> Path:
    """Resuelve ruta de historia usando default conservador."""
    return path if path is not None else DEFAULT_HISTORY_CSV


def _resolve_pvt_config_json(path: Path | None, *, well_id: str) -> Path:
    """Resuelve ruta PVT usando convención data/pvt_config_<well_id>.json."""
    if path is not None:
        return path

    return Path("data") / f"pvt_config_{well_id}.json"


if __name__ == "__main__":
    raise SystemExit(main())