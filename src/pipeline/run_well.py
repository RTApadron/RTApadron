"""CLI pipeline for integrating M1 well history with M2 PVT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.adapters.m1_geometry_adapter import load_well_geometry_context
from src.adapters.m1_loader_adapter import load_history_csv
from src.adapters.m2_pvt_adapter import load_pvt_config
from src.services.integration_service import integrate_history_with_pvt, write_outputs

DEFAULT_HISTORY_CSV = Path("data/history.csv")
DEFAULT_OUTPUT_DIR = Path("output")


def main() -> int:
    """Run M1 + M2 integration from command line."""
    args = _parse_args()

    history_csv = _resolve_history_csv(args.history_csv)
    pvt_config_json = _resolve_pvt_config_json(
        args.pvt_config_json,
        well_id=args.well_id,
    )
    geometry_json = _resolve_optional_geometry_json(
        args.well_geometry_json,
        well_id=args.well_id,
        output_dir=args.output_dir,
    )
    survey_csv = _resolve_optional_survey_csv(
        args.survey_csv,
        well_id=args.well_id,
        output_dir=args.output_dir,
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

        geometry_context = load_well_geometry_context(
            well_id=args.well_id,
            geometry_json=geometry_json,
            survey_csv=survey_csv,
        )

        output = integrate_history_with_pvt(
            history,
            pvt_cfg,
            auto_estimate_missing_pwf=not args.no_auto_pwf_estimation,
            geometry_context=geometry_context,
        )

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
    print(f"[OK] Geometría usada: {geometry_json}")
    print(f"[OK] Survey usado: {survey_csv}")
    print(f"[OK] Estimación automática Pwf: {not args.no_auto_pwf_estimation}")
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
    parser.add_argument(
        "--well-geometry-json",
        default=None,
        type=Path,
        help=(
            "JSON de geometría/estado mecánico. "
            "Por defecto intenta output/<well_id>_well_geometry.json."
        ),
    )
    parser.add_argument(
        "--survey-csv",
        default=None,
        type=Path,
        help=(
            "CSV de survey. "
            "Por defecto intenta output/<well_id>_survey_input.csv."
        ),
    )
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument(
        "--no-auto-pwf-estimation",
        action="store_true",
        help="No estimar Pwf automáticamente cuando falte medida y estimada.",
    )

    return parser.parse_args()


def _resolve_history_csv(path: Path | None) -> Path:
    return path if path is not None else DEFAULT_HISTORY_CSV


def _resolve_pvt_config_json(path: Path | None, *, well_id: str) -> Path:
    if path is not None:
        return path

    return Path("data") / f"pvt_config_{well_id}.json"


def _resolve_optional_geometry_json(
    path: Path | None,
    *,
    well_id: str,
    output_dir: Path,
) -> Path | None:
    if path is not None:
        return path

    candidate = output_dir / f"{well_id}_well_geometry.json"
    return candidate if candidate.exists() else None


def _resolve_optional_survey_csv(
    path: Path | None,
    *,
    well_id: str,
    output_dir: Path,
) -> Path | None:
    if path is not None:
        return path

    candidate = output_dir / f"{well_id}_survey_input.csv"
    return candidate if candidate.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())