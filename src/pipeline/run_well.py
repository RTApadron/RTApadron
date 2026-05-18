"""Pipeline CLI para integrar Módulo 1 y Módulo 2.

Ejemplo:
python -m src.pipeline.run_well \
  --well-id WELL-001 \
  --history-csv data/history.csv \
  --pvt-config-json data/pvt_config_WELL-001.json \
  --from-date 2024-01-01 \
  --to-date 2025-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.adapters.m1_loader_adapter import load_history_csv
from src.adapters.m2_pvt_adapter import load_pvt_config
from src.services.integration_service import integrate_history_with_pvt, write_outputs


def main() -> int:
    args = _parse_args()

    try:
        history = load_history_csv(
            args.history_csv,
            well_id=args.well_id,
            from_date=args.from_date,
            to_date=args.to_date,
        )
        pvt_cfg = load_pvt_config(args.pvt_config_json)

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
    print(f"[OK] Historia enriquecida: {enriched_path}")
    print(f"[OK] Reporte QC: {qc_path}")
    print(f"[OK] Filas integradas: {len(output.enriched)}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integra historia de pozo M1 con propiedades PVT M2.",
    )
    parser.add_argument("--well-id", required=True)
    parser.add_argument("--history-csv", required=True, type=Path)
    parser.add_argument("--pvt-config-json", required=True, type=Path)
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--output-dir", default="output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())