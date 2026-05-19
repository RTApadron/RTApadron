"""CLI entry point for ecoRTA M4 diagnostic preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.rta.diagnostics import run_rta_diagnostics
from src.rta.models import RTAConfig, default_rta_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        msg = f"El archivo JSON no contiene un objeto raíz válido: {path}"
        raise ValueError(msg)

    return data


def load_rta_config(path: Path | None, well_id: str) -> RTAConfig:
    if path is None:
        return default_rta_config(well_id)

    if not path.exists():
        msg = f"No existe rta_config_json: {path}"
        raise FileNotFoundError(msg)

    payload = read_json(path)
    payload["well_id"] = str(payload.get("well_id") or well_id)
    return RTAConfig(**payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare M4 RTA diagnostic variables from enriched history.",
    )
    parser.add_argument("--well-id", required=True)
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=None,
        help="Path to <well_id>_history_enriched.csv.",
    )
    parser.add_argument(
        "--rta-config-json",
        type=Path,
        default=None,
        help="Optional user-editable RTA config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    well_id = args.well_id.strip()
    if not well_id:
        parser.error("--well-id no puede estar vacío.")

    history_csv = args.history_csv or PROJECT_ROOT / "output" / f"{well_id}_history_enriched.csv"
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    config_path = args.rta_config_json
    if config_path is not None and not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_rta_config(config_path, well_id)
    diagnostics_path, qc_path = run_rta_diagnostics(
        history_csv=history_csv,
        config=config,
        output_dir=output_dir,
    )

    print(f"RTA diagnostics written: {diagnostics_path}")
    print(f"RTA QC report written: {qc_path}")


if __name__ == "__main__":
    main()
