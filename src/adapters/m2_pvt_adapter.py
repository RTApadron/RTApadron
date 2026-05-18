"""Adaptador conservador para Módulo 2: propiedades PVT por fecha.

Este starter pack no intenta reemplazar el cálculo PVT existente. Define una
interfaz estable para que luego conectemos src/rta_pvt/make_pvt_cli.py o el
motor PVT definitivo sin tocar servicios ni UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.domain.models import PVTConfig


def load_pvt_config(path: str | Path) -> PVTConfig:
    """Carga configuración PVT desde JSON."""

    config_path = Path(path)
    if not config_path.exists():
        msg = f"No existe el archivo de configuración PVT: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = json.load(file)

    return PVTConfig(**raw)


def build_pvt_table(history_df: pd.DataFrame, cfg: PVTConfig) -> pd.DataFrame:
    """Construye tabla PVT alineada con la historia.

    Por ahora usa valores de configuración y aplica override de laboratorio
    cuando cfg.calibrate=True. La conexión al motor PVT real debe hacerse aquí,
    manteniendo igual la salida.
    """

    required = {"well_id", "date"}
    missing = required.difference(history_df.columns)
    if missing:
        msg = f"Faltan columnas en historia para construir PVT: {sorted(missing)}"
        raise ValueError(msg)

    rows = history_df[["well_id", "date"]].copy()

    rows["bo"] = _pick_value(
        base=cfg.bo_rb_stb,
        lab=cfg.lab_bo_rb_stb,
        calibrate=cfg.calibrate,
    )
    rows["rs"] = _pick_value(
        base=cfg.rs_scf_stb,
        lab=cfg.lab_rs_scf_stb,
        calibrate=cfg.calibrate,
    )
    rows["mu_o_cp"] = _pick_value(
        base=cfg.mu_o_cp,
        lab=cfg.lab_mu_o_cp,
        calibrate=cfg.calibrate,
    )
    rows["rho_o_lbft3"] = _pick_value(
        base=cfg.rho_o_lbft3,
        lab=cfg.lab_rho_o_lbft3,
        calibrate=cfg.calibrate,
    )
    rows["pb_psia"] = _pick_value(
        base=cfg.pb_psia,
        lab=cfg.lab_pb_psia,
        calibrate=cfg.calibrate,
    )

    rows["pvt_model_version"] = cfg.pvt_model_version
    rows["oil_corr"] = cfg.oil_corr
    rows["calibrated_flag"] = bool(cfg.calibrate or cfg.calibrated_flag)

    return rows


def _pick_value(
    *,
    base: float | None,
    lab: float | None,
    calibrate: bool,
) -> float | None:
    if calibrate and lab is not None:
        return lab
    return base