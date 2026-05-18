"""Modelos canónicos para integrar información de pozo, historia y PVT.

Este archivo define el contrato mínimo entre Módulo 1, Módulo 2 y los módulos
futuros DCA/RTA. No depende de Streamlit ni de scripts CLI existentes.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class WellStatic(BaseModel):
    """Información estática mínima del pozo."""

    well_id: str = Field(..., min_length=1)
    well_name: str | None = None
    field_name: str | None = None
    surface_x: float | None = None
    surface_y: float | None = None
    bottomhole_x: float | None = None
    bottomhole_y: float | None = None
    datum_depth_ft: float | None = None
    perforation_mid_tvd_ft: float | None = None

    @field_validator("well_id")
    @classmethod
    def clean_well_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            msg = "well_id no puede estar vacío."
            raise ValueError(msg)
        return value


class HistoryPoint(BaseModel):
    """Punto diario o periódico de historia de producción/presión."""

    well_id: str
    date: date

    qo_stb_d: float | None = None
    qg_mscf_d: float | None = None
    qw_stb_d: float | None = None

    whp_psia: float | None = None
    t_wh_f: float | None = None

    pwf_measured_psia: float | None = None
    pwf_estimated_psia: float | None = None

    @field_validator("well_id")
    @classmethod
    def clean_well_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            msg = "well_id no puede estar vacío."
            raise ValueError(msg)
        return value

    @field_validator(
        "qo_stb_d",
        "qg_mscf_d",
        "qw_stb_d",
        "whp_psia",
        "t_wh_f",
        "pwf_measured_psia",
        "pwf_estimated_psia",
    )
    @classmethod
    def numeric_to_float(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return float(value)


class PVTConfig(BaseModel):
    """Configuración mínima para calcular o asignar propiedades PVT."""

    well_id: str
    api: float
    gamma_g: float
    temp_f: float
    rsb_scf_stb: float | None = None

    pb_psia: float | None = None
    bo_rb_stb: float = 1.20
    rs_scf_stb: float = 250.0
    mu_o_cp: float = 2.0
    rho_o_lbft3: float = 52.0

    oil_corr: str = "placeholder_config"
    pvt_model_version: str = "m2-adapter-0.1"
    calibrate: bool = False
    calibrated_flag: bool = False

    lab_bo_rb_stb: float | None = None
    lab_rs_scf_stb: float | None = None
    lab_mu_o_cp: float | None = None
    lab_rho_o_lbft3: float | None = None
    lab_pb_psia: float | None = None

    @field_validator("well_id")
    @classmethod
    def clean_well_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            msg = "well_id no puede estar vacío."
            raise ValueError(msg)
        return value

    @field_validator("api", "gamma_g", "temp_f")
    @classmethod
    def positive_required(cls, value: float) -> float:
        if value <= 0:
            msg = "api, gamma_g y temp_f deben ser positivos."
            raise ValueError(msg)
        return float(value)


class PVTPoint(BaseModel):
    """Propiedades PVT evaluadas para una fecha/presión/temperatura."""

    well_id: str
    date: date

    bo: float
    rs: float
    mu_o_cp: float
    rho_o_lbft3: float
    pb_psia: float | None = None

    pvt_model_version: str
    oil_corr: str
    calibrated_flag: bool = False


class EnrichedHistoryPoint(BaseModel):
    """Fila canónica enriquecida M1 + M2 lista para DCA/RTA."""

    well_id: str
    date: date

    qo_stb_d: float | None = None
    qg_mscf_d: float | None = None
    qw_stb_d: float | None = None

    whp_psia: float | None = None
    t_wh_f: float | None = None

    pwf_measured_psia: float | None = None
    pwf_estimated_psia: float | None = None
    pwf_used_psia: float | None = None
    pwf_source: Literal["measured", "estimated", "missing"]

    bo: float | None = None
    rs: float | None = None
    mu_o_cp: float | None = None
    rho_o_lbft3: float | None = None
    pb_psia: float | None = None

    pvt_model_version: str | None = None
    oil_corr: str | None = None
    calibrated_flag: bool = False

    created_at_utc: datetime = Field(default_factory=datetime.utcnow)