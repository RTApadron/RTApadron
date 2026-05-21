"""Modelos de dominio para Módulo 5 — Resultados integrados ecoRTA.

WellResultsSummary agrega outputs de M1-M4 en una estructura canónica
serializable (Pydantic v2) que alimenta el dashboard, la exportación y
el reporte de tesis.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

DataStatus = Literal["measured", "estimated", "calculated", "demo", "missing"]


class WellInfoSummary(BaseModel):
    """Resumen estático del pozo (M1)."""

    well_id: str
    well_name: str | None = None
    field_name: str | None = None
    api_gravity: float | None = None
    temp_f: float | None = None
    history_points: int = 0
    date_from: date | None = None
    date_to: date | None = None
    qo_avg_stb_d: float | None = None
    qo_max_stb_d: float | None = None
    pwf_source_counts: dict[str, int] = Field(default_factory=dict)
    qc_warnings: list[str] = Field(default_factory=list)


class PVTSummary(BaseModel):
    """Resumen PVT (M2) — valores representativos a condiciones promedio."""

    oil_corr: str | None = None
    calibrated: bool = False
    avg_bo_rb_stb: float | None = None
    avg_rs_scf_stb: float | None = None
    avg_mu_o_cp: float | None = None
    pb_psia: float | None = None
    status: DataStatus = "estimated"


class DCAModelSummary(BaseModel):
    """Resultado de un modelo Arps individual (M3)."""

    model: str  # exponential | harmonic | hyperbolic
    qi_stb_d: float
    di_nominal_d: float
    b: float
    eur_stb: float
    r2: float
    rmse_stb_d: float
    forecast_days: int
    n_points: int


class DCASummary(BaseModel):
    """Resumen DCA completo (M3) con los tres modelos Arps."""

    best_model: str | None = None  # modelo con mayor R²
    models: list[DCAModelSummary] = Field(default_factory=list)
    eur_exponential_stb: float | None = None
    eur_harmonic_stb: float | None = None
    eur_hyperbolic_stb: float | None = None
    status: DataStatus = "calculated"

    def best_eur_stb(self) -> float | None:
        if not self.models:
            return None
        best = max(self.models, key=lambda m: m.r2)
        return best.eur_stb


class RTASummary(BaseModel):
    """Resumen del match RTA (M4) — parámetros de yacimiento estimados."""

    method: str | None = None  # fetkovich | palacio_blasingame | agarwal_gardner
    kh_md_ft: float | None = None
    k_md: float | None = None
    n_vol_stb: float | None = None       # OOIP volumétrico desde configuración
    re_ft: float | None = None
    area_acres: float | None = None
    x_multiplier: float | None = None
    y_multiplier: float | None = None
    status: DataStatus = "demo"
    qc_warnings: list[str] = Field(default_factory=list)


class WellResultsSummary(BaseModel):
    """Resultado integrado M1-M4 para un pozo.

    Punto de entrada único para el dashboard comparativo (M5)
    y para todos los formatos de exportación.
    """

    well_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    well_info: WellInfoSummary
    pvt: PVTSummary | None = None
    dca: DCASummary | None = None
    rta: RTASummary | None = None

    consolidated_warnings: list[str] = Field(default_factory=list)

    def completeness_flags(self) -> dict[str, bool]:
        return {
            "M1_historia": self.well_info.history_points > 0,
            "M2_PVT": self.pvt is not None,
            "M3_DCA": self.dca is not None and bool(self.dca.models),
            "M4_RTA": self.rta is not None and self.rta.method is not None,
        }
