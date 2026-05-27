"""Modelos de dominio para Módulo 5 — Resultados integrados ecoRTA.

WellResultsSummary agrega outputs de M1-M4 en una estructura canónica
serializable (Pydantic v2) que alimenta el dashboard, la exportación y
el reporte de tesis.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

DataStatus = Literal["measured", "estimated", "calculated", "demo", "preliminary", "missing"]

# Fuente de datos PVT (sub-tipo para claridad en reportes)
PVTSource = Literal["lab", "correlation", "default"]


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
    pvt_source: PVTSource = "correlation"   # "lab" si calibrated=True, "correlation" por defecto
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
    n_vol_stb: float | None = None       # OOIP volumétrico desde configuración (estático)
    re_ft: float | None = None           # radio de drene desde config (estático)
    area_acres: float | None = None      # área de drene desde config (estático)
    # Dynamic match values — populated only when both joystick axes were adjusted
    n_dyn_stb: float | None = None       # OOIP dinámico del match
    re_dyn_ft: float | None = None       # radio de drene dinámico del match
    a_dyn_acres: float | None = None     # área de drene dinámica del match
    x_multiplier: float | None = None
    y_multiplier: float | None = None
    # Trazabilidad por parámetro
    kh_status: DataStatus = "estimated"   # kh/k: siempre estimado del match
    n_vol_status: DataStatus = "estimated"  # OOIP: estimado volumétrico desde config
    status: DataStatus = "demo"
    qc_warnings: list[str] = Field(default_factory=list)


ComparisonStatus = Literal["match", "close", "diverge", "missing"]


class ExternalSoftwareResult(BaseModel):
    """Valores de referencia ingresados manualmente desde software comercial.

    El campo `software_label` permite nombrar la herramienta de referencia
    sin especificarla en el código (default genérico "Software Comercial").
    """

    software_label: str = "Software Comercial"

    # DCA
    eur_stb: float | None = None           # EUR total (STB)
    qi_stb_d: float | None = None          # tasa inicial (STB/d)
    di_nominal_d: float | None = None      # declinación inicial (/d)
    b_factor: float | None = None          # exponente hiperbólico

    # RTA / parámetros de yacimiento
    kh_md_ft: float | None = None          # permeabilidad-espesor (mD·ft)
    k_md: float | None = None              # permeabilidad efectiva (mD)
    n_vol_stb: float | None = None         # OOIP (STB)
    skin: float | None = None              # daño de formación (adimensional)

    # Generales
    notes: str | None = None
    entered_at: datetime = Field(default_factory=datetime.utcnow)


class ComparisonRow(BaseModel):
    """Fila individual de la tabla comparativa ecoRTA vs software comercial."""

    parameter: str          # nombre del parámetro
    units: str              # unidades
    ecorta_value: float | None
    external_value: float | None
    abs_diff: float | None        # |ecoRTA - externo|
    rel_diff_pct: float | None    # (ecoRTA - externo) / externo × 100
    status: ComparisonStatus      # semáforo de concordancia
    note: str = ""                # advertencia o aclaración (ej. "DEMO")


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

    # All RTA method matches saved from M4 (key = method string, e.g. "fetkovich")
    rta_all_methods: dict[str, RTASummary] = Field(default_factory=dict)

    consolidated_warnings: list[str] = Field(default_factory=list)

    def completeness_flags(self) -> dict[str, bool]:
        return {
            "M1_historia": self.well_info.history_points > 0,
            "M2_PVT": self.pvt is not None,
            "M3_DCA": self.dca is not None and bool(self.dca.models),
            "M4_RTA": self.rta is not None and self.rta.method is not None,
        }
