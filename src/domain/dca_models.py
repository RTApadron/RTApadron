"""Modelos de dominio para Módulo 3 - DCA / Arps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

DCAModelName = Literal["exponential", "harmonic", "hyperbolic"]
ForecastStartRateMode = Literal["fitted", "last-window-rate", "manual"]


@dataclass(frozen=True)
class DCAFitResult:
    """Resultado de ajuste DCA para un modelo Arps."""

    well_id: str
    model: DCAModelName
    rate_column: str
    qi_stb_d: float
    forecast_qi_stb_d: float
    forecast_start_rate_mode: ForecastStartRateMode
    di_nominal_d: float
    b: float
    rmse_stb_d: float
    r2: float
    eur_stb: float
    forecast_days: int
    n_points: int

    def to_dict(self) -> dict[str, object]:
        """Convierte el resultado a diccionario serializable."""
        return asdict(self)


@dataclass(frozen=True)
class DCAForecastConfig:
    """Configuración de ajuste y pronóstico DCA."""

    forecast_days: int = 3650
    abandonment_rate_stb_d: float | None = None
    rate_column: str = "qo_stb_d"
    fit_from_date: str | None = None
    fit_to_date: str | None = None
    exclude_first_n: int = 0
    forecast_start_rate_mode: ForecastStartRateMode = "fitted"
    forecast_start_rate_stb_d: float | None = None


@dataclass(frozen=True)
class DCAOutput:
    """Salida completa del servicio DCA."""

    fit_results: list[DCAFitResult]
    forecast_rows: list[dict[str, object]]
    qc_report: dict[str, object]