"""Servicio DCA / Arps.

Implementa una primera versión conservadora del Módulo 3:
- ajuste exponencial
- ajuste armónico
- ajuste hiperbólico
- pronóstico por integración numérica diaria

No usa scipy para evitar nuevas dependencias. El ajuste hiperbólico y armónico
se hace por búsqueda en grilla sobre Di y b.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from src.domain.dca_models import DCAFitResult, DCAForecastConfig, DCAOutput, DCAModelName

SUPPORTED_MODELS: tuple[DCAModelName, ...] = (
    "exponential",
    "harmonic",
    "hyperbolic",
)


def run_dca_analysis(
    history_df: pd.DataFrame,
    *,
    well_id: str,
    config: DCAForecastConfig | None = None,
    models: Iterable[DCAModelName] = SUPPORTED_MODELS,
) -> DCAOutput:
    """Ejecuta ajuste DCA sobre una historia enriquecida M1-M2.

    Args:
        history_df: DataFrame con al menos date y rate_column.
        well_id: Identificador canónico del pozo.
        config: Configuración de pronóstico.
        models: Modelos Arps a ajustar.

    Returns:
        DCAOutput con resultados, pronóstico y QC.
    """
    cfg = config or DCAForecastConfig()
    clean = prepare_rate_history(
        history_df,
        rate_column=cfg.rate_column,
    )

    fit_results: list[DCAFitResult] = []
    forecast_rows: list[dict[str, object]] = []

    for model in models:
        if model not in SUPPORTED_MODELS:
            msg = f"Modelo DCA no soportado: {model}"
            raise ValueError(msg)

        fit = fit_arps_model(
            clean,
            well_id=well_id,
            model=model,
            rate_column=cfg.rate_column,
            forecast_days=cfg.forecast_days,
            abandonment_rate_stb_d=cfg.abandonment_rate_stb_d,
        )
        fit_results.append(fit)

        forecast_rows.extend(
            build_forecast_rows(
                fit,
                start_date=pd.Timestamp(clean["date"].min()),
                forecast_days=cfg.forecast_days,
                abandonment_rate_stb_d=cfg.abandonment_rate_stb_d,
            )
        )

    qc_report = build_dca_qc_report(
        clean,
        fit_results=fit_results,
        config=cfg,
        well_id=well_id,
    )

    return DCAOutput(
        fit_results=fit_results,
        forecast_rows=forecast_rows,
        qc_report=qc_report,
    )


def prepare_rate_history(
    history_df: pd.DataFrame,
    *,
    rate_column: str,
) -> pd.DataFrame:
    """Limpia y prepara historia para ajuste DCA."""
    required = {"date", rate_column}
    missing = required.difference(history_df.columns)
    if missing:
        msg = f"Faltan columnas para DCA: {sorted(missing)}"
        raise ValueError(msg)

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[rate_column] = pd.to_numeric(df[rate_column], errors="coerce")

    df = df[df["date"].notna()].copy()
    df = df[df[rate_column].notna()].copy()
    df = df[df[rate_column] > 0].copy()

    if len(df) < 3:
        msg = (
            "DCA requiere al menos 3 puntos con fecha válida y tasa positiva "
            f"en {rate_column}."
        )
        raise ValueError(msg)

    df = df.sort_values("date").reset_index(drop=True)
    start_date = pd.Timestamp(df["date"].min())
    df["days"] = (df["date"] - start_date).dt.total_seconds() / 86400.0

    if float(df["days"].max()) <= 0:
        msg = "DCA requiere al menos dos fechas distintas."
        raise ValueError(msg)

    return df[["date", "days", rate_column]].copy()


def fit_arps_model(
    clean_df: pd.DataFrame,
    *,
    well_id: str,
    model: DCAModelName,
    rate_column: str,
    forecast_days: int,
    abandonment_rate_stb_d: float | None,
) -> DCAFitResult:
    """Ajusta un modelo Arps específico."""
    t = clean_df["days"].to_numpy(dtype=float)
    q = clean_df[rate_column].to_numpy(dtype=float)

    if model == "exponential":
        qi, di, b = _fit_exponential(t, q)
    elif model == "harmonic":
        qi, di, b = _fit_harmonic_grid(t, q)
    elif model == "hyperbolic":
        qi, di, b = _fit_hyperbolic_grid(t, q)
    else:
        msg = f"Modelo DCA no soportado: {model}"
        raise ValueError(msg)

    q_hat = arps_rate(t, qi=qi, di=di, b=b)
    rmse = _rmse(q, q_hat)
    r2 = _r2(q, q_hat)

    forecast_days_array = _forecast_days_array(
        forecast_days=forecast_days,
        qi=qi,
        di=di,
        b=b,
        abandonment_rate_stb_d=abandonment_rate_stb_d,
    )
    q_forecast = arps_rate(forecast_days_array, qi=qi, di=di, b=b)
    eur = _trapezoid_cumulative(forecast_days_array, q_forecast)

    return DCAFitResult(
        well_id=well_id,
        model=model,
        rate_column=rate_column,
        qi_stb_d=float(qi),
        di_nominal_d=float(di),
        b=float(b),
        rmse_stb_d=float(rmse),
        r2=float(r2),
        eur_stb=float(eur),
        forecast_days=int(forecast_days_array[-1]),
        n_points=int(len(clean_df)),
    )


def arps_rate(
    days: np.ndarray,
    *,
    qi: float,
    di: float,
    b: float,
) -> np.ndarray:
    """Calcula tasa Arps para días dados."""
    t = np.asarray(days, dtype=float)

    if qi <= 0:
        msg = "qi debe ser positivo."
        raise ValueError(msg)

    if di < 0:
        msg = "di no puede ser negativo."
        raise ValueError(msg)

    if abs(b) < 1e-12:
        return qi * np.exp(-di * t)

    denominator = np.power(1.0 + b * di * t, 1.0 / b)
    return qi / denominator


def build_forecast_rows(
    fit: DCAFitResult,
    *,
    start_date: pd.Timestamp,
    forecast_days: int,
    abandonment_rate_stb_d: float | None,
) -> list[dict[str, object]]:
    """Construye filas diarias de pronóstico para exportar."""
    days = _forecast_days_array(
        forecast_days=forecast_days,
        qi=fit.qi_stb_d,
        di=fit.di_nominal_d,
        b=fit.b,
        abandonment_rate_stb_d=abandonment_rate_stb_d,
    )
    rates = arps_rate(
        days,
        qi=fit.qi_stb_d,
        di=fit.di_nominal_d,
        b=fit.b,
    )
    cumulative = _cumulative_trapezoid(days, rates)

    rows: list[dict[str, object]] = []
    for day, rate, cum in zip(days, rates, cumulative, strict=True):
        rows.append(
            {
                "well_id": fit.well_id,
                "model": fit.model,
                "date": (start_date + pd.Timedelta(days=float(day))).date().isoformat(),
                "days": float(day),
                "qo_forecast_stb_d": float(rate),
                "cumulative_oil_stb": float(cum),
            }
        )

    return rows


def build_dca_qc_report(
    clean_df: pd.DataFrame,
    *,
    fit_results: list[DCAFitResult],
    config: DCAForecastConfig,
    well_id: str,
) -> dict[str, object]:
    """Genera reporte QC para DCA."""
    best = min(fit_results, key=lambda item: item.rmse_stb_d)

    return {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "module": "M3_DCA",
        "well_id": well_id,
        "rate_column": config.rate_column,
        "input_rows_used": int(len(clean_df)),
        "date_min": str(pd.Timestamp(clean_df["date"].min()).date()),
        "date_max": str(pd.Timestamp(clean_df["date"].max()).date()),
        "forecast_days_requested": int(config.forecast_days),
        "abandonment_rate_stb_d": config.abandonment_rate_stb_d,
        "models": [fit.model for fit in fit_results],
        "best_model_by_rmse": best.model,
        "best_model_rmse_stb_d": float(best.rmse_stb_d),
        "best_model_r2": float(best.r2),
        "warnings": _build_dca_warnings(clean_df, fit_results, config),
    }


def _fit_exponential(t: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    """Ajusta q = qi * exp(-Di*t) por regresión lineal en ln(q)."""
    slope, intercept = np.polyfit(t, np.log(q), deg=1)
    qi = float(np.exp(intercept))
    di = float(max(-slope, 0.0))
    return qi, di, 0.0


def _fit_harmonic_grid(t: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    """Ajusta modelo armónico usando grilla sobre Di."""
    di_grid = _build_di_grid(t)
    return _fit_shape_grid(t, q, b_values=[1.0], di_values=di_grid)


def _fit_hyperbolic_grid(t: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    """Ajusta modelo hiperbólico usando grilla sobre b y Di."""
    di_grid = _build_di_grid(t)
    b_grid = np.linspace(0.05, 0.95, 37)
    return _fit_shape_grid(t, q, b_values=b_grid, di_values=di_grid)


def _fit_shape_grid(
    t: np.ndarray,
    q: np.ndarray,
    *,
    b_values: Iterable[float],
    di_values: Iterable[float],
) -> tuple[float, float, float]:
    """Busca qi óptimo para cada par b-Di y selecciona menor SSE."""
    best_qi = float(q[0])
    best_di = 0.0
    best_b = 0.0
    best_sse = float("inf")

    for b in b_values:
        for di in di_values:
            shape = arps_rate(t, qi=1.0, di=float(di), b=float(b))
            denominator = float(np.sum(shape * shape))
            if denominator <= 0:
                continue

            qi = float(np.sum(q * shape) / denominator)
            if qi <= 0:
                continue

            pred = qi * shape
            sse = float(np.sum((q - pred) ** 2))

            if sse < best_sse:
                best_sse = sse
                best_qi = qi
                best_di = float(di)
                best_b = float(b)

    return best_qi, best_di, best_b


def _build_di_grid(t: np.ndarray) -> np.ndarray:
    duration = max(float(np.max(t)), 1.0)

    min_di = max(1e-7, 1.0 / (duration * 10000.0))
    max_di = min(1.0, max(0.05, 10.0 / duration))

    return np.logspace(np.log10(min_di), np.log10(max_di), 160)


def _forecast_days_array(
    *,
    forecast_days: int,
    qi: float,
    di: float,
    b: float,
    abandonment_rate_stb_d: float | None,
) -> np.ndarray:
    if forecast_days <= 0:
        msg = "forecast_days debe ser mayor que cero."
        raise ValueError(msg)

    days = np.arange(0, forecast_days + 1, dtype=float)

    if abandonment_rate_stb_d is None:
        return days

    if abandonment_rate_stb_d <= 0:
        msg = "abandonment_rate_stb_d debe ser positivo si se especifica."
        raise ValueError(msg)

    rates = arps_rate(days, qi=qi, di=di, b=b)
    valid = rates >= abandonment_rate_stb_d

    if not bool(valid.any()):
        return np.array([0.0], dtype=float)

    last_idx = int(np.where(valid)[0][-1])
    return days[: last_idx + 1]


def _cumulative_trapezoid(days: np.ndarray, rates: np.ndarray) -> np.ndarray:
    cumulative = np.zeros_like(days, dtype=float)

    if len(days) <= 1:
        return cumulative

    increments = 0.5 * (rates[1:] + rates[:-1]) * np.diff(days)
    cumulative[1:] = np.cumsum(increments)
    return cumulative


def _trapezoid_cumulative(days: np.ndarray, rates: np.ndarray) -> float:
    if len(days) <= 1:
        return 0.0

    return float(np.trapezoid(rates, days))

def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))

    if ss_tot <= 0:
        return 1.0 if ss_res <= 1e-12 else 0.0

    return 1.0 - ss_res / ss_tot


def _build_dca_warnings(
    clean_df: pd.DataFrame,
    fit_results: list[DCAFitResult],
    config: DCAForecastConfig,
) -> list[str]:
    warnings: list[str] = []

    if len(clean_df) < 8:
        warnings.append(
            "Historia corta para DCA; los ajustes pueden ser sensibles al ruido."
        )

    if config.abandonment_rate_stb_d is None:
        warnings.append(
            "EUR calculado por integración hasta forecast_days; no usa tasa límite "
            "de abandono."
        )

    poor_fits = [fit.model for fit in fit_results if fit.r2 < 0.5]
    if poor_fits:
        warnings.append(
            "Uno o más modelos tienen R² bajo y deben revisarse visualmente: "
            + ", ".join(poor_fits)
        )

    return warnings