"""Technical QC checks for M4 RTA overlay data and match quality.

Each check returns a QCResult with severity ok / warning / error and a
short title + detail message.  The caller decides how to display them.

Checks implemented
------------------
POINT_COUNT        — too few valid overlay points for a reliable match
DRAWDOWN_STABILITY — high coefficient of variation in Δp = pi − pwf
DATA_SPAN          — data spans < 1 log cycle in MBT (non-unique x match)
MATCH_NOT_ADJUSTED — effective multipliers still at default (1.0, 1.0)
QDD_RANGE          — all qDd values compressed near 1.0 (non-unique y match)
TRANSIENT_ONLY     — log-log slope of last points ≈ −0.5 (no BDF seen yet)

References
----------
Fetkovich (SPE-4629, 1980) — section on non-uniqueness of visual matching
Palacio-Blasingame (SPE-25909, 1993) — diagnostic criteria for BDF onset
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from src.services.rta_transform_service import RTATransformPoint


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

Severity = Literal["ok", "warning", "error"]


@dataclass(frozen=True)
class QCResult:
    """Result of a single technical QC check."""

    code: str
    severity: Severity
    title: str
    detail: str


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_MIN_POINTS_ERROR = 5       # < 5 points: error
_MIN_POINTS_WARNING = 15    # < 15 points: warning
_DRAWDOWN_CV_ERROR = 0.30   # CV(Δp) > 30 %: error
_DRAWDOWN_CV_WARNING = 0.15 # CV(Δp) > 15 %: warning
_SPAN_ERROR = 0.5           # < 0.5 log cycles: error
_SPAN_WARNING = 1.0         # < 1.0 log cycle: warning
_MULT_DEFAULT_TOL = 1e-3    # multiplier within this of 1.0 → "not adjusted"
_QDD_HIGH_THRESHOLD = 0.70  # if all qDd > this → non-unique y
_TRANSIENT_SLOPE_THRESHOLD = -0.35  # log-log slope > this → purely transient


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_point_count(points: list[RTATransformPoint]) -> list[QCResult]:
    n = len(points)
    if n < _MIN_POINTS_ERROR:
        return [QCResult(
            code="POINT_COUNT",
            severity="error",
            title="Datos insuficientes",
            detail=(
                f"{n} punto(s) válido(s) — se requieren al menos {_MIN_POINTS_ERROR} "
                "para un overlay significativo. Verifique filtros y columnas."
            ),
        )]
    if n < _MIN_POINTS_WARNING:
        return [QCResult(
            code="POINT_COUNT",
            severity="warning",
            title="Pocos puntos",
            detail=(
                f"{n} puntos válidos. Con menos de {_MIN_POINTS_WARNING} puntos "
                "el match visual tiene alta incertidumbre. Idealmente > 20."
            ),
        )]
    return [QCResult(
        code="POINT_COUNT",
        severity="ok",
        title="Cantidad de puntos",
        detail=f"{n} puntos válidos — suficientes para un overlay confiable.",
    )]


def _check_drawdown_stability(points: list[RTATransformPoint]) -> list[QCResult]:
    if not points:
        return []
    delta_ps = [p.delta_p_psia for p in points]
    mean_dp = sum(delta_ps) / len(delta_ps)
    if mean_dp <= 0:
        return [QCResult(
            code="DRAWDOWN_STABILITY",
            severity="error",
            title="Drawdown negativo o cero",
            detail="Δp promedio ≤ 0. Verifique pi_psia y pwf_used_psia.",
        )]
    variance = sum((dp - mean_dp) ** 2 for dp in delta_ps) / len(delta_ps)
    std_dp = math.sqrt(variance)
    cv = std_dp / mean_dp

    if cv > _DRAWDOWN_CV_ERROR:
        return [QCResult(
            code="DRAWDOWN_STABILITY",
            severity="error",
            title="Drawdown muy inestable",
            detail=(
                f"CV(Δp) = {cv * 100:.1f} % (umbral error: {_DRAWDOWN_CV_ERROR * 100:.0f} %). "
                "La tasa normalizada qo/Δp no es representativa con BHP tan variable. "
                "Considere estabilizar las condiciones de flujo o usar Δp promedio."
            ),
        )]
    if cv > _DRAWDOWN_CV_WARNING:
        return [QCResult(
            code="DRAWDOWN_STABILITY",
            severity="warning",
            title="Drawdown moderadamente variable",
            detail=(
                f"CV(Δp) = {cv * 100:.1f} % (umbral aviso: {_DRAWDOWN_CV_WARNING * 100:.0f} %). "
                "El drawdown varía — la tasa normalizada puede introducir dispersión. "
                "Revise que pwf_used_psia sea el BHP representativo."
            ),
        )]
    return [QCResult(
        code="DRAWDOWN_STABILITY",
        severity="ok",
        title="Drawdown estable",
        detail=f"CV(Δp) = {cv * 100:.1f} % — dentro del rango aceptable.",
    )]


def _check_data_span(points: list[RTATransformPoint]) -> list[QCResult]:
    if len(points) < 2:
        return []
    mbts = [p.material_balance_time for p in points if p.material_balance_time > 0]
    if len(mbts) < 2:
        return []
    log_span = math.log10(max(mbts)) - math.log10(min(mbts))

    if log_span < _SPAN_ERROR:
        return [QCResult(
            code="DATA_SPAN",
            severity="error",
            title="Ventana de datos muy estrecha",
            detail=(
                f"Los datos cubren {log_span:.2f} ciclos log en MBT "
                f"(mín {min(mbts):.1f} — máx {max(mbts):.1f} días). "
                f"Se requiere al menos {_SPAN_ERROR} ciclo para anclar el multiplicador X."
            ),
        )]
    if log_span < _SPAN_WARNING:
        return [QCResult(
            code="DATA_SPAN",
            severity="warning",
            title="Rango de MBT limitado",
            detail=(
                f"Los datos cubren {log_span:.2f} ciclos log en MBT. "
                f"Con menos de {_SPAN_WARNING} ciclo el multiplicador X es poco confiable. "
                "Cuanto mayor el rango, más único es el match."
            ),
        )]
    return [QCResult(
        code="DATA_SPAN",
        severity="ok",
        title="Rango de MBT adecuado",
        detail=f"{log_span:.1f} ciclos log en MBT — buena ventana de datos.",
    )]


def _check_match_adjusted(
    effective_x_multiplier: float,
    effective_y_multiplier: float,
) -> list[QCResult]:
    x_default = abs(effective_x_multiplier - 1.0) < _MULT_DEFAULT_TOL
    y_default = abs(effective_y_multiplier - 1.0) < _MULT_DEFAULT_TOL

    if x_default and y_default:
        return [QCResult(
            code="MATCH_NOT_ADJUSTED",
            severity="warning",
            title="Match no realizado",
            detail=(
                "Ambos multiplicadores están en 1.0 (posición inicial). "
                "Use el joystick para superponer la nube de puntos sobre una curva tipo. "
                "Los parámetros kh, k y N no tienen significado hasta completar el match."
            ),
        )]
    if y_default:
        return [QCResult(
            code="MATCH_NOT_ADJUSTED",
            severity="warning",
            title="Escala Y sin ajustar",
            detail=(
                "El multiplicador Y está en 1.0 — kh no puede estimarse. "
                "Ajuste la escala Y para que la nube coincida verticalmente con la curva tipo."
            ),
        )]
    return [QCResult(
        code="MATCH_NOT_ADJUSTED",
        severity="ok",
        title="Match ajustado",
        detail=(
            f"Escala X = {effective_x_multiplier:.4g}, "
            f"Escala Y = {effective_y_multiplier:.4g}."
        ),
    )]


def _check_qdd_range(
    points: list[RTATransformPoint],
    effective_y_multiplier: float,
) -> list[QCResult]:
    """Check if all qDd values are compressed near 1.0 (purely early-time data)."""
    if not points or abs(effective_y_multiplier - 1.0) < _MULT_DEFAULT_TOL:
        return []   # can't assess until match is attempted
    qdd_values = [p.normalized_rate * effective_y_multiplier for p in points]
    min_qdd = min(qdd_values)
    if min_qdd > _QDD_HIGH_THRESHOLD:
        return [QCResult(
            code="QDD_RANGE",
            severity="warning",
            title="Datos en zona de alta tasa (no unicidad Y)",
            detail=(
                f"El qDd mínimo es {min_qdd:.2f} — todos los puntos están en la parte alta "
                "de la curva tipo. El multiplicador Y tiene alta incertidumbre porque "
                "no se ven puntos en la región de declinación avanzada (qDd < 0.5). "
                "Obtenga datos de mayor duración para anclar el match vertical."
            ),
        )]
    return [QCResult(
        code="QDD_RANGE",
        severity="ok",
        title="Rango qDd adecuado",
        detail=f"qDd mínimo = {min_qdd:.2f} — datos con suficiente declinación.",
    )]


def _check_transient_only(points: list[RTATransformPoint]) -> list[QCResult]:
    """Estimate log-log slope of last few points to detect absence of BDF."""
    if len(points) < 4:
        return []

    # Use last quarter of points (or min 3) sorted by MBT
    sorted_pts = sorted(points, key=lambda p: p.material_balance_time)
    n_tail = max(3, len(sorted_pts) // 4)
    tail = sorted_pts[-n_tail:]

    # Compute log-log slope: d(log nr) / d(log MBT) via linear regression
    log_x = [math.log10(p.material_balance_time) for p in tail if p.material_balance_time > 0]
    log_y = [math.log10(p.normalized_rate) for p in tail if p.normalized_rate > 0]

    if len(log_x) < 3:
        return []

    n = len(log_x)
    mean_x = sum(log_x) / n
    mean_y = sum(log_y) / n
    num = sum((log_x[i] - mean_x) * (log_y[i] - mean_y) for i in range(n))
    den = sum((log_x[i] - mean_x) ** 2 for i in range(n))

    if den < 1e-12:
        return []

    slope = num / den

    # Transient IARF slope ≈ -0.5; BDF slope ≈ -1.0 (exponential) to -0.5 (harmonic)
    # If slope > _TRANSIENT_SLOPE_THRESHOLD (closer to 0 than -0.35), warn
    if slope > _TRANSIENT_SLOPE_THRESHOLD:
        return [QCResult(
            code="TRANSIENT_ONLY",
            severity="warning",
            title="Posible flujo transiente dominante",
            detail=(
                f"Pendiente log-log en los últimos puntos: {slope:.2f}. "
                "Una pendiente cercana a 0 o positiva indica que el pozo aún no ha "
                "alcanzado el límite de drene (BDF). El match en zona transiente es "
                "altamente no único: kh y N no pueden determinarse independientemente."
            ),
        )]
    return [QCResult(
        code="TRANSIENT_ONLY",
        severity="ok",
        title="Señal de BDF detectada",
        detail=f"Pendiente log-log = {slope:.2f} — consistente con declinación BDF.",
    )]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_rta_qc(
    *,
    points: list[RTATransformPoint],
    effective_x_multiplier: float,
    effective_y_multiplier: float,
) -> list[QCResult]:
    """Run all technical QC checks and return a list of QCResult.

    Args:
        points:                  Valid RTATransformPoint list for one method.
        effective_x_multiplier:  Current x-axis match multiplier.
        effective_y_multiplier:  Current y-axis match multiplier.

    Returns:
        List of QCResult ordered by check sequence.  Severity is one of
        'ok', 'warning', or 'error'.  The list always contains one result
        per check (never empty for a given check if data allows it).
    """
    results: list[QCResult] = []
    results.extend(_check_point_count(points))
    results.extend(_check_drawdown_stability(points))
    results.extend(_check_data_span(points))
    results.extend(_check_match_adjusted(effective_x_multiplier, effective_y_multiplier))
    results.extend(_check_qdd_range(points, effective_y_multiplier))
    results.extend(_check_transient_only(points))
    return results


def qc_severity_level(results: list[QCResult]) -> Severity:
    """Return the worst severity across all QC results."""
    if any(r.severity == "error" for r in results):
        return "error"
    if any(r.severity == "warning" for r in results):
        return "warning"
    return "ok"
