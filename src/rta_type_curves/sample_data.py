"""Analytically-generated demo type-curve data for M4 integration.

BDF (boundary-dominated flow) stems are computed from the exact Arps
decline equations and are mathematically correct:

    qDd(b=0,  tDd) = exp(-tDd)                  exponential
    qDd(b>0,  tDd) = (1 + b·tDd)^(-1/b)         hyperbolic
    qDd(b=1,  tDd) = 1 / (1 + tDd)              harmonic

These represent the BDF portion of the Fetkovich type-curve family.
Transient stems (dependent on re/rw) require digitization from
SPE-4629 and are not included here.

STATUS: demo — analytically correct Arps BDF curves, but NOT the full
Fetkovich/Blasingame/Agarwal-Gardner families from the published papers.
Replace with paper-digitized tables before using for final interpretations.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_range(start: float, stop: float, n: int) -> list[float]:
    """Return n log-spaced values from start to stop (inclusive)."""
    log_start = math.log10(start)
    log_stop = math.log10(stop)
    return [10 ** (log_start + i * (log_stop - log_start) / (n - 1)) for i in range(n)]


def _arps_qdd(b: float, tdd: float) -> float:
    """Arps dimensionless decline rate for BDF portion of type curve."""
    if b == 0.0:
        return math.exp(-tdd)
    if b == 1.0:
        return 1.0 / (1.0 + tdd)
    return (1.0 + b * tdd) ** (-1.0 / b)


# ---------------------------------------------------------------------------
# Curve generation
# ---------------------------------------------------------------------------

_N_POINTS = 80
_TDD_RANGE = _log_range(0.001, 100.0, _N_POINTS)
_MIN_QDD = 1e-6  # discard points below this — not useful on log-log plots

_B_VALUES: list[tuple[float, str]] = [
    (0.0,  "b=0.0 (exponencial)"),
    (0.3,  "b=0.3"),
    (0.5,  "b=0.5"),
    (0.7,  "b=0.7"),
    (1.0,  "b=1.0 (armónica)"),
]

_NOTE_SUFFIX = (
    "Curva BDF analítica Arps. "
    "Tallos transientes (re/rw) pendientes de digitalización desde SPE-4629."
)


def _build_bdf_rows(
    method: str,
    curve_id_prefix: str,
    curve_family: str,
    x_label: str,
    y_label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for b, b_label in _B_VALUES:
        b_str = f"{b:.1f}".replace(".", "_")
        curve_id = f"{curve_id_prefix}_{b_str}"
        for tdd in _TDD_RANGE:
            qdd = _arps_qdd(b, tdd)
            if qdd < _MIN_QDD:
                continue
            rows.append({
                "method": method,
                "curve_id": curve_id,
                "curve_family": curve_family,
                "x": round(tdd, 10),
                "y": round(qdd, 10),
                "x_label": x_label,
                "y_label": y_label,
                "source": "analytical_arps_bdf",
                "status": "demo",
                "notes": f"{b_label} — {_NOTE_SUFFIX}",
            })
    return rows


# ---------------------------------------------------------------------------
# Public table
# ---------------------------------------------------------------------------

DEMO_TYPE_CURVE_ROWS: list[dict[str, object]] = [
    *_build_bdf_rows(
        method="fetkovich",
        curve_id_prefix="fetkovich_bdf",
        curve_family="arps_bdf",
        x_label="tDd (tiempo adimensional de declinación)",
        y_label="qDd (tasa adimensional de declinación)",
    ),
    *_build_bdf_rows(
        method="palacio_blasingame",
        curve_id_prefix="blasingame_bdf",
        curve_family="arps_bdf_normalized",
        x_label="tDd (tiempo de balance de materia adimensional)",
        y_label="qDd (tasa normalizada adimensional)",
    ),
    *_build_bdf_rows(
        method="agarwal_gardner",
        curve_id_prefix="agarwal_gardner_bdf",
        curve_family="arps_bdf_area",
        x_label="tDA (tiempo adimensional basado en área)",
        y_label="qDd (tasa adimensional)",
    ),
]
