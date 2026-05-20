"""Analytically-generated demo type-curve data for M4 integration.

BDF stems (boundary-dominated flow) — exact Arps equations:

    b=0:   qDd = exp(-tDd)
    0<b<1: qDd = (1 + b·tDd)^(-1/b)
    b=1:   qDd = 1/(1 + tDd)

Transient stems (infinite-acting radial flow) — log approximation:

    qD(tD) = 1 / [0.5·(ln(tD) + 0.80907)]          valid for tD > 5

    Converted to decline variables:
        qDd = qD · [ln(re/rw) - 0.5]
        tDd = tD / { [0.5·(re/rw)² - 0.5] · [ln(re/rw) - 0.5] }

    Valid for tD from ~5 up to ~0.1·(re/rw)² (end of infinite-acting period).

STATUS: demo — analytically correct, but NOT the full digitized families
from the published papers (SPE-4629, SPE-25909, SPE-49222).  Transient
stems here use the log approximation (tD > 5); exact Ei-function solution
and full re/rw grid pending digitization.
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
    """Arps BDF dimensionless decline rate."""
    if b == 0.0:
        return math.exp(-tdd)
    if b == 1.0:
        return 1.0 / (1.0 + tdd)
    return (1.0 + b * tdd) ** (-1.0 / b)


def _transient_qd(tD: float) -> float:
    """Infinite-acting radial flow qD via log approximation (valid tD > 5)."""
    return 1.0 / (0.5 * (math.log(tD) + 0.80907))


# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------

_N_BDF = 80
_N_TRANSIENT = 45
_MIN_QDD = 1e-6

_TDD_BDF = _log_range(0.001, 100.0, _N_BDF)

_B_VALUES: list[tuple[float, str]] = [
    (0.0, "b=0.0 (exponencial)"),
    (0.3, "b=0.3"),
    (0.5, "b=0.5"),
    (0.7, "b=0.7"),
    (1.0, "b=1.0 (armónica)"),
]

# re/rw values for transient stems
_RE_RW_VALUES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

_BDF_NOTE = (
    "Curva BDF analítica Arps (exacta). "
    "Tallos transientes calculados analíticamente incluidos."
)
_TRANSIENT_NOTE = (
    "Tallo transiente — flujo radial infinito (aprox. logarítmica, tD > 5). "
    "Reemplazar con digitización de SPE-4629 para interpretación final."
)


# ---------------------------------------------------------------------------
# BDF stem builder
# ---------------------------------------------------------------------------

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
        for tdd in _TDD_BDF:
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
                "notes": f"{b_label} — {_BDF_NOTE}",
            })
    return rows


# ---------------------------------------------------------------------------
# Transient stem builder
# ---------------------------------------------------------------------------

def _build_transient_rows(
    method: str,
    curve_id_prefix: str,
    x_label: str,
    y_label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for re_rw in _RE_RW_VALUES:
        ln_term = math.log(re_rw) - 0.5
        # Decline-variable normalization denominator
        tD_norm = (0.5 * (re_rw ** 2 - 1.0)) * ln_term

        # tD range: start at 5 (log approx valid), end at ~0.1*(re/rw)^2
        tD_start = 5.0
        tD_end = 0.1 * re_rw ** 2
        if tD_end <= tD_start:
            continue

        tD_values = _log_range(tD_start, tD_end, _N_TRANSIENT)
        curve_id = f"{curve_id_prefix}_re_rw_{re_rw}"

        for tD in tD_values:
            qD = _transient_qd(tD)
            qDd = qD * ln_term
            tDd = tD / tD_norm
            if qDd < _MIN_QDD or tDd <= 0:
                continue
            rows.append({
                "method": method,
                "curve_id": curve_id,
                "curve_family": "transient_stem",
                "x": round(tDd, 12),
                "y": round(qDd, 10),
                "x_label": x_label,
                "y_label": y_label,
                "source": "analytical_transient",
                "status": "demo",
                "notes": f"re/rw={re_rw} — {_TRANSIENT_NOTE}",
            })
    return rows


# ---------------------------------------------------------------------------
# Public table
# ---------------------------------------------------------------------------

DEMO_TYPE_CURVE_ROWS: list[dict[str, object]] = [
    # Fetkovich
    *_build_bdf_rows(
        method="fetkovich",
        curve_id_prefix="fetkovich_bdf",
        curve_family="arps_bdf",
        x_label="tDd",
        y_label="qDd",
    ),
    *_build_transient_rows(
        method="fetkovich",
        curve_id_prefix="fetkovich_transient",
        x_label="tDd",
        y_label="qDd",
    ),
    # Palacio-Blasingame
    *_build_bdf_rows(
        method="palacio_blasingame",
        curve_id_prefix="blasingame_bdf",
        curve_family="arps_bdf_normalized",
        x_label="tDd",
        y_label="qDd",
    ),
    *_build_transient_rows(
        method="palacio_blasingame",
        curve_id_prefix="blasingame_transient",
        x_label="tDd",
        y_label="qDd",
    ),
    # Agarwal-Gardner
    *_build_bdf_rows(
        method="agarwal_gardner",
        curve_id_prefix="agarwal_gardner_bdf",
        curve_family="arps_bdf_area",
        x_label="tDA",
        y_label="qDd",
    ),
    *_build_transient_rows(
        method="agarwal_gardner",
        curve_id_prefix="agarwal_gardner_transient",
        x_label="tDA",
        y_label="qDd",
    ),
]
