"""PVT service — pressure-dependent property tables for M2.

Orchestrates the pure correlation functions in pvt_correlations.py to build
a complete pressure grid of PVT properties (Rs, Bo, μo, ρo) for a given
fluid system.

Typical usage
-------------
    from src.services.pvt_service import PVTTableInput, compute_pvt_table

    inp = PVTTableInput(
        api=14.5, gamma_g=0.75, t_f=195.0,
        rsb_scf_stb=120.0,          # Rs at Pb (defines Pb via Standing)
        correlation="standing",
    )
    pb, points = compute_pvt_table(inp)
    print(f"Pb = {pb:.0f} psia")
    for pt in points[:3]:
        print(pt)

Design decisions
----------------
* The service always inserts the Pb point regardless of the pressure grid so
  users can read Bo@Pb and μo@Pb directly.
* Above Pb, Rs is fixed at Rsb.  Bo is computed with the exponential
  compressibility model; μo uses Beggs-Robinson undersaturated extension.
* The output list is sorted ascending by pressure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from src.services.pvt_correlations import (
    api_to_sg,
    bo_undersat,
    br_mu_dead,
    br_mu_sat,
    br_mu_undersat,
    oil_density_lb_ft3,
    standing_bo,
    standing_pb,
    standing_rs,
    vb_bo,
    vb_pb,
    vb_rs,
)

__all__ = [
    "PVTTableInput",
    "PVTPressurePoint",
    "compute_pvt_table",
]

Correlation = Literal["standing", "vasquez_beggs"]


# ---------------------------------------------------------------------------
# Input / output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PVTTableInput:
    """Configuration for a PVT pressure-sweep calculation.

    Args:
        api:           Oil gravity [°API].
        gamma_g:       Gas specific gravity (air = 1) [dimensionless].
        t_f:           Reservoir temperature [°F].
        rsb_scf_stb:   Solution GOR at bubble point [scf/STB].  Used to
                       compute Pb internally.
        p_min_psia:    Lower bound of the pressure grid [psia].  Default 14.7.
        p_max_psia:    Upper bound of the pressure grid [psia].  Default 5000.
        n_points:      Number of grid points (Pb is always added). Default 60.
        co_psi:        Oil compressibility above Pb [psi⁻¹].  Default 1.2e-5.
        correlation:   ``"standing"`` or ``"vasquez_beggs"``.  Default ``"standing"``.
    """
    api: float
    gamma_g: float
    t_f: float
    rsb_scf_stb: float
    p_min_psia: float = 14.7
    p_max_psia: float = 5000.0
    n_points: int = 60
    co_psi: float = 1.2e-5
    correlation: Correlation = "standing"


@dataclass
class PVTPressurePoint:
    """PVT properties evaluated at a single pressure.

    Attributes:
        p_psia:         Pressure [psia].
        rs_scf_stb:     Solution GOR [scf/STB].
        bo_rb_stb:      Oil FVF [bbl/STB].
        mu_o_cp:        Oil viscosity [cp].
        rho_o_lb_ft3:   Oil density at reservoir conditions [lb/ft³].
        regime:         ``"saturated"`` (P ≤ Pb) or ``"undersaturated"`` (P > Pb).
        is_pb:          True for the single point inserted exactly at Pb.
    """
    p_psia: float
    rs_scf_stb: float
    bo_rb_stb: float
    mu_o_cp: float
    rho_o_lb_ft3: float
    regime: Literal["saturated", "undersaturated"]
    is_pb: bool = False


# ---------------------------------------------------------------------------
# Main service function
# ---------------------------------------------------------------------------

def compute_pvt_table(inp: PVTTableInput) -> tuple[float, list[PVTPressurePoint]]:
    """Compute a pressure-sweep PVT table.

    Args:
        inp: :class:`PVTTableInput` configuration.

    Returns:
        ``(pb_psia, points)`` — bubble-point pressure and sorted list of
        :class:`PVTPressurePoint` objects.

    Raises:
        ValueError: If Pb falls below ``p_min_psia`` or inputs are not positive.
    """
    # ── Validate ───────────────────────────────────────────────────────────
    if inp.api <= 0:
        raise ValueError(f"api must be positive, got {inp.api}")
    if inp.gamma_g <= 0:
        raise ValueError(f"gamma_g must be positive, got {inp.gamma_g}")
    if inp.t_f <= 0:
        raise ValueError(f"t_f must be positive [°F], got {inp.t_f}")
    if inp.rsb_scf_stb <= 0:
        raise ValueError(f"rsb_scf_stb must be positive, got {inp.rsb_scf_stb}")

    gamma_o = api_to_sg(inp.api)
    corr = inp.correlation

    # ── Compute Pb ─────────────────────────────────────────────────────────
    if corr == "standing":
        pb = standing_pb(inp.rsb_scf_stb, inp.api, inp.gamma_g, inp.t_f)
    else:  # vasquez_beggs
        pb = vb_pb(inp.rsb_scf_stb, inp.api, inp.gamma_g, inp.t_f)

    # Guard: Pb must be positive (can happen with very low Rsb or bad inputs)
    if pb <= 0:
        raise ValueError(
            f"Computed Pb={pb:.1f} psia ≤ 0. "
            f"Check rsb_scf_stb ({inp.rsb_scf_stb}), API ({inp.api}), "
            f"gamma_g ({inp.gamma_g}), T ({inp.t_f} °F)."
        )

    # ── Dead-oil viscosity (constant, independent of P) ────────────────────
    mu_dead = br_mu_dead(inp.api, inp.t_f)

    # Viscosity and Rs at Pb (anchor for undersaturated branch)
    rs_b = inp.rsb_scf_stb
    bo_b = _bo(corr, rs_b, inp.api, inp.gamma_g, gamma_o, inp.t_f)
    mu_b = br_mu_sat(rs_b, mu_dead)

    # ── Build pressure grid ─────────────────────────────────────────────────
    # linspace from p_min to p_max, always include pb
    p_min = max(1.0, inp.p_min_psia)
    p_max = inp.p_max_psia
    n = max(2, inp.n_points)
    step = (p_max - p_min) / (n - 1)
    pressures: list[float] = [p_min + i * step for i in range(n)]

    # Insert Pb exactly if not already present (within 0.1 psi)
    if not any(abs(p - pb) < 0.1 for p in pressures):
        pressures.append(pb)

    pressures.sort()

    # ── Evaluate each pressure point ────────────────────────────────────────
    points: list[PVTPressurePoint] = []
    for p in pressures:
        is_pb_pt = abs(p - pb) < 0.1
        if p <= pb:
            # Saturated
            rs = _rs(corr, p, inp.api, inp.gamma_g, inp.t_f)
            # Ensure Rs does not exceed Rsb (numerical rounding near Pb)
            rs = min(rs, rs_b)
            bo = _bo(corr, rs, inp.api, inp.gamma_g, gamma_o, inp.t_f)
            mu = br_mu_sat(rs, mu_dead)
            regime: Literal["saturated", "undersaturated"] = "saturated"
        else:
            # Undersaturated
            rs = rs_b
            bo = bo_undersat(pb, bo_b, p, inp.co_psi)
            mu = br_mu_undersat(p, pb, mu_b)
            regime = "undersaturated"

        rho = oil_density_lb_ft3(gamma_o, inp.gamma_g, rs, bo)
        points.append(
            PVTPressurePoint(
                p_psia=p,
                rs_scf_stb=rs,
                bo_rb_stb=bo,
                mu_o_cp=mu,
                rho_o_lb_ft3=rho,
                regime=regime,
                is_pb=is_pb_pt,
            )
        )

    return pb, points


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rs(
    corr: Correlation,
    p: float,
    api: float,
    gamma_g: float,
    t_f: float,
) -> float:
    if corr == "standing":
        return standing_rs(p, api, gamma_g, t_f)
    return vb_rs(p, api, gamma_g, t_f)


def _bo(
    corr: Correlation,
    rs: float,
    api: float,
    gamma_g: float,
    gamma_o: float,
    t_f: float,
) -> float:
    if corr == "standing":
        return standing_bo(rs, gamma_g, gamma_o, t_f)
    return vb_bo(rs, api, gamma_g, t_f)
