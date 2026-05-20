"""Pwf estimation utilities for M1.

estimate_pwf_v1  — simplified hydrostatic + linear-friction (legacy, kept for
                   backward compatibility with existing tests and CSV pipeline).

estimate_pwf_v2  — improved model using Darcy-Weisbach with Churchill (1977)
                   friction-factor correlation, which covers all flow regimes
                   (laminar Re < 2300, transition, turbulent) without iteration.

Physical basis
--------------
Both functions assume single-phase liquid flow in the tubing.  For multiphase
flow (gas + liquid) a correlation such as Hagedorn-Brown or Gray is required
(planned for a future M2-Lift commit).

Field-unit formulas used in v2
------------------------------
    Re = 1.488 * rho_lb_ft3 * v_ft_s * D_ft / mu_cp          [dimensionless]
    dP_fric [psi] = f_D * (L_ft / D_ft) * rho_lb_ft3 * v_ft_s**2 / (2 * 32.174 * 144)
    dP_hyd  [psi] = 0.433 * SG * TVD_ft
    Pwf     [psia] = WHP + dP_hyd + dP_fric

References
----------
Churchill (1977), "Friction factor equation spans all fluid flow regimes",
Chemical Engineering, 84(24), 91-92.
"""

import math
from dataclasses import dataclass
from typing import Optional
from .models import MechState, Lift

BBL_TO_FT3 = 5.614583333
DAY_TO_S = 86400.0

@dataclass
class PwfInputs:
    qo_stb_d: float
    qw_stb_d: float
    api: Optional[float] = 30.0
    sg_o: Optional[float] = None     # si quieres forzar SG del aceite
    sg_w: float = 1.0
    whp_psia: float = 100.0
    tvd_perf_ft: float = 6000.0
    tubing_id_in: float = 2.375
    length_ft: Optional[float] = None   # si None, usa tvd_perf_ft
    Cf: float = 0.02  # coef. fricción simplificado (ajustable)

def api_to_sg(api: float) -> float:
    return 141.5 / (api + 131.5)

def estimate_pwf_v1(inp: PwfInputs) -> float:
    sg_o = inp.sg_o if inp.sg_o is not None else api_to_sg(inp.api or 30.0)
    ql_stb_d = max(1e-9, (inp.qo_stb_d + inp.qw_stb_d))
    sg_mix = (inp.qo_stb_d * sg_o + inp.qw_stb_d * inp.sg_w) / ql_stb_d
    grad_psi_per_ft = 0.433 * sg_mix

    L = inp.length_ft if inp.length_ft else inp.tvd_perf_ft
    ID_ft = (inp.tubing_id_in / 12.0)
    area_ft2 = math.pi * (ID_ft**2) / 4.0

    # velocidad líquida (ft/s) asumiendo líquido equivalente
    ql_ft3_s = (ql_stb_d * BBL_TO_FT3) / DAY_TO_S
    v = ql_ft3_s / max(1e-9, area_ft2)

    # fricción simplificada (lineal en v): Cf * v * (L/ID)
    dp_fric_psi = inp.Cf * v * (L / max(1e-9, ID_ft))

    pwf = inp.whp_psia + grad_psi_per_ft * inp.tvd_perf_ft + dp_fric_psi
    return pwf


# ---------------------------------------------------------------------------
# v2 — Darcy-Weisbach with Churchill (1977) friction factor
# ---------------------------------------------------------------------------

def _churchill_friction_factor(Re: float, eps_D: float = 0.0) -> float:
    """Darcy-Weisbach friction factor via Churchill (1977) — all flow regimes.

    Args:
        Re:    Reynolds number (dimensionless).
        eps_D: Relative pipe roughness ε/D (dimensionless). Default 0 (smooth).

    Returns:
        Darcy friction factor f_D (dimensionless).
    """
    if Re < 1.0:
        Re = 1.0   # avoid division by zero
    A = (-2.457 * math.log((7.0 / Re) ** 0.9 + 0.27 * eps_D)) ** 16
    B = (37530.0 / Re) ** 16
    f_D = 8.0 * ((8.0 / Re) ** 12 + (A + B) ** (-1.5)) ** (1.0 / 12.0)
    return f_D


def estimate_pwf_v2(inp: PwfInputs, mu_o_cp: float = 5.0, eps_in: float = 0.0006) -> float:
    """Estimate flowing BHP using Darcy-Weisbach friction (single-phase liquid).

    Args:
        inp:       PwfInputs — same dataclass used by v1.
        mu_o_cp:   Oil viscosity [cp].  Typical range 1–100 cp for Llanos.
        eps_in:    Absolute pipe roughness [in].  Default 0.0006" (commercial steel).

    Returns:
        Pwf [psia].
    """
    sg_o = inp.sg_o if inp.sg_o is not None else api_to_sg(inp.api or 30.0)
    ql_stb_d = max(1e-9, inp.qo_stb_d + inp.qw_stb_d)
    sg_mix = (inp.qo_stb_d * sg_o + inp.qw_stb_d * inp.sg_w) / ql_stb_d

    rho_lb_ft3 = 62.4 * sg_mix               # density [lb/ft³]
    D_ft = inp.tubing_id_in / 12.0           # tubing ID [ft]
    area_ft2 = math.pi * D_ft ** 2 / 4.0

    ql_ft3_s = ql_stb_d * BBL_TO_FT3 / DAY_TO_S
    v_ft_s = ql_ft3_s / max(1e-12, area_ft2)  # average velocity [ft/s]

    # Reynolds number (field units: 1 cp = 6.72e-4 lb/(ft·s))
    Re = rho_lb_ft3 * v_ft_s * D_ft / max(1e-12, mu_o_cp * 6.72e-4)

    eps_D = (eps_in / 12.0) / max(1e-12, D_ft)  # relative roughness
    f_D = _churchill_friction_factor(Re, eps_D)

    L_ft = inp.length_ft if inp.length_ft else inp.tvd_perf_ft

    # Darcy-Weisbach in field units [psi]
    # dP = f_D * (L/D) * rho * v² / (2 * g_c * 144)
    # g_c = 32.174 lbm·ft/(lbf·s²); 144 in²/ft²
    dp_fric_psi = f_D * (L_ft / max(1e-12, D_ft)) * rho_lb_ft3 * v_ft_s ** 2 / (2 * 32.174 * 144)

    dp_hyd_psi = 0.433 * sg_mix * inp.tvd_perf_ft

    return inp.whp_psia + dp_hyd_psi + dp_fric_psi
