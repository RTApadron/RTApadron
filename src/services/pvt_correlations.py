"""PVT correlations for oil reservoir fluids — M2.

Pure functions only (no I/O, no Streamlit, no pandas).  Each function takes
scalar floats and returns a scalar float so the service layer can vectorise
over pressure grids as needed.

Correlations implemented
------------------------
Rs / Pb
    Standing (1947)            — Llanos light/medium oils, 9–45 °API
    Vasquez-Beggs (1980)       — wider API range, good for heavy oils

Bo (saturated)
    Standing (1947)            — consistent with Standing Rs
    Vasquez-Beggs (1980)       — alternative for VB Rs

Bo (undersaturated, > Pb)
    Exponential oil compressibility model (requires co input)

μo (dead oil)
    Beggs-Robinson (1975)      — sole model for now

μo (saturated, ≤ Pb)
    Beggs-Robinson (1975)

μo (undersaturated, > Pb)
    Beggs-Robinson (1975) extended

Oil density (reservoir conditions)
    Mass-balance formula (Craft & Hawkins)

Temperature convention
----------------------
All functions expect temperature in **°F** (not °R).  The Vasquez-Beggs
coefficients internally use T + 460 (°R) wherever required — that conversion
is handled inside the function, not by the caller.

References
----------
Standing, M.B. (1947). "A Pressure-Volume-Temperature Correlation for
    Mixtures of California Oils and Gases." Drill. Prod. Pract., API, 275.
Vasquez, M.E. & Beggs, H.D. (1980). "Correlations for Fluid Physical
    Property Prediction." JPT, June, 968–970.
Beggs, H.D. & Robinson, J.R. (1975). "Estimating the Viscosity of Crude
    Oil Systems." JPT, September, 1140–1141.
Craft, B.C. & Hawkins, M.F. (1959). "Applied Petroleum Reservoir
    Engineering." Prentice-Hall.
"""

from __future__ import annotations

import math

__all__ = [
    "api_to_sg",
    # Standing
    "standing_pb",
    "standing_rs",
    "standing_bo",
    # Vasquez-Beggs
    "vb_pb",
    "vb_rs",
    "vb_bo",
    # Beggs-Robinson
    "br_mu_dead",
    "br_mu_sat",
    "br_mu_undersat",
    # Undersaturated Bo
    "bo_undersat",
    # Density
    "oil_density_lb_ft3",
]

# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def api_to_sg(api: float) -> float:
    """Convert °API to specific gravity (water = 1, 60 °F).

    Args:
        api: Oil gravity [°API].

    Returns:
        Specific gravity γo [dimensionless].
    """
    return 141.5 / (api + 131.5)


# ---------------------------------------------------------------------------
# Standing (1947) — Rs, Pb, Bo
# ---------------------------------------------------------------------------

def _standing_exponent(api: float, t_f: float) -> float:
    """Exponent used in Standing Rs/Pb formulas.

    Correct form (T in °F directly, **not** °R):
        a = 0.0125 * API - 0.00091 * T_F

    Note: the sign convention means that at Rs equation the factor is
    10^a (positive), while in the Pb equation it appears as 10^(-a).
    """
    return 0.0125 * api - 0.00091 * t_f


def standing_pb(rsb_scf_stb: float, api: float, gamma_g: float, t_f: float) -> float:
    """Bubble-point pressure via Standing (1947).

    Args:
        rsb_scf_stb: Solution GOR at bubble point [scf/STB].
        api:         Oil gravity [°API].
        gamma_g:     Gas specific gravity (air = 1) [dimensionless].
        t_f:         Reservoir temperature [°F].

    Returns:
        Bubble-point pressure Pb [psia].
    """
    a = _standing_exponent(api, t_f)
    # Pb = 18.2 * [(Rsb/γg)^0.83 · 10^(-a) - 1.4]
    return 18.2 * ((rsb_scf_stb / gamma_g) ** 0.83 * 10.0 ** (-a) - 1.4)


def standing_rs(p_psia: float, api: float, gamma_g: float, t_f: float) -> float:
    """Solution GOR (saturated) via Standing (1947).

    Valid for P ≤ Pb.  Returns Rs at P = Pb when P = Pb.

    Args:
        p_psia:  Pressure [psia].
        api:     Oil gravity [°API].
        gamma_g: Gas specific gravity (air = 1) [dimensionless].
        t_f:     Reservoir temperature [°F].

    Returns:
        Rs [scf/STB].
    """
    a = _standing_exponent(api, t_f)
    # Rs = γg · [(P/18.2 + 1.4) · 10^a]^(1/0.83)
    return gamma_g * ((p_psia / 18.2 + 1.4) * 10.0 ** a) ** (1.0 / 0.83)


def standing_bo(rs: float, gamma_g: float, gamma_o: float, t_f: float) -> float:
    """Oil formation volume factor (saturated) via Standing (1947).

    Args:
        rs:      Solution GOR [scf/STB].
        gamma_g: Gas specific gravity (air = 1) [dimensionless].
        gamma_o: Oil specific gravity (water = 1) [dimensionless].
        t_f:     Reservoir temperature [°F].

    Returns:
        Bo [bbl/STB].
    """
    F = rs * (gamma_g / gamma_o) ** 0.5 + 1.25 * t_f
    return 0.972 + 0.000147 * F ** 1.175


# ---------------------------------------------------------------------------
# Vasquez-Beggs (1980) — Rs, Pb, Bo
# ---------------------------------------------------------------------------

_VB_COEF: dict[str, tuple[float, float, float]] = {
    # key: (C1, C2, C3) for Rs = C1·γg·P^C2·exp(C3·API/(T+460))
    "heavy": (0.0362, 1.0937, 25.724),   # API ≤ 30
    "light": (0.0178, 1.1870, 23.931),   # API > 30
}

_VB_BO_COEF: dict[str, tuple[float, float, float]] = {
    # key: (C1, C2, C3) for Bo = 1 + C1·Rs + C2·(T-60)·(API/γg) + C3·Rs·(T-60)·(API/γg)
    "heavy": (4.677e-4, 1.751e-5, -1.811e-8),   # API ≤ 30
    "light": (4.670e-4, 1.100e-5,  1.337e-9),   # API > 30
}


def _vb_key(api: float) -> str:
    return "heavy" if api <= 30.0 else "light"


def vb_pb(rsb_scf_stb: float, api: float, gamma_g: float, t_f: float) -> float:
    """Bubble-point pressure via Vasquez-Beggs (1980).

    Derived by inverting vb_rs(Pb, ...) = Rsb:
        Pb = (Rsb / (C1·γg·exp(C3·API/(T+460))))^(1/C2)

    Args:
        rsb_scf_stb: Solution GOR at bubble point [scf/STB].
        api:         Oil gravity [°API].
        gamma_g:     Gas specific gravity (air = 1) [dimensionless].
        t_f:         Reservoir temperature [°F].

    Returns:
        Bubble-point pressure Pb [psia].
    """
    c1, c2, c3 = _VB_COEF[_vb_key(api)]
    denom = c1 * gamma_g * math.exp(c3 * api / (t_f + 460.0))
    return (rsb_scf_stb / denom) ** (1.0 / c2)


def vb_rs(p_psia: float, api: float, gamma_g: float, t_f: float) -> float:
    """Solution GOR (saturated) via Vasquez-Beggs (1980).

    Args:
        p_psia:  Pressure [psia].
        api:     Oil gravity [°API].
        gamma_g: Gas specific gravity (air = 1) [dimensionless].
        t_f:     Reservoir temperature [°F].

    Returns:
        Rs [scf/STB].
    """
    c1, c2, c3 = _VB_COEF[_vb_key(api)]
    return c1 * gamma_g * p_psia ** c2 * math.exp(c3 * api / (t_f + 460.0))


def vb_bo(rs: float, api: float, gamma_g: float, t_f: float) -> float:
    """Oil FVF (saturated) via Vasquez-Beggs (1980).

    Args:
        rs:      Solution GOR [scf/STB].
        api:     Oil gravity [°API].
        gamma_g: Gas specific gravity (air = 1) [dimensionless].
        t_f:     Reservoir temperature [°F].

    Returns:
        Bo [bbl/STB].
    """
    c1, c2, c3 = _VB_BO_COEF[_vb_key(api)]
    return 1.0 + c1 * rs + c2 * (t_f - 60.0) * (api / gamma_g) + c3 * rs * (t_f - 60.0) * (api / gamma_g)


# ---------------------------------------------------------------------------
# Undersaturated Bo (> Pb)
# ---------------------------------------------------------------------------

def bo_undersat(pb_psia: float, bo_b: float, p_psia: float, co_psi: float) -> float:
    """Oil FVF above bubble point (undersaturated oil).

    Uses the exponential compressibility model:
        Bo(P > Pb) = Bo_b · exp(co · (Pb − P))

    Args:
        pb_psia: Bubble-point pressure [psia].
        bo_b:    Oil FVF at Pb [bbl/STB].
        p_psia:  Current pressure (> Pb) [psia].
        co_psi:  Oil compressibility [psi⁻¹].  Typical: 1e-5 to 2e-5 psi⁻¹.

    Returns:
        Bo [bbl/STB].
    """
    return bo_b * math.exp(co_psi * (pb_psia - p_psia))


# ---------------------------------------------------------------------------
# Beggs-Robinson (1975) — μo dead, saturated, undersaturated
# ---------------------------------------------------------------------------

def br_mu_dead(api: float, t_f: float) -> float:
    """Dead-oil viscosity via Beggs-Robinson (1975).

    Args:
        api: Oil gravity [°API].
        t_f: Temperature [°F].

    Returns:
        μ_dead [cp].
    """
    gamma_o = api_to_sg(api)
    x = t_f ** (-1.163) * math.exp(13.108 - 6.591 / gamma_o)
    return 10.0 ** x - 1.0


def br_mu_sat(rs: float, mu_dead: float) -> float:
    """Saturated live-oil viscosity via Beggs-Robinson (1975).

    Args:
        rs:      Solution GOR at current pressure [scf/STB].
        mu_dead: Dead-oil viscosity [cp].

    Returns:
        μo_sat [cp].
    """
    a = 10.715 * (rs + 100.0) ** (-0.515)
    b = 5.44 * (rs + 150.0) ** (-0.338)
    return a * mu_dead ** b


def br_mu_undersat(p_psia: float, pb_psia: float, mu_b: float) -> float:
    """Undersaturated oil viscosity via Beggs-Robinson (1975).

    Args:
        p_psia:  Current pressure (> Pb) [psia].
        pb_psia: Bubble-point pressure [psia].
        mu_b:    Oil viscosity at Pb [cp].

    Returns:
        μo_undersat [cp].
    """
    m = 2.6 * p_psia ** 1.187 * math.exp(-11.513 - 8.98e-5 * p_psia)
    return mu_b * (p_psia / pb_psia) ** m


# ---------------------------------------------------------------------------
# Oil density (reservoir conditions)
# ---------------------------------------------------------------------------

def oil_density_lb_ft3(
    gamma_o: float,
    gamma_g: float,
    rs: float,
    bo: float,
) -> float:
    """Oil density at reservoir conditions via mass-balance (Craft & Hawkins).

    Formula:
        ρo = (62.4·γo + 0.0136·Rs·γg) / Bo    [lb/ft³]

    Args:
        gamma_o: Oil specific gravity (water = 1) [dimensionless].
        gamma_g: Gas specific gravity (air = 1) [dimensionless].
        rs:      Solution GOR [scf/STB].
        bo:      Oil FVF [bbl/STB].

    Returns:
        ρo [lb/ft³].
    """
    return (62.4 * gamma_o + 0.0136 * rs * gamma_g) / bo
