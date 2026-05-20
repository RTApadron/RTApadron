"""Synthetic RTA case generator for analytical validation.

Generates production histories from KNOWN reservoir parameters using the
Fetkovich exponential-decline model (BDF, b=0, constant Pwf).  The output
can be fed into compute_rta_transforms + compute_match_params to verify that
the recovered parameters match the inputs to within numerical tolerance.

Physical model
--------------
Pseudo-steady state, radial drainage, constant flowing BHP (Pwf):

    qi  = kh · (pi - pwf) / (141.2 · μ · Bo · [ln(re/rw) - ½])   [STB/d]
    q(t)= qi · exp(-Di · t)                                         [STB/d]
    Np(t)= (qi / Di) · (1 - exp(-Di · t))                          [STB]
    pwf(t) = pwf_const                                              [psia]

The theoretical y_multiplier that maps (q/Δp) → qDd is:

    y_mult_true = 141.2 · μ · Bo · [ln(re/rw) - ½] / kh
                = Δp / qi

This is constant for all t in BDF, so applying y_mult_true via
compute_match_params should recover kh_true exactly (algebraic identity
in the field-unit Darcy equation).

STATUS: internal validation tool — not for production/field interpretation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyntheticCase:
    """Synthetic production history paired with its known true parameters."""

    history_df: pd.DataFrame        # columns: well_id, date, qo_stb_d, pwf_used_psia
    well_id: str

    # Reservoir geometry
    k_md: float                     # permeability [mD]
    h_ft: float                     # net pay [ft]
    kh_md_ft: float                 # = k * h
    re_ft: float                    # drainage radius [ft]
    rw_ft: float                    # wellbore radius [ft]
    ln_re_rw_term: float            # ln(re/rw) - 0.5

    # Fluid
    pi_psia: float                  # initial pressure [psia]
    pwf_psia: float                 # constant flowing BHP [psia]
    delta_p_psia: float             # = pi - pwf
    Bo_rb_stb: float
    mu_o_cp: float

    # Derived production parameters
    qi_stb_d: float                 # initial rate [STB/d]
    Di_per_day: float               # exponential decline rate [1/day]

    # Volumetric OOIP (Swi = 0)
    N_stb: float

    # The y-multiplier that maps (q/Δp) → qDd for this specific case
    y_multiplier_true: float        # = Δp / qi = 141.2·μ·Bo·ln_term / kh


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_BBL_PER_FT3 = 1.0 / 5.615


def generate_exponential_decline(
    *,
    well_id: str = "SYNTH-001",
    k_md: float,
    h_ft: float,
    re_ft: float,
    rw_ft: float,
    phi_frac: float,
    pi_psia: float,
    pwf_psia: float,
    Bo_rb_stb: float,
    mu_o_cp: float,
    Di_per_day: float,
    n_months: int = 36,
    start_date: date | None = None,
) -> SyntheticCase:
    """Generate synthetic exponential-decline history from known reservoir parameters.

    Args:
        well_id:     Well identifier string.
        k_md:        Permeability [mD].
        h_ft:        Net pay [ft].
        re_ft:       Drainage radius [ft].
        rw_ft:       Wellbore radius [ft].
        phi_frac:    Porosity (fraction).
        pi_psia:     Initial reservoir pressure [psia].
        pwf_psia:    Constant flowing BHP [psia].
        Bo_rb_stb:   Oil formation volume factor [RB/STB].
        mu_o_cp:     Oil viscosity [cp].
        Di_per_day:  Exponential decline rate [1/day].  Use a value consistent
                     with the reservoir parameters (e.g. 0.003–0.02 for typical oil).
        n_months:    Number of monthly production points to generate.
        start_date:  First production date (defaults to 2020-01-01).

    Returns:
        SyntheticCase with history_df and all true parameters for validation.
    """
    if pwf_psia >= pi_psia:
        raise ValueError(f"pwf_psia ({pwf_psia}) must be less than pi_psia ({pi_psia}).")
    if re_ft <= rw_ft:
        raise ValueError(f"re_ft ({re_ft}) must be greater than rw_ft ({rw_ft}).")
    if Di_per_day <= 0:
        raise ValueError(f"Di_per_day must be positive, got {Di_per_day}.")

    # Derived geometry / fluid
    ln_term = math.log(re_ft / rw_ft) - 0.5
    kh = k_md * h_ft
    delta_p = pi_psia - pwf_psia

    # Initial rate from Darcy equation (field units)
    # qi [STB/d] = kh [mD·ft] · Δp [psia] / (141.2 · μ [cp] · Bo [RB/STB] · ln_term)
    qi = kh * delta_p / (141.2 * mu_o_cp * Bo_rb_stb * ln_term)

    # Theoretical y-multiplier: y_mult = qDd / (q/Δp) = Δp / qi (constant in BDF)
    y_mult_true = delta_p / qi  # = 141.2 · μ · Bo · ln_term / kh

    # Volumetric OOIP (Swi = 0 for simplicity)
    area_ft2 = math.pi * re_ft ** 2
    N_stb = phi_frac * h_ft * area_ft2 * _BBL_PER_FT3 / Bo_rb_stb

    # Generate monthly production points (t=30, 60, 90, …  days)
    t0 = start_date or date(2020, 1, 1)
    rows = []
    for month in range(1, n_months + 1):
        t_days = month * 30.0
        q = qi * math.exp(-Di_per_day * t_days)
        rows.append({
            "well_id": well_id,
            "date": (t0 + timedelta(days=int(t_days))).isoformat(),
            "qo_stb_d": round(q, 6),
            "pwf_used_psia": pwf_psia,
        })

    history_df = pd.DataFrame(rows)

    return SyntheticCase(
        history_df=history_df,
        well_id=well_id,
        k_md=k_md,
        h_ft=h_ft,
        kh_md_ft=kh,
        re_ft=re_ft,
        rw_ft=rw_ft,
        ln_re_rw_term=ln_term,
        pi_psia=pi_psia,
        pwf_psia=pwf_psia,
        delta_p_psia=delta_p,
        Bo_rb_stb=Bo_rb_stb,
        mu_o_cp=mu_o_cp,
        qi_stb_d=qi,
        Di_per_day=Di_per_day,
        N_stb=N_stb,
        y_multiplier_true=y_mult_true,
    )
