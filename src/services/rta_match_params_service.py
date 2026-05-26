"""Compute reservoir parameters from a visual RTA type-curve match.

The match is expressed as two multipliers from the M4 joystick overlay:

    effective_x_multiplier  =  tDd_MP  / MBT_MP        (dimensionless/days)
    effective_y_multiplier  =  qDd_MP  / (q/Dp)_MP     (days·psi/STB)

Parameters derived here:
    kh    [mD·ft]  — from the Y-multiplier via the Darcy flow equation
    k     [mD]     — kh / h
    N_vol [STB]    — volumetric OOIP from re or drainage area in RTAConfig
    N_dyn [STB]    — dynamic OOIP from the match position (x AND y multipliers)

N_dyn derivation (BDF time-normalization approach):
    From Fetkovich tDd definition:
        αD   = 0.5·(re/rw)²·[ln(re/rw) - 0.5]      (approx for re >> rw)
        tDd  = 0.000264·k·MBT / (φ·μ·ct·rw²·αD)
    At match: x_mult = tDd/MBT  →  αD = 0.000264·k / (φ·μ·ct·rw²·x_mult)
    Since αD ≈ 0.5·(re/rw)²·ln_term and A = π·re²:
        A_dyn = 2π·0.000264·k / (φ·μ·ct·x_mult·ln_term)
        N_dyn = φ·h·A_dyn·(1-Swi) / (5.615·Bo)
              = C·(1-Swi)·kh / (Bo·μ·ct·x_mult·ln_term)
    where C = 2π·0.000264/5.615 ≈ 2.954e-4.
    N_dyn changes with every joystick click (both x and y multipliers update it).

STATUS: all outputs are marked DEMO until type curves are digitized and
validated.  Do not use for final reservoir characterization.

Reference equations:
    Fetkovich (SPE 4629, 1980) — qD = 141.2·q·μ·Bo / (kh·Δp)  [Ec. 6]
    Palacio-Blasingame (SPE 25909, 1993) — Appendix A
    Agarwal-Gardner (SPE 49222, 1998) — Appendix equations A-6/A-8
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.rta.models import RTAConfig


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RTAMatchParams:
    """Reservoir parameters inferred from a visual type-curve match.

    All numerical fields are None when the required inputs are missing
    (e.g. re_ft and area_acres both absent → no drainage geometry → no kh).
    """

    well_id: str
    method: str

    # Geometry used for the calculation
    re_ft: float | None
    area_acres: float | None
    ln_re_rw_term: float | None     # ln(re/rw) - 0.5

    # Primary match parameters
    kh_md_ft: float | None          # mD·ft
    k_md: float | None              # mD

    # Volumetric OOIP (does NOT come from the match — uses config geometry)
    n_vol_stb: float | None         # STB

    # Match multipliers used (logged for reproducibility)
    effective_x_multiplier: float
    effective_y_multiplier: float

    # Dynamic OOIP from the match position (updates with every joystick click)
    # None until both x_mult and y_mult have been adjusted from 1.0
    n_dyn_stb: float | None = None  # STB

    status: str = "demo"
    notes: str = (
        "DEMO — curvas tipo aún no digitalizadas/validadas. "
        "No usar para interpretación técnica."
    )
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "well_id": self.well_id,
            "method": self.method,
            "status": self.status,
            "re_ft": self.re_ft,
            "area_acres": self.area_acres,
            "ln_re_rw_term": self.ln_re_rw_term,
            "kh_md_ft": self.kh_md_ft,
            "k_md": self.k_md,
            "n_vol_stb": self.n_vol_stb,
            "n_dyn_stb": self.n_dyn_stb,
            "effective_x_multiplier": self.effective_x_multiplier,
            "effective_y_multiplier": self.effective_y_multiplier,
            "notes": self.notes,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FT2_PER_ACRE = 43_560.0
_BBL_PER_FT3 = 1.0 / 5.615
# N_dyn constant: 2π × 0.000264 / 5.615  (field-units BDF time normalization)
_C_N_DYN = 2.0 * math.pi * 0.000264 / 5.615


def _resolve_drainage_geometry(
    config: RTAConfig,
) -> tuple[float | None, float | None]:
    """Return (re_ft, area_acres) using whichever the user supplied.

    If re_ft is provided, area is back-computed; if area_acres is provided,
    re is back-computed from a circular assumption.  If neither, returns (None, None).
    """
    if config.re_ft is not None and config.re_ft > 0:
        re = config.re_ft
        area = math.pi * re**2 / _FT2_PER_ACRE
        return re, area

    if config.area_acres is not None and config.area_acres > 0:
        area = config.area_acres
        re = math.sqrt(area * _FT2_PER_ACRE / math.pi)
        return re, area

    return None, None


def _ln_re_rw_term(re_ft: float, rw_ft: float) -> float:
    """Return ln(re/rw) - 0.5 (Fetkovich/Blasingame/A-G drainage term)."""
    if re_ft <= rw_ft:
        raise ValueError(
            f"re_ft ({re_ft:.2f}) must be greater than rw_ft ({rw_ft:.4f})."
        )
    return math.log(re_ft / rw_ft) - 0.5


def _compute_kh(
    mu_o_cp: float,
    Bo_rb_stb: float,
    ln_term: float,
    effective_y_multiplier: float,
) -> float:
    """kh [mD·ft] from the Y match multiplier.

    Derivation (field units):
        qD  = 141.2 · q · μ · Bo / (kh · Δp)        [Fetkovich Ec. 6]
        qDd = qD · [ln(re/rw) - ½]                   [Fetkovich Ec. 21]
        y_mult = qDd / (q/Δp)

    Solving for kh:
        kh = 141.2 · μ · Bo · [ln(re/rw) - ½] / y_mult
    """
    if effective_y_multiplier <= 0:
        raise ValueError("effective_y_multiplier must be positive.")
    return 141.2 * mu_o_cp * Bo_rb_stb * ln_term / effective_y_multiplier


def _compute_n_vol(
    phi: float,
    h_ft: float,
    area_ft2: float,
    Bo_rb_stb: float,
    swi_frac: float,
) -> float:
    """Volumetric OOIP [STB].

    N = φ · h · A · (1 - Swi) / (5.615 · Bo)
    """
    pore_vol_bbl = phi * h_ft * area_ft2 * _BBL_PER_FT3 * (1.0 - swi_frac)
    return pore_vol_bbl / Bo_rb_stb


def _compute_n_dyn(
    kh_md_ft: float,
    Bo_rb_stb: float,
    mu_o_cp: float,
    ct_1psi: float,
    effective_x_multiplier: float,
    ln_term: float,
    swi_frac: float,
) -> float:
    """Dynamic OOIP [STB] from the type-curve match position.

    Derived from the BDF time-normalization of the Fetkovich/Blasingame type
    curves (see module docstring for full derivation):

        N_dyn = C · (1-Swi) · kh / (Bo · μ · ct · x_mult · ln_term)

    where C = 2π · 0.000264 / 5.615 ≈ 2.954e-4.

    Unlike N_vol, this value updates with every joystick click because it
    depends on both effective_x_multiplier (time-axis scaling) and kh (which
    comes from effective_y_multiplier).
    """
    return (
        _C_N_DYN
        * (1.0 - swi_frac)
        * kh_md_ft
        / (Bo_rb_stb * mu_o_cp * ct_1psi * effective_x_multiplier * ln_term)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_match_params(
    *,
    config: RTAConfig,
    effective_x_multiplier: float,
    effective_y_multiplier: float,
    method: str = "fetkovich",
) -> RTAMatchParams:
    """Derive reservoir parameters from joystick match multipliers + RTAConfig.

    Args:
        config:                 RTAConfig with reservoir/fluid inputs.
        effective_x_multiplier: ManualMatchConfig.effective_x_multiplier.
        effective_y_multiplier: ManualMatchConfig.effective_y_multiplier.
        method:                 RTA method label (fetkovich/palacio_blasingame/
                                agarwal_gardner). Used only for labelling.

    Returns:
        RTAMatchParams with kh, k, and volumetric N.  Fields are None when
        the required geometry inputs are missing.
    """
    warnings: list[str] = []
    re_ft, area_acres = _resolve_drainage_geometry(config)

    # --- Drainage geometry ---
    if re_ft is None:
        warnings.append(
            "re_ft y area_acres no definidos — no es posible calcular kh ni N. "
            "Ingresa al menos uno en el panel de configuración."
        )
        return RTAMatchParams(
            well_id=config.well_id,
            method=method,
            re_ft=None,
            area_acres=None,
            ln_re_rw_term=None,
            kh_md_ft=None,
            k_md=None,
            n_vol_stb=None,
            n_dyn_stb=None,
            effective_x_multiplier=effective_x_multiplier,
            effective_y_multiplier=effective_y_multiplier,
            warnings=warnings,
        )

    # --- ln(re/rw) - 0.5 ---
    try:
        ln_term = _ln_re_rw_term(re_ft, config.rw_ft)
    except ValueError as exc:
        warnings.append(str(exc))
        return RTAMatchParams(
            well_id=config.well_id,
            method=method,
            re_ft=re_ft,
            area_acres=area_acres,
            ln_re_rw_term=None,
            kh_md_ft=None,
            k_md=None,
            n_vol_stb=None,
            n_dyn_stb=None,
            effective_x_multiplier=effective_x_multiplier,
            effective_y_multiplier=effective_y_multiplier,
            warnings=warnings,
        )

    # --- kh and k ---
    kh: float | None = None
    k: float | None = None

    if effective_y_multiplier != 1.0:
        try:
            kh = _compute_kh(
                mu_o_cp=config.mu_o_cp,
                Bo_rb_stb=config.Bo_rb_stb,
                ln_term=ln_term,
                effective_y_multiplier=effective_y_multiplier,
            )
            k = kh / config.h_ft
        except ValueError as exc:
            warnings.append(f"kh no calculable: {exc}")
    else:
        warnings.append(
            "y_multiplier = 1.0 (match en posición inicial) — "
            "kh no tiene significado físico hasta ajustar la nube."
        )

    # --- Volumetric OOIP ---
    area_ft2 = (area_acres or 0.0) * _FT2_PER_ACRE
    swi = config.swi_frac if config.swi_frac is not None else 0.0
    if swi == 0.0 and config.swi_frac is None:
        warnings.append(
            "Swi no definido — N volumétrico calculado con Swi = 0 "
            "(cota superior, sobreestima OOIP)."
        )

    n_vol = _compute_n_vol(
        phi=config.phi_frac,
        h_ft=config.h_ft,
        area_ft2=area_ft2,
        Bo_rb_stb=config.Bo_rb_stb,
        swi_frac=swi,
    )

    # --- Dynamic OOIP from match (requires kh AND x_mult adjusted from 1.0) ---
    n_dyn: float | None = None
    if kh is not None and effective_x_multiplier != 1.0 and ln_term > 0:
        n_dyn = _compute_n_dyn(
            kh_md_ft=kh,
            Bo_rb_stb=config.Bo_rb_stb,
            mu_o_cp=config.mu_o_cp,
            ct_1psi=config.ct_1psi,
            effective_x_multiplier=effective_x_multiplier,
            ln_term=ln_term,
            swi_frac=swi,
        )

    return RTAMatchParams(
        well_id=config.well_id,
        method=method,
        re_ft=re_ft,
        area_acres=area_acres,
        ln_re_rw_term=ln_term,
        kh_md_ft=kh,
        k_md=k,
        n_vol_stb=n_vol,
        n_dyn_stb=n_dyn,
        effective_x_multiplier=effective_x_multiplier,
        effective_y_multiplier=effective_y_multiplier,
        warnings=warnings,
    )
