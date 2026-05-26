"""Tests for src/services/rta_match_params_service.py."""

from __future__ import annotations

import math

import pytest

from src.rta.models import RTAConfig
from src.services.rta_match_params_service import (
    RTAMatchParams,
    _C_N_DYN,
    _compute_kh,
    _compute_n_dyn,
    _compute_n_vol,
    _ln_re_rw_term,
    _resolve_drainage_geometry,
    compute_match_params,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _config(
    re_ft: float | None = 1000.0,
    area_acres: float | None = None,
    h_ft: float = 50.0,
    phi_frac: float = 0.18,
    rw_ft: float = 0.328,
    mu_o_cp: float = 2.0,
    Bo_rb_stb: float = 1.20,
    swi_frac: float | None = None,
) -> RTAConfig:
    return RTAConfig(
        well_id="W-TEST",
        pi_psia=3500.0,
        h_ft=h_ft,
        phi_frac=phi_frac,
        rw_ft=rw_ft,
        re_ft=re_ft,
        area_acres=area_acres,
        mu_o_cp=mu_o_cp,
        Bo_rb_stb=Bo_rb_stb,
        swi_frac=swi_frac,
    )


# ---------------------------------------------------------------------------
# _resolve_drainage_geometry
# ---------------------------------------------------------------------------

def test_resolve_uses_re_ft_when_provided() -> None:
    re, area = _resolve_drainage_geometry(_config(re_ft=1000.0, area_acres=None))
    assert re == pytest.approx(1000.0)
    assert area == pytest.approx(math.pi * 1000.0**2 / 43560.0, rel=1e-6)


def test_resolve_uses_area_acres_when_re_absent() -> None:
    re, area = _resolve_drainage_geometry(_config(re_ft=None, area_acres=40.0))
    assert area == pytest.approx(40.0)
    assert re == pytest.approx(math.sqrt(40.0 * 43560.0 / math.pi), rel=1e-6)


def test_resolve_prefers_re_ft_over_area() -> None:
    re, _ = _resolve_drainage_geometry(_config(re_ft=500.0, area_acres=100.0))
    assert re == pytest.approx(500.0)


def test_resolve_returns_none_when_both_absent() -> None:
    re, area = _resolve_drainage_geometry(_config(re_ft=None, area_acres=None))
    assert re is None
    assert area is None


# ---------------------------------------------------------------------------
# _ln_re_rw_term
# ---------------------------------------------------------------------------

def test_ln_re_rw_term_known_value() -> None:
    re_ft = math.e * 0.328  # ln(re/rw) = 1.0 exactly
    result = _ln_re_rw_term(re_ft=re_ft, rw_ft=0.328)
    assert result == pytest.approx(0.5, rel=1e-6)


def test_ln_re_rw_term_raises_when_re_le_rw() -> None:
    with pytest.raises(ValueError, match="re_ft"):
        _ln_re_rw_term(re_ft=0.1, rw_ft=0.328)


# ---------------------------------------------------------------------------
# _compute_kh
# ---------------------------------------------------------------------------

def test_compute_kh_dimensional_consistency() -> None:
    # With y_mult=1, kh = 141.2 * mu * Bo * ln_term
    kh = _compute_kh(mu_o_cp=1.0, Bo_rb_stb=1.0, ln_term=1.0, effective_y_multiplier=1.0)
    assert kh == pytest.approx(141.2, rel=1e-6)


def test_compute_kh_scales_inversely_with_y_mult() -> None:
    kh_base = _compute_kh(
        mu_o_cp=2.0, Bo_rb_stb=1.2, ln_term=5.0, effective_y_multiplier=1.0
    )
    kh_double = _compute_kh(
        mu_o_cp=2.0, Bo_rb_stb=1.2, ln_term=5.0, effective_y_multiplier=2.0
    )
    assert kh_base == pytest.approx(2.0 * kh_double, rel=1e-6)


def test_compute_kh_raises_on_nonpositive_multiplier() -> None:
    with pytest.raises(ValueError):
        _compute_kh(mu_o_cp=1.0, Bo_rb_stb=1.0, ln_term=1.0, effective_y_multiplier=0.0)


# ---------------------------------------------------------------------------
# _compute_n_vol
# ---------------------------------------------------------------------------

def test_compute_n_vol_known_case() -> None:
    # 1-acre, 1-ft thick, phi=1, Bo=1, Swi=0: N = 43560/5.615 = 7758 STB
    area_ft2 = 43_560.0
    n = _compute_n_vol(phi=1.0, h_ft=1.0, area_ft2=area_ft2, Bo_rb_stb=1.0, swi_frac=0.0)
    assert n == pytest.approx(7758.0, rel=1e-3)


def test_compute_n_vol_scales_linearly_with_area() -> None:
    base = _compute_n_vol(
        phi=0.18, h_ft=50.0, area_ft2=43560.0, Bo_rb_stb=1.2, swi_frac=0.0
    )
    double = _compute_n_vol(
        phi=0.18, h_ft=50.0, area_ft2=2 * 43560.0, Bo_rb_stb=1.2, swi_frac=0.0
    )
    assert double == pytest.approx(2.0 * base, rel=1e-6)


def test_compute_n_vol_swi_reduces_ooip() -> None:
    no_swi = _compute_n_vol(phi=0.2, h_ft=50.0, area_ft2=43560.0, Bo_rb_stb=1.0, swi_frac=0.0)
    with_swi = _compute_n_vol(phi=0.2, h_ft=50.0, area_ft2=43560.0, Bo_rb_stb=1.0, swi_frac=0.3)
    assert with_swi == pytest.approx(0.7 * no_swi, rel=1e-6)


# ---------------------------------------------------------------------------
# compute_match_params — integration
# ---------------------------------------------------------------------------

def test_returns_none_fields_when_no_drainage_geometry() -> None:
    result = compute_match_params(
        config=_config(re_ft=None, area_acres=None),
        effective_x_multiplier=0.5,
        effective_y_multiplier=2.0,
    )
    assert result.kh_md_ft is None
    assert result.k_md is None
    assert result.n_vol_stb is None
    assert len(result.warnings) > 0


def test_kh_and_k_computed_with_valid_inputs() -> None:
    result = compute_match_params(
        config=_config(re_ft=1000.0, h_ft=50.0, mu_o_cp=2.0, Bo_rb_stb=1.2),
        effective_x_multiplier=1.0,
        effective_y_multiplier=2.0,
        method="fetkovich",
    )
    assert result.kh_md_ft is not None
    assert result.kh_md_ft > 0
    assert result.k_md == pytest.approx(result.kh_md_ft / 50.0, rel=1e-9)


def test_k_scales_inversely_with_y_multiplier() -> None:
    cfg = _config(re_ft=1000.0)
    r1 = compute_match_params(
        config=cfg, effective_x_multiplier=1.0, effective_y_multiplier=3.0
    )
    r2 = compute_match_params(
        config=cfg, effective_x_multiplier=1.0, effective_y_multiplier=6.0
    )
    # kh ∝ 1/y_mult  →  kh(3) = 2 * kh(6)
    assert r1.kh_md_ft is not None and r2.kh_md_ft is not None
    assert r1.kh_md_ft == pytest.approx(2.0 * r2.kh_md_ft, rel=1e-6)


def test_n_vol_uses_swi_when_provided() -> None:
    no_swi = compute_match_params(
        config=_config(re_ft=1000.0, swi_frac=None),
        effective_x_multiplier=1.0,
        effective_y_multiplier=2.0,
    )
    with_swi = compute_match_params(
        config=_config(re_ft=1000.0, swi_frac=0.3),
        effective_x_multiplier=1.0,
        effective_y_multiplier=2.0,
    )
    assert no_swi.n_vol_stb is not None and with_swi.n_vol_stb is not None
    assert with_swi.n_vol_stb == pytest.approx(0.7 * no_swi.n_vol_stb, rel=1e-6)


def test_warns_when_y_multiplier_is_one() -> None:
    result = compute_match_params(
        config=_config(re_ft=1000.0),
        effective_x_multiplier=1.0,
        effective_y_multiplier=1.0,
    )
    assert any("y_multiplier = 1.0" in w for w in result.warnings)


def test_status_is_always_demo() -> None:
    result = compute_match_params(
        config=_config(re_ft=1000.0),
        effective_x_multiplier=0.5,
        effective_y_multiplier=3.0,
    )
    assert result.status == "demo"


def test_as_dict_contains_expected_keys() -> None:
    result = compute_match_params(
        config=_config(re_ft=1000.0),
        effective_x_multiplier=0.5,
        effective_y_multiplier=3.0,
    )
    d = result.as_dict()
    for key in ("well_id", "method", "kh_md_ft", "k_md", "n_vol_stb", "n_dyn_stb", "status", "warnings"):
        assert key in d


# ---------------------------------------------------------------------------
# _compute_n_dyn — unit tests
# ---------------------------------------------------------------------------
# Synthetic reference: kh=100 mD·ft, re=1000 ft, rw=0.3 ft, φ=0.15, h=50 ft
# Bo=1.2, μ=2 cP, ct=1e-5 1/psi, Swi=0.2
# N_vol ≈ 2.80e6 STB
# x_mult derived from Fetkovich tDd normalization → N_dyn ≈ N_vol within 2%

_KH = 100.0
_RE = 1000.0
_RW = 0.3
_BO = 1.2
_MU = 2.0
_CT = 1e-5
_SWI = 0.2
_PHI = 0.15
_H = 50.0
_LN_T = math.log(_RE / _RW) - 0.5                             # ≈ 7.612
_ALPHA_D = 0.5 * (_RE / _RW) ** 2 * _LN_T
_X_MULT = 0.000264 * (_KH / _H) / (_PHI * _MU * _CT * _RW**2 * _ALPHA_D)


def test_n_dyn_matches_n_vol_within_2pct() -> None:
    """N_dyn must reproduce the volumetric OOIP within 2% for the reference case."""
    n_dyn = _compute_n_dyn(_KH, _BO, _MU, _CT, _X_MULT, _LN_T, _SWI)
    n_vol = _compute_n_vol(_PHI, _H, math.pi * _RE**2, _BO, _SWI)
    assert abs(n_dyn - n_vol) / n_vol < 0.02


def test_n_dyn_scales_inversely_with_x_mult() -> None:
    n1 = _compute_n_dyn(_KH, _BO, _MU, _CT, _X_MULT, _LN_T, _SWI)
    n2 = _compute_n_dyn(_KH, _BO, _MU, _CT, _X_MULT * 2, _LN_T, _SWI)
    assert n2 == pytest.approx(n1 / 2, rel=1e-9)


def test_n_dyn_scales_linearly_with_kh() -> None:
    n1 = _compute_n_dyn(_KH, _BO, _MU, _CT, _X_MULT, _LN_T, _SWI)
    n2 = _compute_n_dyn(_KH * 2, _BO, _MU, _CT, _X_MULT, _LN_T, _SWI)
    assert n2 == pytest.approx(2 * n1, rel=1e-9)


def test_c_n_dyn_constant_value() -> None:
    assert _C_N_DYN == pytest.approx(2.0 * math.pi * 0.000264 / 5.615, rel=1e-9)


# ---------------------------------------------------------------------------
# compute_match_params — N_dyn integration
# ---------------------------------------------------------------------------

def _cfg_for_ndyn() -> RTAConfig:
    return RTAConfig(
        well_id="W-TEST",
        pi_psia=3500.0,
        rw_ft=_RW,
        re_ft=_RE,
        h_ft=_H,
        phi_frac=_PHI,
        ct_1psi=_CT,
        mu_o_cp=_MU,
        Bo_rb_stb=_BO,
        swi_frac=_SWI,
        CA=31.62,
    )


def test_n_dyn_populated_when_both_multipliers_adjusted() -> None:
    mp = compute_match_params(config=_cfg_for_ndyn(), effective_x_multiplier=_X_MULT, effective_y_multiplier=_LN_T * 141.2 * _MU * _BO / _KH)
    assert mp.n_dyn_stb is not None and mp.n_dyn_stb > 0


def test_n_dyn_none_when_x_mult_is_one() -> None:
    mp = compute_match_params(config=_cfg_for_ndyn(), effective_x_multiplier=1.0, effective_y_multiplier=5.0)
    assert mp.n_dyn_stb is None


def test_n_dyn_none_when_y_mult_is_one() -> None:
    mp = compute_match_params(config=_cfg_for_ndyn(), effective_x_multiplier=_X_MULT, effective_y_multiplier=1.0)
    assert mp.n_dyn_stb is None


def test_n_dyn_changes_when_x_mult_changes() -> None:
    y = _LN_T * 141.2 * _MU * _BO / _KH
    mp1 = compute_match_params(config=_cfg_for_ndyn(), effective_x_multiplier=_X_MULT, effective_y_multiplier=y)
    mp2 = compute_match_params(config=_cfg_for_ndyn(), effective_x_multiplier=_X_MULT / 2, effective_y_multiplier=y)
    assert mp1.n_dyn_stb is not None and mp2.n_dyn_stb is not None
    assert mp2.n_dyn_stb == pytest.approx(2 * mp1.n_dyn_stb, rel=1e-6)
