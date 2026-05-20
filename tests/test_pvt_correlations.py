"""Tests for src/services/pvt_correlations.py and src/services/pvt_service.py.

Physical behaviour validated (not absolute values):
  - Monotonicity of Rs, Bo, μo with pressure
  - Round-trip Pb ↔ Rsb consistency
  - μo decreases below Pb (more dissolved gas → lower viscosity)
  - μo increases above Pb (undersaturated compression)
  - Bo peaks at Pb (maximum FVF at bubble point)
  - Density in physically realistic range
  - Correlation API boundary (VB uses different constants at API = 30)
"""

from __future__ import annotations

import math
import pytest

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
from src.services.pvt_service import PVTTableInput, PVTPressurePoint, compute_pvt_table


# ===========================================================================
# TestApiToSg
# ===========================================================================

class TestApiToSg:
    def test_water_api(self) -> None:
        """10 °API ≈ 1.0 SG (water)."""
        # 141.5 / (10 + 131.5) = 141.5 / 141.5 = 1.0
        assert api_to_sg(10.0) == pytest.approx(1.0, abs=1e-6)

    def test_30_api(self) -> None:
        # 141.5 / (30 + 131.5) = 141.5 / 161.5 ≈ 0.8762
        assert api_to_sg(30.0) == pytest.approx(0.8762, rel=0.001)

    def test_lighter_oil_lower_sg(self) -> None:
        """Heavier crude (lower API) has higher SG."""
        assert api_to_sg(40.0) < api_to_sg(20.0)


# ===========================================================================
# TestStandingCorrelation
# ===========================================================================

class TestStandingCorrelation:
    # Common fluid system: Llanos-style heavy-medium crude
    API = 14.5
    GG  = 0.75
    TF  = 195.0
    RSB = 120.0  # scf/STB at Pb

    def test_standing_pb_positive(self) -> None:
        pb = standing_pb(self.RSB, self.API, self.GG, self.TF)
        assert pb > 0, f"Pb must be positive, got {pb:.1f}"

    def test_standing_pb_round_trip(self) -> None:
        """Rs at Pb must equal Rsb (within 1 %)."""
        pb = standing_pb(self.RSB, self.API, self.GG, self.TF)
        rs_at_pb = standing_rs(pb, self.API, self.GG, self.TF)
        assert rs_at_pb == pytest.approx(self.RSB, rel=0.01)

    def test_standing_rs_increases_with_pressure(self) -> None:
        """Rs is monotonically increasing with pressure below Pb."""
        pb = standing_pb(self.RSB, self.API, self.GG, self.TF)
        pressures = [pb * f for f in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]]
        rs_vals = [standing_rs(p, self.API, self.GG, self.TF) for p in pressures]
        for i in range(len(rs_vals) - 1):
            assert rs_vals[i] < rs_vals[i + 1], (
                f"Rs not monotonic at pressures {pressures[i]:.0f}/{pressures[i+1]:.0f}"
            )

    def test_standing_bo_greater_than_one(self) -> None:
        """Bo must be > 1 for any reasonable Rs / temperature."""
        for rs in [50, 100, 200, 400, 600]:
            bo = standing_bo(rs, self.GG, api_to_sg(self.API), self.TF)
            assert bo > 1.0, f"Bo={bo:.4f} ≤ 1 at Rs={rs}"

    def test_standing_bo_increases_with_rs(self) -> None:
        """More dissolved gas expands the oil → higher Bo."""
        go = api_to_sg(self.API)
        bo_low  = standing_bo(50,  self.GG, go, self.TF)
        bo_high = standing_bo(400, self.GG, go, self.TF)
        assert bo_high > bo_low

    def test_higher_temp_gives_higher_pb(self) -> None:
        """At fixed Rsb, higher T gives higher Pb (oil needs more P to keep gas in solution)."""
        pb_low_t  = standing_pb(self.RSB, self.API, self.GG, 160.0)
        pb_high_t = standing_pb(self.RSB, self.API, self.GG, 250.0)
        assert pb_high_t > pb_low_t

    def test_higher_rsb_gives_higher_pb(self) -> None:
        """More dissolved gas → higher bubble-point pressure."""
        pb_low  = standing_pb(80,  self.API, self.GG, self.TF)
        pb_high = standing_pb(200, self.API, self.GG, self.TF)
        assert pb_high > pb_low

    def test_standing_32api_reasonable_pb(self) -> None:
        """Medium crude (32 °API): Pb with Rsb=300 should be 800–2500 psia."""
        pb = standing_pb(300.0, 32.0, 0.75, 180.0)
        assert 800 < pb < 2500, f"Pb={pb:.0f} outside expected range"


# ===========================================================================
# TestVasquezBeggsCorrelation
# ===========================================================================

class TestVasquezBeggsCorrelation:
    API = 14.5
    GG  = 0.75
    TF  = 195.0
    RSB = 120.0

    def test_vb_pb_positive(self) -> None:
        pb = vb_pb(self.RSB, self.API, self.GG, self.TF)
        assert pb > 0

    def test_vb_pb_round_trip(self) -> None:
        """Rs at Pb must equal Rsb (within 1 %)."""
        pb = vb_pb(self.RSB, self.API, self.GG, self.TF)
        rs_at_pb = vb_rs(pb, self.API, self.GG, self.TF)
        assert rs_at_pb == pytest.approx(self.RSB, rel=0.01)

    def test_vb_rs_monotonic(self) -> None:
        """VB Rs increases with pressure below Pb."""
        pb = vb_pb(self.RSB, self.API, self.GG, self.TF)
        pressures = [pb * f for f in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]]
        rs_vals = [vb_rs(p, self.API, self.GG, self.TF) for p in pressures]
        for i in range(len(rs_vals) - 1):
            assert rs_vals[i] < rs_vals[i + 1]

    def test_vb_api_boundary_heavy(self) -> None:
        """API ≤ 30 uses heavy-oil constants."""
        # Just verify it doesn't raise and returns positive value
        rs = vb_rs(1000.0, 30.0, 0.75, 180.0)
        assert rs > 0

    def test_vb_api_boundary_light(self) -> None:
        """API > 30 uses light-oil constants; discontinuity at 30 is expected."""
        rs_heavy = vb_rs(1000.0, 30.0,  0.75, 180.0)
        rs_light = vb_rs(1000.0, 30.01, 0.75, 180.0)
        # Different constants → different Rs; light oil typically gives lower Rs in VB
        assert rs_heavy != pytest.approx(rs_light, rel=0.001)

    def test_vb_bo_greater_than_one(self) -> None:
        for rs in [50, 100, 200, 400]:
            bo = vb_bo(rs, self.API, self.GG, self.TF)
            assert bo > 1.0, f"VB Bo={bo:.4f} ≤ 1 at Rs={rs}"

    def test_vb_bo_increases_with_rs(self) -> None:
        bo_low  = vb_bo(50,  self.API, self.GG, self.TF)
        bo_high = vb_bo(300, self.API, self.GG, self.TF)
        assert bo_high > bo_low


# ===========================================================================
# TestBoUndersat
# ===========================================================================

class TestBoUndersat:
    def test_bo_at_pb_equals_bo_b(self) -> None:
        """At P = Pb, undersaturated Bo must equal Bo at Pb."""
        bo_b = 1.25
        pb = 2000.0
        bo = bo_undersat(pb, bo_b, pb, co_psi=1.5e-5)
        assert bo == pytest.approx(bo_b, rel=1e-9)

    def test_bo_decreases_above_pb(self) -> None:
        """Bo decreases as P rises above Pb (oil compresses)."""
        bo_b = 1.25
        pb = 2000.0
        bo_2500 = bo_undersat(pb, bo_b, 2500.0, co_psi=1.5e-5)
        bo_4000 = bo_undersat(pb, bo_b, 4000.0, co_psi=1.5e-5)
        assert bo_2500 < bo_b
        assert bo_4000 < bo_2500

    def test_higher_co_gives_steeper_decline(self) -> None:
        pb = 2000.0
        bo_b = 1.25
        p = 3000.0
        bo_low_co  = bo_undersat(pb, bo_b, p, co_psi=5e-6)
        bo_high_co = bo_undersat(pb, bo_b, p, co_psi=3e-5)
        assert bo_high_co < bo_low_co


# ===========================================================================
# TestBeggsRobinsonViscosity
# ===========================================================================

class TestBeggsRobinsonViscosity:
    API = 14.5
    TF  = 195.0

    def test_mu_dead_positive(self) -> None:
        assert br_mu_dead(self.API, self.TF) > 0

    def test_mu_dead_heavy_crude_high(self) -> None:
        """Heavy crude (low API) has much higher μ_dead than medium crude."""
        mu_heavy  = br_mu_dead(10.0, 180.0)
        mu_medium = br_mu_dead(30.0, 180.0)
        assert mu_heavy > mu_medium * 5  # order-of-magnitude difference expected

    def test_mu_dead_decreases_with_temperature(self) -> None:
        mu_cold = br_mu_dead(self.API, 140.0)
        mu_hot  = br_mu_dead(self.API, 250.0)
        assert mu_cold > mu_hot

    def test_mu_sat_decreases_with_rs(self) -> None:
        """More dissolved gas → lower live-oil viscosity."""
        mu_dead = br_mu_dead(self.API, self.TF)
        mu_low_rs  = br_mu_sat(50,  mu_dead)
        mu_high_rs = br_mu_sat(400, mu_dead)
        assert mu_low_rs > mu_high_rs

    def test_mu_sat_positive(self) -> None:
        mu_dead = br_mu_dead(self.API, self.TF)
        assert br_mu_sat(100, mu_dead) > 0

    def test_mu_undersat_increases_with_pressure(self) -> None:
        """Above Pb, viscosity increases as pressure compresses the oil."""
        mu_b = 5.0
        pb   = 2000.0
        mu_2500 = br_mu_undersat(2500.0, pb, mu_b)
        mu_4000 = br_mu_undersat(4000.0, pb, mu_b)
        assert mu_2500 > mu_b
        assert mu_4000 > mu_2500

    def test_mu_undersat_at_pb_approximately_mu_b(self) -> None:
        """At P = Pb, undersaturated μo should be ≈ μ_b (continuity at Pb)."""
        mu_b = 5.0
        pb   = 2000.0
        mu_at_pb = br_mu_undersat(pb, pb, mu_b)
        assert mu_at_pb == pytest.approx(mu_b, rel=1e-9)


# ===========================================================================
# TestOilDensity
# ===========================================================================

class TestOilDensity:
    def test_density_physical_range(self) -> None:
        """Typical oil at reservoir conditions: 40–65 lb/ft³."""
        gamma_o = api_to_sg(14.5)
        rho = oil_density_lb_ft3(gamma_o, 0.75, 120.0, 1.10)
        assert 40.0 < rho < 65.0, f"Density {rho:.1f} lb/ft³ outside physical range"

    def test_higher_rs_increases_density_contribution(self) -> None:
        """More dissolved gas in numerator → higher density if Bo constant."""
        gamma_o = api_to_sg(14.5)
        rho_low  = oil_density_lb_ft3(gamma_o, 0.75, 50,  1.08)
        rho_high = oil_density_lb_ft3(gamma_o, 0.75, 250, 1.08)
        assert rho_high > rho_low

    def test_higher_bo_decreases_density(self) -> None:
        """Same mass in larger reservoir volume → lower density."""
        gamma_o = api_to_sg(30.0)
        rho_low_bo  = oil_density_lb_ft3(gamma_o, 0.75, 200, 1.40)
        rho_high_bo = oil_density_lb_ft3(gamma_o, 0.75, 200, 1.10)
        assert rho_high_bo > rho_low_bo


# ===========================================================================
# TestComputePvtTableService
# ===========================================================================

class TestComputePvtTableService:
    """End-to-end tests for the pvt_service compute_pvt_table function."""

    BASE = PVTTableInput(
        api=14.5,
        gamma_g=0.75,
        t_f=195.0,
        rsb_scf_stb=120.0,
        p_min_psia=50.0,
        p_max_psia=4000.0,
        n_points=40,
        correlation="standing",
    )

    def test_returns_positive_pb(self) -> None:
        pb, _ = compute_pvt_table(self.BASE)
        assert pb > 0

    def test_returns_nonempty_points(self) -> None:
        _, points = compute_pvt_table(self.BASE)
        assert len(points) > 0

    def test_pb_point_is_marked(self) -> None:
        """Exactly one point should be flagged as is_pb=True."""
        pb, points = compute_pvt_table(self.BASE)
        pb_points = [p for p in points if p.is_pb]
        assert len(pb_points) == 1
        assert pb_points[0].p_psia == pytest.approx(pb, abs=0.2)

    def test_rs_monotonic_below_pb(self) -> None:
        """Rs increases with pressure in the saturated regime."""
        pb, points = compute_pvt_table(self.BASE)
        sat = [p for p in points if p.regime == "saturated"]
        sat.sort(key=lambda p: p.p_psia)
        for i in range(len(sat) - 1):
            assert sat[i].rs_scf_stb <= sat[i + 1].rs_scf_stb + 1e-6

    def test_rs_constant_above_pb(self) -> None:
        """Rs is pinned at Rsb in the undersaturated regime."""
        pb, points = compute_pvt_table(self.BASE)
        usat = [p for p in points if p.regime == "undersaturated"]
        rsb = self.BASE.rsb_scf_stb
        for pt in usat:
            assert pt.rs_scf_stb == pytest.approx(rsb, rel=1e-6)

    def test_bo_peaks_at_pb(self) -> None:
        """Bo is maximum at Pb — above Pb it compresses, below Pb less gas dissolved."""
        pb, points = compute_pvt_table(self.BASE)
        pb_pt = next(p for p in points if p.is_pb)
        sat = [p for p in points if p.regime == "saturated" and not p.is_pb]
        usat = [p for p in points if p.regime == "undersaturated" and not p.is_pb]
        for p in sat:
            assert p.bo_rb_stb <= pb_pt.bo_rb_stb + 1e-4
        for p in usat:
            assert p.bo_rb_stb <= pb_pt.bo_rb_stb + 1e-4

    def test_mu_decreases_below_pb(self) -> None:
        """Viscosity decreases as P rises toward Pb (more dissolved gas)."""
        pb, points = compute_pvt_table(self.BASE)
        sat = sorted(
            [p for p in points if p.regime == "saturated"],
            key=lambda p: p.p_psia,
        )
        for i in range(len(sat) - 1):
            assert sat[i].mu_o_cp >= sat[i + 1].mu_o_cp - 1e-6

    def test_mu_increases_above_pb(self) -> None:
        """Viscosity increases as P rises above Pb (undersaturated)."""
        pb, points = compute_pvt_table(self.BASE)
        usat = sorted(
            [p for p in points if p.regime == "undersaturated"],
            key=lambda p: p.p_psia,
        )
        if len(usat) < 2:
            pytest.skip("Not enough undersaturated points (p_max too close to pb)")
        for i in range(len(usat) - 1):
            assert usat[i].mu_o_cp <= usat[i + 1].mu_o_cp + 1e-6

    def test_all_densities_positive_and_physical(self) -> None:
        """Density should be in 35–75 lb/ft³ for any sensible oil."""
        _, points = compute_pvt_table(self.BASE)
        for pt in points:
            assert 35.0 < pt.rho_o_lb_ft3 < 75.0, (
                f"Density {pt.rho_o_lb_ft3:.1f} lb/ft³ out of range at P={pt.p_psia:.0f}"
            )

    def test_vb_correlation_also_works(self) -> None:
        """Vasquez-Beggs path returns a valid table."""
        inp = PVTTableInput(
            api=14.5, gamma_g=0.75, t_f=195.0,
            rsb_scf_stb=120.0, correlation="vasquez_beggs",
        )
        pb, points = compute_pvt_table(inp)
        assert pb > 0
        assert any(p.is_pb for p in points)

    def test_invalid_api_raises(self) -> None:
        inp = PVTTableInput(api=-5, gamma_g=0.75, t_f=195.0, rsb_scf_stb=100.0)
        with pytest.raises(ValueError, match="api"):
            compute_pvt_table(inp)

    def test_invalid_gamma_g_raises(self) -> None:
        inp = PVTTableInput(api=14.5, gamma_g=0.0, t_f=195.0, rsb_scf_stb=100.0)
        with pytest.raises(ValueError, match="gamma_g"):
            compute_pvt_table(inp)

    def test_invalid_rsb_raises(self) -> None:
        inp = PVTTableInput(api=14.5, gamma_g=0.75, t_f=195.0, rsb_scf_stb=0.0)
        with pytest.raises(ValueError, match="rsb_scf_stb"):
            compute_pvt_table(inp)

    def test_medium_crude_standing(self) -> None:
        """Llanos-style medium crude (32 °API, 300 scf/STB) — sanity check."""
        inp = PVTTableInput(
            api=32.0, gamma_g=0.75, t_f=180.0,
            rsb_scf_stb=300.0,
            p_min_psia=50.0, p_max_psia=5000.0,
        )
        pb, points = compute_pvt_table(inp)
        assert 800 < pb < 2500
        pb_pt = next(p for p in points if p.is_pb)
        assert pb_pt.bo_rb_stb == pytest.approx(1.0, abs=0.6)   # Bo at Pb ≈ 1.1–1.5
        assert pb_pt.mu_o_cp > 0.1

    def test_heavy_crude_high_viscosity(self) -> None:
        """Heavy crude (10 °API) has high μ_dead and moderate Rs."""
        inp = PVTTableInput(
            api=10.0, gamma_g=0.70, t_f=200.0,
            rsb_scf_stb=60.0,
        )
        pb, points = compute_pvt_table(inp)
        pb_pt = next(p for p in points if p.is_pb)
        # Dead-oil viscosity for 10 °API is hundreds of cP
        assert pb_pt.mu_o_cp > 10.0
