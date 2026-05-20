"""Synthetic case validation — end-to-end parameter recovery test.

Strategy
--------
Generate a production history from KNOWN reservoir parameters using the
exponential-decline Fetkovich model (BDF, b=0, constant Pwf).  Then run the
full RTA pipeline and verify that the recovered kh, k and N match the inputs
to within engineering tolerance.

This replaces the traditional "compare against Harmony" validation with a
fully self-contained ground-truth test that can be reproduced without any
third-party software or license.

Physical basis
--------------
For pseudo-steady state exponential decline at constant Pwf:

    q(t) = qi · exp(-Di · t)
    qi   = kh · Δp / (141.2 · μ · Bo · [ln(re/rw) - ½])   [Darcy, field units]

The y-multiplier that maps (q/Δp) → qDd is constant for all t in BDF:

    y_mult_true = Δp / qi = 141.2 · μ · Bo · [ln(re/rw) - ½] / kh

Applying y_mult_true in compute_match_params must therefore recover kh_true
exactly (it is an algebraic identity in the Darcy equation, not a curve fit).
Any deviation > 0.1 % indicates a bug in the unit conversion chain.
"""

from __future__ import annotations

import math

import pytest

from src.rta.models import RTAConfig
from src.rta_type_curves.models import RTATypeCurveMethod
from src.services.rta_match_params_service import compute_match_params
from src.services.rta_synthetic_case import SyntheticCase, generate_exponential_decline
from src.services.rta_transform_service import compute_rta_transforms


# ---------------------------------------------------------------------------
# Shared synthetic case fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth() -> SyntheticCase:
    """Standard synthetic case used across all validation tests.

    Parameters chosen to represent a moderate-permeability Llanos oil well:
        k=5 mD, h=50 ft, re=2000 ft, φ=0.15, pi=3000 psia, pwf=1000 psia
    """
    return generate_exponential_decline(
        well_id="SYNTH-001",
        k_md=5.0,
        h_ft=50.0,
        re_ft=2000.0,
        rw_ft=0.328,
        phi_frac=0.15,
        pi_psia=3000.0,
        pwf_psia=1000.0,
        Bo_rb_stb=1.2,
        mu_o_cp=2.5,
        Di_per_day=0.005,
        n_months=36,
    )


@pytest.fixture(scope="module")
def rta_config(synth: SyntheticCase) -> RTAConfig:
    """RTAConfig built from the synthetic case's true parameters."""
    return RTAConfig(
        well_id=synth.well_id,
        pi_psia=synth.pi_psia,
        phi_frac=0.15,
        h_ft=synth.h_ft,
        ct_1psi=15e-6,
        rw_ft=synth.rw_ft,
        re_ft=synth.re_ft,
        Bo_rb_stb=synth.Bo_rb_stb,
        mu_o_cp=synth.mu_o_cp,
        swi_frac=0.0,
    )


# ---------------------------------------------------------------------------
# 1. Synthetic case structural checks
# ---------------------------------------------------------------------------

class TestSyntheticCaseStructure:

    def test_history_has_required_columns(self, synth: SyntheticCase) -> None:
        required = {"well_id", "date", "qo_stb_d", "pwf_used_psia"}
        assert required.issubset(set(synth.history_df.columns))

    def test_history_has_expected_row_count(self, synth: SyntheticCase) -> None:
        assert len(synth.history_df) == 36

    def test_rates_are_positive_and_declining(self, synth: SyntheticCase) -> None:
        rates = synth.history_df["qo_stb_d"].tolist()
        assert all(q > 0 for q in rates)
        # Each rate should be strictly less than the previous (exponential decline)
        for i in range(1, len(rates)):
            assert rates[i] < rates[i - 1], f"Rate not declining at index {i}"

    def test_pwf_is_constant(self, synth: SyntheticCase) -> None:
        pwf_values = synth.history_df["pwf_used_psia"].unique()
        assert len(pwf_values) == 1
        assert pwf_values[0] == synth.pwf_psia

    def test_qi_matches_darcy_equation(self, synth: SyntheticCase) -> None:
        """Verify qi = kh·Δp / (141.2·μ·Bo·[ln(re/rw)-0.5]) in field units."""
        expected_qi = (
            synth.kh_md_ft * synth.delta_p_psia
            / (141.2 * synth.mu_o_cp * synth.Bo_rb_stb * synth.ln_re_rw_term)
        )
        assert synth.qi_stb_d == pytest.approx(expected_qi, rel=1e-9)

    def test_y_multiplier_true_equals_delta_p_over_qi(self, synth: SyntheticCase) -> None:
        """y_mult_true = Δp/qi — the constant that maps (q/Δp) to qDd in BDF."""
        assert synth.y_multiplier_true == pytest.approx(
            synth.delta_p_psia / synth.qi_stb_d, rel=1e-9
        )

    def test_y_multiplier_true_equals_darcy_constant_over_kh(self, synth: SyntheticCase) -> None:
        """y_mult_true = 141.2·μ·Bo·[ln(re/rw)-0.5] / kh (Fetkovich Ec. 6/21)."""
        expected = (
            141.2 * synth.mu_o_cp * synth.Bo_rb_stb * synth.ln_re_rw_term
            / synth.kh_md_ft
        )
        assert synth.y_multiplier_true == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. RTA transforms on synthetic data
# ---------------------------------------------------------------------------

class TestRTATransformsOnSyntheticData:

    def test_transforms_return_points(self, synth: SyntheticCase) -> None:
        points = compute_rta_transforms(
            dataframe=synth.history_df,
            pi_psia=synth.pi_psia,
            methods=[RTATypeCurveMethod.FETKOVICH],
        )
        assert len(points) > 0

    def test_normalized_rate_equals_q_over_delta_p(self, synth: SyntheticCase) -> None:
        points = compute_rta_transforms(
            dataframe=synth.history_df,
            pi_psia=synth.pi_psia,
            methods=[RTATypeCurveMethod.FETKOVICH],
        )
        for p in points:
            expected = p.qo_stb_d / synth.delta_p_psia
            assert p.normalized_rate == pytest.approx(expected, rel=1e-6)

    def test_qDd_in_range_0_to_1(self, synth: SyntheticCase) -> None:
        """Dimensionless decline rate qDd = normalized_rate * y_mult must be in (0,1]."""
        points = compute_rta_transforms(
            dataframe=synth.history_df,
            pi_psia=synth.pi_psia,
            methods=[RTATypeCurveMethod.FETKOVICH],
        )
        for p in points:
            qDd = p.normalized_rate * synth.y_multiplier_true
            assert 0 < qDd <= 1.0 + 1e-9, f"qDd={qDd:.4f} out of (0,1] at t={p.date}"

    def test_mbt_is_positive_and_increasing(self, synth: SyntheticCase) -> None:
        points = sorted(
            compute_rta_transforms(
                dataframe=synth.history_df,
                pi_psia=synth.pi_psia,
                methods=[RTATypeCurveMethod.FETKOVICH],
            ),
            key=lambda p: p.date,
        )
        mbts = [p.material_balance_time for p in points]
        assert all(m > 0 for m in mbts)
        for i in range(1, len(mbts)):
            assert mbts[i] > mbts[i - 1]


# ---------------------------------------------------------------------------
# 3. kh and k recovery — core validation
# ---------------------------------------------------------------------------

class TestKhRecovery:
    """Apply the theoretical y_multiplier and verify kh/k are recovered exactly.

    Tolerance: 0.1 % (limited by floating-point arithmetic in field-unit constants).
    This is the primary validation that the Darcy-equation chain has no bugs.
    """

    def test_kh_recovered_within_tolerance(
        self, synth: SyntheticCase, rta_config: RTAConfig
    ) -> None:
        match = compute_match_params(
            config=rta_config,
            effective_x_multiplier=1.0,          # x_mult doesn't affect kh
            effective_y_multiplier=synth.y_multiplier_true,
            method="fetkovich",
        )
        assert match.kh_md_ft is not None
        assert match.kh_md_ft == pytest.approx(synth.kh_md_ft, rel=1e-3), (
            f"Recovered kh={match.kh_md_ft:.2f} mD·ft, "
            f"expected {synth.kh_md_ft:.2f} mD·ft"
        )

    def test_k_recovered_within_tolerance(
        self, synth: SyntheticCase, rta_config: RTAConfig
    ) -> None:
        match = compute_match_params(
            config=rta_config,
            effective_x_multiplier=1.0,
            effective_y_multiplier=synth.y_multiplier_true,
            method="fetkovich",
        )
        assert match.k_md is not None
        assert match.k_md == pytest.approx(synth.k_md, rel=1e-3), (
            f"Recovered k={match.k_md:.4f} mD, expected {synth.k_md:.4f} mD"
        )

    def test_kh_recovery_is_method_agnostic(
        self, synth: SyntheticCase, rta_config: RTAConfig
    ) -> None:
        """kh formula is the same for all three methods — result must not change."""
        kh_values = {}
        for method in ("fetkovich", "palacio_blasingame", "agarwal_gardner"):
            match = compute_match_params(
                config=rta_config,
                effective_x_multiplier=1.0,
                effective_y_multiplier=synth.y_multiplier_true,
                method=method,
            )
            kh_values[method] = match.kh_md_ft

        ref = kh_values["fetkovich"]
        for method, kh in kh_values.items():
            assert kh == pytest.approx(ref, rel=1e-9), (
                f"kh differs between fetkovich ({ref}) and {method} ({kh})"
            )


# ---------------------------------------------------------------------------
# 4. Volumetric OOIP recovery
# ---------------------------------------------------------------------------

class TestNVolRecovery:
    """N_vol is computed from geometry, not the match — verify the formula."""

    def test_n_vol_matches_synthetic_truth(
        self, synth: SyntheticCase, rta_config: RTAConfig
    ) -> None:
        match = compute_match_params(
            config=rta_config,
            effective_x_multiplier=1.0,
            effective_y_multiplier=synth.y_multiplier_true,
            method="fetkovich",
        )
        assert match.n_vol_stb is not None
        assert match.n_vol_stb == pytest.approx(synth.N_stb, rel=1e-6)

    def test_n_vol_in_realistic_range(
        self, synth: SyntheticCase, rta_config: RTAConfig
    ) -> None:
        """For the chosen geometry, OOIP should be in the tens of MM STB range."""
        match = compute_match_params(
            config=rta_config,
            effective_x_multiplier=1.0,
            effective_y_multiplier=synth.y_multiplier_true,
            method="fetkovich",
        )
        assert match.n_vol_stb is not None
        n_mm = match.n_vol_stb / 1e6
        assert 1.0 < n_mm < 100.0, f"N={n_mm:.1f} MM STB outside expected range"


# ---------------------------------------------------------------------------
# 5. Sensitivity: different k values scale kh linearly
# ---------------------------------------------------------------------------

class TestKhScaling:
    """kh ∝ 1/y_mult — doubling k should halve the y_multiplier and halve kh."""

    def test_doubling_k_halves_y_multiplier(self) -> None:
        case1 = generate_exponential_decline(
            k_md=5.0, h_ft=50.0, re_ft=2000.0, rw_ft=0.328,
            phi_frac=0.15, pi_psia=3000.0, pwf_psia=1000.0,
            Bo_rb_stb=1.2, mu_o_cp=2.5, Di_per_day=0.005,
        )
        case2 = generate_exponential_decline(
            k_md=10.0, h_ft=50.0, re_ft=2000.0, rw_ft=0.328,
            phi_frac=0.15, pi_psia=3000.0, pwf_psia=1000.0,
            Bo_rb_stb=1.2, mu_o_cp=2.5, Di_per_day=0.005,
        )
        assert case2.y_multiplier_true == pytest.approx(
            case1.y_multiplier_true / 2.0, rel=1e-9
        )

    def test_kh_recovery_for_low_permeability(self) -> None:
        """Verify recovery also works for tight reservoir (k=0.5 mD)."""
        case = generate_exponential_decline(
            k_md=0.5, h_ft=100.0, re_ft=1000.0, rw_ft=0.328,
            phi_frac=0.10, pi_psia=4000.0, pwf_psia=500.0,
            Bo_rb_stb=1.1, mu_o_cp=1.5, Di_per_day=0.002,
        )
        config = RTAConfig(
            well_id="SYNTH-TIGHT",
            pi_psia=case.pi_psia,
            phi_frac=0.10,
            h_ft=case.h_ft,
            ct_1psi=15e-6,
            rw_ft=case.rw_ft,
            re_ft=case.re_ft,
            Bo_rb_stb=case.Bo_rb_stb,
            mu_o_cp=case.mu_o_cp,
            swi_frac=0.0,
        )
        match = compute_match_params(
            config=config,
            effective_x_multiplier=1.0,
            effective_y_multiplier=case.y_multiplier_true,
            method="fetkovich",
        )
        assert match.kh_md_ft == pytest.approx(case.kh_md_ft, rel=1e-3)
        assert match.k_md == pytest.approx(case.k_md, rel=1e-3)
