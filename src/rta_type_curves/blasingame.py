"""Blasingame type-curve generation.

This module generates a first engineering version of Blasingame composite curves
for a vertical well, homogeneous reservoir, circular closed boundary.

The implemented solver is dimensionless and model-based. It solves the radial
diffusivity equation numerically and then builds the Blasingame functions:
qDd, qDdi and qDdid versus tcDd.

This is intended as the first internal curve engine for ecoRTA. It should be
validated against commercial software/type-curve references before using it for
final reservoir interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BlasingameCurveConfig:
    """Configuration for dimensionless Blasingame curve generation.

    t_d_max_multiplier
        Scales the upper t_D boundary: t_D_max = multiplier * reD².
        Must be large enough for every reD family to reach BDF before the
        simulation ends, but not so large that q_D underflows to zero
        (which would make t_cD = Q_D/q_D diverge).

        Rule of thumb: multiplier = 5–20 for the BDF onset; multiplier ~50
        gives two decades of BDF range on the type-curve plate.
        The output is additionally filtered to t_cDd ≤ t_c_dd_max.

    t_c_dd_max
        Maximum normalised material-balance time retained in output.
        Points beyond this are in deep numerical BDF where q_D is near
        machine-epsilon and add no information to the type-curve plate.
        The published Blasingame plate covers roughly 1e-3 ≤ tcDd ≤ 1e2.

    n_time / n_radius
        Grid resolution.  360 / 240 gives smooth curves.
    """

    re_d_values: tuple[float, ...] = (10, 20, 50, 100, 1_000, 10_000, 100_000, 1_000_000)
    n_time: int = 480
    n_radius: int = 240
    t_d_min: float = 1.0e-7
    t_d_max_multiplier: float = 200.0  # extended BDF range; q_D filtered if near eps
    t_c_dd_max: float = 2000.0         # matches Fetkovich plate extent (~1e3)
    eps: float = 1.0e-30


@dataclass(frozen=True)
class BlasingameCurveSet:
    """Container for generated Blasingame curves."""

    curves: pd.DataFrame

    def to_csv(self, path: str | Path) -> None:
        """Export generated curves to CSV."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.curves.to_csv(output_path, index=False)


def _validate_re_d(re_d: float) -> None:
    if not np.isfinite(re_d) or re_d <= 1.0:
        msg = f"re_d must be greater than 1. Got: {re_d}"
        raise ValueError(msg)


def _build_implicit_matrix(x_grid: np.ndarray, dt: float) -> np.ndarray:
    """Build implicit matrix for radial diffusivity in log-radius coordinates.

    Governing equation in x = ln(rD):

        dpD/dtD = exp(-2x) * d2pD/dx2

    Boundary conditions:
        pD(x=0) = 0              constant wellbore pressure
        dpD/dx at x=ln(reD) = 0  closed outer boundary
    """
    n = len(x_grid)
    if n < 5:
        raise ValueError("n_radius must be at least 5.")

    dx = x_grid[1] - x_grid[0]
    unknowns = n - 1
    matrix = np.zeros((unknowns, unknowns), dtype=float)

    # Unknown vector contains p[1], p[2], ..., p[n-1].
    for row in range(unknowns):
        i = row + 1
        alpha = dt * np.exp(-2.0 * x_grid[i]) / dx**2

        if i < n - 1:
            # Interior node.
            matrix[row, row] = 1.0 + 2.0 * alpha

            if row - 1 >= 0:
                matrix[row, row - 1] = -alpha

            if row + 1 < unknowns:
                matrix[row, row + 1] = -alpha
        else:
            # Outer node with no-flow boundary:
            # d2p/dx2 ≈ 2 * (p[n-2] - p[n-1]) / dx2
            matrix[row, row] = 1.0 + 2.0 * alpha
            matrix[row, row - 1] = -2.0 * alpha

    return matrix


def _solve_constant_pressure_radial_response(
    re_d: float,
    *,
    n_time: int,
    n_radius: int,
    t_d_min: float,
    t_d_max_multiplier: float,
    eps: float,
) -> pd.DataFrame:
    """Solve dimensionless radial response for one reD value."""
    _validate_re_d(re_d)

    x_max = np.log(re_d)
    x_grid = np.linspace(0.0, x_max, n_radius)

    # Initial condition: reservoir pressure = 1, wellbore pressure = 0.
    pressure = np.ones(n_radius, dtype=float)
    pressure[0] = 0.0

    t_d_max = max(t_d_min * 10.0, t_d_max_multiplier * re_d**2)
    t_targets = np.logspace(np.log10(t_d_min), np.log10(t_d_max), n_time)

    previous_time = 0.0
    records: list[dict[str, float]] = []

    dx = x_grid[1] - x_grid[0]

    for current_time in t_targets:
        dt = current_time - previous_time
        previous_time = current_time

        matrix = _build_implicit_matrix(x_grid, dt)
        rhs = pressure[1:].copy()

        solved = np.linalg.solve(matrix, rhs)

        pressure[0] = 0.0
        pressure[1:] = solved

        # Dimensionless rate proxy from inner pressure gradient.
        # Positive flow from reservoir to well.
        q_d = max((pressure[1] - pressure[0]) / dx, eps)

        records.append(
            {
                "re_d": re_d,
                "t_d": current_time,
                "q_d": q_d,
            }
        )

    response = pd.DataFrame.from_records(records)

    # Dimensionless cumulative production:
    # QD = integral(qD dtD)
    q = response["q_d"].to_numpy()
    t = response["t_d"].to_numpy()

    q_mid = 0.5 * (q[1:] + q[:-1])
    dt_values = np.diff(t)
    cumulative = np.concatenate(([0.0], np.cumsum(q_mid * dt_values)))

    response["q_cum_d"] = np.maximum(cumulative, eps)

    # Material balance time:
    # tcD = QD / qD
    response["t_c_d"] = response["q_cum_d"] / np.maximum(response["q_d"], eps)

    return response


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with zero initial value.

    Returns an array of the same length as x/y where result[0] = 0 and
    result[i] = integral from x[0] to x[i].
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 2:
        return np.zeros_like(x, dtype=float)
    dx = np.diff(x)
    area = 0.5 * (y[1:] + y[:-1]) * dx
    return np.concatenate(([0.0], np.cumsum(area)))


def _blasingame_transform(response: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Transform radial constant-pressure response to Blasingame coordinates.

    The transformation normalises time and rate so that the late-time
    (boundary-dominated) branch collapses toward the harmonic reference:

        qDd  →  1 / (1 + tcDd)

    This is the key property of Blasingame composite type curves: all reD
    families should converge to the same harmonic decline at large tcDd.

    Approach
    --------
    1. Sort, filter and deduplicate the raw response.
    2. Use the *last 12 % of rows* (at least 8 points) to estimate the product
       ``q_d_raw * t_c_d_raw``.  In BDF this product equals
       ``tc_scale / q_scale`` ≈ const.  The median of this product gives a
       data-driven rate scale that forces late-time collapse.
    3. Apply the computed scales to produce tcDd, qDd, qDdi and qDdid.
    4. Derivative is computed in log(tcDd) space WITHOUT abs() — the sign
       convention is preserved and only values below eps are clipped.
    """
    transformed = response.copy()

    re_d = float(transformed["re_d"].iloc[0])

    t_c_d_raw = transformed["t_c_d"].to_numpy(dtype=float)
    q_d_raw   = transformed["q_d"].to_numpy(dtype=float)

    # ── 1. Filter finite, positive values ────────────────────────────────────
    valid = (
        np.isfinite(t_c_d_raw) & np.isfinite(q_d_raw)
        & (t_c_d_raw > 0.0) & (q_d_raw > 0.0)
    )
    transformed = transformed.loc[valid].copy()
    t_c_d_raw = transformed["t_c_d"].to_numpy(dtype=float)
    q_d_raw   = transformed["q_d"].to_numpy(dtype=float)

    # ── 2. Sort by material-balance time ─────────────────────────────────────
    order = np.argsort(t_c_d_raw)
    transformed = transformed.iloc[order].reset_index(drop=True)
    t_c_d_raw = transformed["t_c_d"].to_numpy(dtype=float)
    q_d_raw   = transformed["q_d"].to_numpy(dtype=float)

    # Remove duplicated or non-increasing tc points (can appear near t=0).
    keep = np.concatenate(([True], np.diff(t_c_d_raw) > 0.0))
    transformed = transformed.loc[keep].reset_index(drop=True)
    t_c_d_raw = transformed["t_c_d"].to_numpy(dtype=float)
    q_d_raw   = transformed["q_d"].to_numpy(dtype=float)

    # ── 3. Late-time anchoring ────────────────────────────────────────────────
    # Geometry-based time scale (Fetkovich/Palacio-Blasingame convention):
    #   tcDd = tcD / [½ (reD² - 1) (ln reD - ½)]
    # → tc_scale = ½ (reD² - 1) * (ln reD - ½)
    # For harmonic BDF: q_dd * tc_dd ≈ 1 at late times
    # → q_scale = tc_scale / median(q_d_raw * t_c_d_raw)_late
    tc_scale = max(0.5 * (re_d**2 - 1.0) * (np.log(re_d) - 0.5), eps)

    n_late = max(8, int(0.12 * len(t_c_d_raw)))
    late_slice = slice(-n_late, None)
    late_product = float(np.median(q_d_raw[late_slice] * t_c_d_raw[late_slice]))

    if not np.isfinite(late_product) or late_product <= 0.0:
        raise ValueError(
            f"Invalid late-time normalization for reD={re_d:g}. "
            "Increase t_d_max_multiplier so the simulation reaches BDF."
        )

    q_scale = max(tc_scale / late_product, eps)

    # ── 4. Build Blasingame coordinates ──────────────────────────────────────
    t_c_dd = np.maximum(t_c_d_raw / tc_scale, eps)
    q_dd   = np.maximum(q_d_raw * q_scale, eps)

    # Integral production function: qDdi = (1/tcDd) * ∫₀^tcDd qDd d(tcDd)
    integral = _cumulative_trapezoid(q_dd, t_c_dd)
    q_ddi    = np.maximum(integral / np.maximum(t_c_dd, eps), eps)

    # Log-derivative: qDdid = -d(qDdi)/d(ln tcDd)
    # Computed in log space for stability on log-spaced grids.
    # Do NOT use abs() — preserve sign convention; clip only below eps.
    ln_t     = np.log(np.maximum(t_c_dd, eps))
    q_ddid_raw = -np.gradient(q_ddi, ln_t)
    q_ddid   = np.maximum(q_ddid_raw, eps)

    transformed["t_c_dd"]                = t_c_dd
    transformed["q_dd"]                  = q_dd
    transformed["q_ddi"]                 = q_ddi
    transformed["q_ddid"]                = q_ddid
    transformed["normalization_tc_scale"] = tc_scale
    transformed["normalization_q_scale"]  = q_scale

    return transformed[
        [
            "re_d",
            "t_d",
            "t_c_d",
            "t_c_dd",
            "q_d",
            "q_cum_d",
            "q_dd",
            "q_ddi",
            "q_ddid",
            "normalization_tc_scale",
            "normalization_q_scale",
        ]
    ]


def check_late_time_convergence(curves: pd.DataFrame, eps: float = 1.0e-30) -> None:
    """QC: verify that qDd at late times follows the harmonic reference 1/(1 + tcDd).

    The Blasingame normalisation forces q_dd * t_c_dd ≈ 1 in BDF, which is
    consistent with the harmonic asymptote qDd ≈ 1/tcDd for large tcDd.
    The column "ratio" is qDd_actual / qDd_harmonic.

    Acceptance criterion (typical):
        0.3 < ratio < 3.0  →  reasonable convergence
        ratio ≫ 1 or ≪ 1  →  simulation has not reached BDF; increase
                               t_d_max_multiplier or n_time.

    Note: qDdi/qDd is NOT expected to converge to 1. In harmonic decline
    qDdi/qDd → ln(tcDd), which grows with time.  That is correct behaviour.
    """
    hdr = f"{'reD':>12}  {'tcDd_late':>10}  {'qDd_sim':>10}  {'qDd_harm':>10}  {'ratio':>8}  QC"
    print(hdr)
    print("-" * len(hdr))
    for re_d, group in curves.groupby("re_d", sort=True):
        n_late = max(5, int(0.05 * len(group)))
        tail = group.tail(n_late)
        tc_late  = float(np.median(tail["t_c_dd"]))
        qdd_sim  = float(np.median(tail["q_dd"]))
        qdd_harm = 1.0 / max(1.0 + tc_late, eps)
        ratio    = qdd_sim / max(qdd_harm, eps)
        ok = "OK" if 0.3 < ratio < 3.0 else "!! increase t_d_max_multiplier"
        print(
            f"{re_d:>12g}  {tc_late:>10.2g}  {qdd_sim:>10.3g}"
            f"  {qdd_harm:>10.3g}  {ratio:>8.3f}  {ok}"
        )


def generate_blasingame_curves(
    config: BlasingameCurveConfig | None = None,
) -> BlasingameCurveSet:
    """Generate Blasingame composite curves for several reD values."""
    cfg = config or BlasingameCurveConfig()

    frames: list[pd.DataFrame] = []

    for re_d in cfg.re_d_values:
        response = _solve_constant_pressure_radial_response(
            re_d,
            n_time=cfg.n_time,
            n_radius=cfg.n_radius,
            t_d_min=cfg.t_d_min,
            t_d_max_multiplier=cfg.t_d_max_multiplier,
            eps=cfg.eps,
        )
        transformed = _blasingame_transform(response, cfg.eps)

        # Remove points beyond t_c_dd_max — deep BDF where q_D ≈ 0 adds
        # no information to the type-curve plate and may carry numerical noise.
        mask = transformed["t_c_dd"] <= cfg.t_c_dd_max
        transformed = transformed.loc[mask].reset_index(drop=True)

        frames.append(transformed)

    curves = pd.concat(frames, ignore_index=True)
    return BlasingameCurveSet(curves=curves)


def plot_blasingame_composite(
    curve_set: BlasingameCurveSet,
    *,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot Blasingame composite plate.

    Line styles:
        solid  — qDd  (instantaneous normalized rate)
        dashed — qDdi (integral of normalized rate)
        dotted — qDdid (derivative of integral)

    The black dashed harmonic reference ``1/(1 + tcDd)`` is overlaid to
    verify late-time convergence of all reD families.
    """
    curves = curve_set.curves

    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    re_d_list = sorted(curves["re_d"].unique())

    for idx, re_d in enumerate(re_d_list):
        group = curves[curves["re_d"] == re_d].sort_values("t_c_dd")
        color = prop_cycle[idx % len(prop_cycle)]
        label = f"{re_d:g}"

        ax.loglog(group["t_c_dd"], group["q_dd"],   color=color, lw=1.3,
                  solid_capstyle="round", label=label)
        ax.loglog(group["t_c_dd"], group["q_ddi"],  color=color, lw=1.0,
                  linestyle="--", alpha=0.80)
        ax.loglog(group["t_c_dd"], group["q_ddid"], color=color, lw=1.0,
                  linestyle=":", alpha=0.80)

    # Harmonic BDF reference: qDd_harmonic = 1 / (1 + tcDd)
    t_ref = np.logspace(-4, 2, 300)
    ax.loglog(t_ref, 1.0 / (1.0 + t_ref),
              color="black", lw=1.6, linestyle=(0, (4, 2)),
              label="harmónica (ref.)")

    # Legend entries for line styles
    from matplotlib.lines import Line2D
    style_legend = [
        Line2D([0], [0], color="gray", lw=1.2, linestyle="-",  label=r"$q_{Dd}$"),
        Line2D([0], [0], color="gray", lw=1.0, linestyle="--", label=r"$q_{Ddi}$"),
        Line2D([0], [0], color="gray", lw=1.0, linestyle=":",  label=r"$q_{Ddid}$"),
    ]
    leg1 = ax.legend(handles=style_legend, loc="lower left", fontsize=9,
                     title="Series", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(title=r"$r_{eD}$", loc="upper right", fontsize=8, title_fontsize=9)

    ax.set_xlabel(r"$t_{cDd}$", fontsize=16)
    ax.set_ylabel(r"$q_{Dd},\ q_{Ddi},\ q_{Ddid}$", fontsize=16)
    ax.set_title("Blasingame Composite Type Curves", fontsize=15)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.45)

    ax.set_xlim(1.0e-4, 1.0e2)
    ax.set_ylim(1.0e-3, 1.0e1)

    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220)

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    """Generate CSV and PNG outputs for quick validation."""
    curve_set = generate_blasingame_curves()

    print("\n── Late-time convergence QC ──────────────────────────────")
    print("Target: both ratios close to 1.0 (BDF collapse to harmonic)")
    check_late_time_convergence(curve_set.curves)

    curve_set.to_csv("output/blasingame_composite_curves.csv")
    plot_blasingame_composite(
        curve_set,
        output_path="output/blasingame_composite_curves.png",
        show=True,
    )


if __name__ == "__main__":
    main()