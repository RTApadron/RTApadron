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
    """Configuration for dimensionless Blasingame curve generation."""

    re_d_values: tuple[float, ...] = (10, 20, 50, 100, 1_000, 10_000, 100_000, 1_000_000)
    n_time: int = 220
    n_radius: int = 180
    t_d_min: float = 1.0e-6
    t_d_max_multiplier: float = 2.0e3
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


def _blasingame_transform(response: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Build qDd, qDdi and qDdid from dimensionless rate response."""
    transformed = response.copy()

    re_d = float(transformed["re_d"].iloc[0])

    # Practical scaling for modified decline-curve coordinates.
    # This keeps families visually comparable on a common Blasingame plate.
    # It is isolated here so it can later be replaced by the exact Palacio-
    # Blasingame/Fetkovich normalization constants during validation.
    time_scale = max(0.5 * (re_d**2 - 1.0), eps)
    rate_scale = max(np.log(re_d) - 0.5, eps)

    transformed["t_c_dd"] = np.maximum(transformed["t_c_d"] / time_scale, eps)
    transformed["q_dd"] = np.maximum(transformed["q_d"] * rate_scale, eps)

    x = transformed["t_c_dd"].to_numpy()
    y = transformed["q_dd"].to_numpy()

    # Integral function:
    # qDdi = 1/tcDd * integral(qDd d(tcDd))
    integral = np.concatenate(
        (
            [0.0],
            np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x)),
        )
    )
    transformed["q_ddi"] = np.maximum(integral / np.maximum(x, eps), eps)

    # Integral derivative:
    # qDdid = - d(qDdi) / d ln(tcDd)
    ln_x = np.log(np.maximum(x, eps))
    q_ddi = transformed["q_ddi"].to_numpy()

    derivative = -np.gradient(q_ddi, ln_x)
    transformed["q_ddid"] = np.maximum(np.abs(derivative), eps)

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
        ]
    ]


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
        frames.append(_blasingame_transform(response, cfg.eps))

    curves = pd.concat(frames, ignore_index=True)
    return BlasingameCurveSet(curves=curves)


def plot_blasingame_composite(
    curve_set: BlasingameCurveSet,
    *,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot Blasingame composite plate."""
    curves = curve_set.curves

    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    for re_d, group in curves.groupby("re_d", sort=True):
        label = f"{re_d:g}"

        ax.loglog(
            group["t_c_dd"],
            group["q_dd"],
            linewidth=1.2,
            label=label,
        )
        ax.loglog(
            group["t_c_dd"],
            group["q_ddi"],
            linewidth=0.9,
            alpha=0.65,
        )
        ax.loglog(
            group["t_c_dd"],
            group["q_ddid"],
            linewidth=0.9,
            alpha=0.65,
        )

    ax.set_xlabel(r"$t_{cDd}$", fontsize=16)
    ax.set_ylabel(r"$q_{Dd},\ q_{Ddi},\ q_{Ddid}$", fontsize=16)
    ax.set_title("Blasingame Composite Type Curves", fontsize=15)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.45)
    ax.legend(title=r"$r_{eD}$", loc="upper right")

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

    curve_set.to_csv("output/blasingame_composite_curves.csv")
    plot_blasingame_composite(
        curve_set,
        output_path="output/blasingame_composite_curves.png",
        show=True,
    )


if __name__ == "__main__":
    main()