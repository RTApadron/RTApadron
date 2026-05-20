"""Visual overlay helpers for RTA type curves.

This module does not create analytical type curves.
It only overlays already-loaded type curves with user/RTA points and applies
manual visual matching multipliers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.rta_type_curves.models import TypeCurve


class RTAOverlayPoint(BaseModel):
    """Single RTA point to be visually overlaid on a type curve."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(gt=0)
    y: float = Field(gt=0)
    label: str = ""
    date: str | None = None


class ManualMatchConfig(BaseModel):
    """Manual matching configuration for visual overlay.

    x_multiplier and y_multiplier are always applied.

    If anchor_data_x/anchor_data_y and target_curve_x/target_curve_y are provided,
    an additional anchor-based multiplier is computed so the selected data point
    is moved toward the selected target point on the type curve.
    """

    model_config = ConfigDict(extra="forbid")

    x_multiplier: float = Field(default=1.0, gt=0)
    y_multiplier: float = Field(default=1.0, gt=0)

    anchor_data_x: float | None = Field(default=None, gt=0)
    anchor_data_y: float | None = Field(default=None, gt=0)
    target_curve_x: float | None = Field(default=None, gt=0)
    target_curve_y: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_anchor_pair(self) -> "ManualMatchConfig":
        """Validate that anchor and target coordinates are complete."""
        anchor_values = [
            self.anchor_data_x,
            self.anchor_data_y,
            self.target_curve_x,
            self.target_curve_y,
        ]

        provided_count = sum(value is not None for value in anchor_values)

        if provided_count not in {0, 4}:
            raise ValueError(
                "Anchor matching requires all four values: "
                "anchor_data_x, anchor_data_y, target_curve_x, target_curve_y."
            )

        return self

    @property
    def effective_x_multiplier(self) -> float:
        """Return final X multiplier including optional anchor correction."""
        if self.anchor_data_x is None or self.target_curve_x is None:
            return self.x_multiplier

        return self.x_multiplier * (self.target_curve_x / self.anchor_data_x)

    @property
    def effective_y_multiplier(self) -> float:
        """Return final Y multiplier including optional anchor correction."""
        if self.anchor_data_y is None or self.target_curve_y is None:
            return self.y_multiplier

        return self.y_multiplier * (self.target_curve_y / self.anchor_data_y)


class OverlaySeries(BaseModel):
    """Prepared data series for plotting."""

    model_config = ConfigDict(extra="forbid")

    x: list[float]
    y: list[float]
    label: str


class TypeCurveOverlayResult(BaseModel):
    """Prepared overlay between one type curve and one RTA point cloud."""

    model_config = ConfigDict(extra="forbid")

    curve_id: str
    method: str
    curve_family: str
    x_label: str
    y_label: str
    type_curve: OverlaySeries
    rta_points_raw: OverlaySeries
    rta_points_matched: OverlaySeries
    match_config: ManualMatchConfig


def build_overlay(
    *,
    type_curve: TypeCurve,
    rta_points: list[RTAOverlayPoint],
    match_config: ManualMatchConfig | None = None,
) -> TypeCurveOverlayResult:
    """Build transformed overlay data for visual matching."""
    if not rta_points:
        raise ValueError("At least one RTA point is required for overlay.")

    config = match_config or ManualMatchConfig()

    curve_x = [point.x for point in type_curve.points]
    curve_y = [point.y for point in type_curve.points]

    raw_x = [point.x for point in rta_points]
    raw_y = [point.y for point in rta_points]

    matched_x = [value * config.effective_x_multiplier for value in raw_x]
    matched_y = [value * config.effective_y_multiplier for value in raw_y]

    return TypeCurveOverlayResult(
        curve_id=type_curve.curve_id,
        method=type_curve.method.value,
        curve_family=type_curve.curve_family,
        x_label=type_curve.x_label,
        y_label=type_curve.y_label,
        type_curve=OverlaySeries(
            x=curve_x,
            y=curve_y,
            label=f"Type curve: {type_curve.curve_id}",
        ),
        rta_points_raw=OverlaySeries(
            x=raw_x,
            y=raw_y,
            label="RTA points raw",
        ),
        rta_points_matched=OverlaySeries(
            x=matched_x,
            y=matched_y,
            label="RTA points matched",
        ),
        match_config=config,
    )


def plot_overlay(
    *,
    overlay: TypeCurveOverlayResult,
    output_path: Path | str | None = None,
    show_raw_points: bool = True,
) -> Path | None:
    """Plot type curve and matched RTA points in log-log scale.

    If output_path is provided, the figure is saved and the path is returned.
    Otherwise the figure is created and closed without writing a file.
    """
    plt.figure()

    plt.loglog(
        overlay.type_curve.x,
        overlay.type_curve.y,
        marker="",
        label=overlay.type_curve.label,
    )

    if show_raw_points:
        plt.loglog(
            overlay.rta_points_raw.x,
            overlay.rta_points_raw.y,
            linestyle="",
            marker="o",
            label=overlay.rta_points_raw.label,
        )

    plt.loglog(
        overlay.rta_points_matched.x,
        overlay.rta_points_matched.y,
        linestyle="",
        marker="x",
        label=overlay.rta_points_matched.label,
    )

    plt.xlabel(overlay.x_label)
    plt.ylabel(overlay.y_label)
    plt.title(f"{overlay.method} overlay - {overlay.curve_id}")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    if output_path is None:
        plt.close()
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

    return path