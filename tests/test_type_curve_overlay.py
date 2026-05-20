from pathlib import Path

import pytest

from src.rta_type_curves.loader import TypeCurveLoader
from src.rta_type_curves.models import RTATypeCurveMethod
from src.rta_type_curves.overlay import (
    ManualMatchConfig,
    RTAOverlayPoint,
    build_overlay,
    plot_overlay,
)
from src.rta_type_curves.registry import TypeCurveRegistry


def _get_fetkovich_demo_curve():
    curves = TypeCurveLoader().load_demo_curves()
    registry = TypeCurveRegistry(curves)
    curve_id = registry.list_curve_ids(RTATypeCurveMethod.FETKOVICH)[0]
    return registry.get(RTATypeCurveMethod.FETKOVICH, curve_id)


def test_build_overlay_applies_basic_multipliers() -> None:
    type_curve = _get_fetkovich_demo_curve()

    rta_points = [
        RTAOverlayPoint(x=1.0, y=10.0, label="p1"),
        RTAOverlayPoint(x=2.0, y=20.0, label="p2"),
    ]

    overlay = build_overlay(
        type_curve=type_curve,
        rta_points=rta_points,
        match_config=ManualMatchConfig(
            x_multiplier=10.0,
            y_multiplier=0.1,
        ),
    )

    assert overlay.rta_points_raw.x == [1.0, 2.0]
    assert overlay.rta_points_raw.y == [10.0, 20.0]

    assert overlay.rta_points_matched.x == [10.0, 20.0]
    assert overlay.rta_points_matched.y == [1.0, 2.0]


def test_build_overlay_applies_anchor_matching() -> None:
    type_curve = _get_fetkovich_demo_curve()

    rta_points = [
        RTAOverlayPoint(x=2.0, y=4.0, label="anchor"),
    ]

    overlay = build_overlay(
        type_curve=type_curve,
        rta_points=rta_points,
        match_config=ManualMatchConfig(
            x_multiplier=1.0,
            y_multiplier=1.0,
            anchor_data_x=2.0,
            anchor_data_y=4.0,
            target_curve_x=20.0,
            target_curve_y=8.0,
        ),
    )

    assert overlay.match_config.effective_x_multiplier == 10.0
    assert overlay.match_config.effective_y_multiplier == 2.0
    assert overlay.rta_points_matched.x == [20.0]
    assert overlay.rta_points_matched.y == [8.0]


def test_anchor_matching_requires_complete_anchor_definition() -> None:
    with pytest.raises(ValueError, match="Anchor matching requires all four values"):
        ManualMatchConfig(
            anchor_data_x=1.0,
            anchor_data_y=2.0,
            target_curve_x=3.0,
        )


def test_build_overlay_requires_at_least_one_rta_point() -> None:
    type_curve = _get_fetkovich_demo_curve()

    with pytest.raises(ValueError, match="At least one RTA point"):
        build_overlay(
            type_curve=type_curve,
            rta_points=[],
            match_config=ManualMatchConfig(),
        )


def test_overlay_preserves_type_curve_metadata() -> None:
    type_curve = _get_fetkovich_demo_curve()

    overlay = build_overlay(
        type_curve=type_curve,
        rta_points=[RTAOverlayPoint(x=1.0, y=1.0)],
    )

    assert overlay.curve_id == type_curve.curve_id
    assert overlay.method == type_curve.method.value
    assert overlay.curve_family == type_curve.curve_family
    assert overlay.x_label == type_curve.x_label
    assert overlay.y_label == type_curve.y_label


def test_plot_overlay_writes_png(tmp_path: Path) -> None:
    type_curve = _get_fetkovich_demo_curve()

    overlay = build_overlay(
        type_curve=type_curve,
        rta_points=[
            RTAOverlayPoint(x=0.01, y=1.0),
            RTAOverlayPoint(x=0.1, y=0.8),
            RTAOverlayPoint(x=1.0, y=0.4),
        ],
        match_config=ManualMatchConfig(
            x_multiplier=1.0,
            y_multiplier=1.0,
        ),
    )

    output_path = tmp_path / "fetkovich_overlay.png"

    result_path = plot_overlay(
        overlay=overlay,
        output_path=output_path,
        show_raw_points=True,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0