from pathlib import Path

import pandas as pd
import pytest

from src.rta_type_curves.loader import TypeCurveLoader
from src.rta_type_curves.models import CurveDataStatus, RTATypeCurveMethod
from src.rta_type_curves.registry import TypeCurveRegistry


def test_load_demo_curves_has_three_methods() -> None:
    loader = TypeCurveLoader()
    curves = loader.load_demo_curves()

    methods = {curve.method for curve in curves}

    assert RTATypeCurveMethod.FETKOVICH in methods
    assert RTATypeCurveMethod.PALACIO_BLASINGAME in methods
    assert RTATypeCurveMethod.AGARWAL_GARDNER in methods


def test_load_demo_curves_are_marked_as_demo() -> None:
    loader = TypeCurveLoader()
    curves = loader.load_demo_curves()

    assert curves
    assert all(curve.status == CurveDataStatus.DEMO for curve in curves)
    assert all(point.status == CurveDataStatus.DEMO for curve in curves for point in curve.points)


def test_points_are_sorted_by_x() -> None:
    loader = TypeCurveLoader()
    curves = loader.load_demo_curves()

    for curve in curves:
        x_values = [point.x for point in curve.points]
        assert x_values == sorted(x_values)


def test_load_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "fetkovich_test.csv"
    dataframe = pd.DataFrame(
        [
            {
                "method": "fetkovich",
                "curve_id": "test_curve",
                "curve_family": "test_family",
                "x": 1.0,
                "y": 0.5,
                "x_label": "tDd",
                "y_label": "qDd",
                "source": "test",
                "status": "demo",
                "notes": "test row",
            },
            {
                "method": "fetkovich",
                "curve_id": "test_curve",
                "curve_family": "test_family",
                "x": 0.1,
                "y": 0.9,
                "x_label": "tDd",
                "y_label": "qDd",
                "source": "test",
                "status": "demo",
                "notes": "test row",
            },
        ]
    )
    dataframe.to_csv(csv_path, index=False)

    curves = TypeCurveLoader().load_from_csv(csv_path)

    assert len(curves) == 1
    assert curves[0].method == RTATypeCurveMethod.FETKOVICH
    assert curves[0].curve_id == "test_curve"
    assert [point.x for point in curves[0].points] == [0.1, 1.0]


def test_load_from_csv_requires_expected_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame([{"method": "fetkovich"}]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        TypeCurveLoader().load_from_csv(csv_path)


def test_registry_lists_and_gets_curves() -> None:
    curves = TypeCurveLoader().load_demo_curves()
    registry = TypeCurveRegistry(curves)

    assert RTATypeCurveMethod.FETKOVICH in registry.list_methods()

    fetkovich_ids = registry.list_curve_ids(RTATypeCurveMethod.FETKOVICH)
    assert fetkovich_ids

    curve = registry.get(RTATypeCurveMethod.FETKOVICH, fetkovich_ids[0])
    assert curve.method == RTATypeCurveMethod.FETKOVICH


def test_registry_raises_for_missing_curve() -> None:
    registry = TypeCurveRegistry(TypeCurveLoader().load_demo_curves())

    with pytest.raises(KeyError, match="Type curve not found"):
        registry.get(RTATypeCurveMethod.FETKOVICH, "missing_curve")