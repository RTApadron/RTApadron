"""Load RTA type curves from CSV files or internal demo tables."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from pydantic import ValidationError

from src.rta_type_curves.models import RTATypeCurveMethod, TypeCurve, TypeCurvePoint
from src.rta_type_curves.sample_data import DEMO_TYPE_CURVE_ROWS

REQUIRED_COLUMNS: tuple[str, ...] = (
    "method",
    "curve_id",
    "curve_family",
    "x",
    "y",
    "x_label",
    "y_label",
    "source",
    "status",
    "notes",
)


class TypeCurveLoader:
    """Loader for RTA type curves.

    CSV files are preferred. Internal demo rows are only intended to keep
    the M4 pipeline testable before validated/digitized curves are available.
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else Path("data/type_curves")

    def load_from_csv(self, path: Path | str) -> list[TypeCurve]:
        """Load type curves from a single CSV file."""
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Type-curve CSV not found: {csv_path}")

        dataframe = pd.read_csv(csv_path)
        return self._build_curves_from_dataframe(dataframe=dataframe, context=str(csv_path))

    def load_from_directory(self, data_dir: Path | str | None = None) -> list[TypeCurve]:
        """Load all CSV type curves from a directory."""
        directory = Path(data_dir) if data_dir is not None else self.data_dir
        if not directory.exists():
            raise FileNotFoundError(f"Type-curve directory not found: {directory}")

        csv_files = sorted(directory.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No type-curve CSV files found in: {directory}")

        curves: list[TypeCurve] = []
        for csv_file in csv_files:
            curves.extend(self.load_from_csv(csv_file))

        return curves

    def load_demo_curves(self) -> list[TypeCurve]:
        """Load internal demo curves.

        These curves are not validated engineering references.
        """
        dataframe = pd.DataFrame(DEMO_TYPE_CURVE_ROWS)
        return self._build_curves_from_dataframe(dataframe=dataframe, context="internal_demo")

    def load_available(self, allow_demo_fallback: bool = True) -> list[TypeCurve]:
        """Load CSV curves if available, otherwise optionally use demo curves."""
        try:
            return self.load_from_directory(self.data_dir)
        except FileNotFoundError:
            if allow_demo_fallback:
                return self.load_demo_curves()
            raise

    def _build_curves_from_dataframe(self, dataframe: pd.DataFrame, context: str) -> list[TypeCurve]:
        """Validate rows and group them into TypeCurve objects."""
        self._validate_columns(dataframe=dataframe, context=context)

        points = self._build_points(dataframe=dataframe, context=context)
        grouped: dict[tuple[str, str], list[TypeCurvePoint]] = defaultdict(list)

        for point in points:
            grouped[(point.method.value, point.curve_id)].append(point)

        curves: list[TypeCurve] = []
        for (_, _), group_points in grouped.items():
            sorted_points = sorted(group_points, key=lambda point: point.x)
            first = sorted_points[0]
            curves.append(
                TypeCurve(
                    method=first.method,
                    curve_id=first.curve_id,
                    curve_family=first.curve_family,
                    x_label=first.x_label,
                    y_label=first.y_label,
                    source=first.source,
                    status=first.status,
                    notes=first.notes,
                    points=sorted_points,
                )
            )

        return sorted(curves, key=lambda curve: (curve.method.value, curve.curve_id))

    def _validate_columns(self, dataframe: pd.DataFrame, context: str) -> None:
        """Validate required CSV columns."""
        missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in type-curve data '{context}': {missing}. "
                f"Required columns: {list(REQUIRED_COLUMNS)}"
            )

    def _build_points(self, dataframe: pd.DataFrame, context: str) -> list[TypeCurvePoint]:
        """Build validated points from a dataframe."""
        points: list[TypeCurvePoint] = []

        for row_number, row in enumerate(dataframe.to_dict(orient="records"), start=2):
            try:
                points.append(TypeCurvePoint(**row))
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid type-curve row in '{context}' at CSV row {row_number}: {exc}"
                ) from exc

        if not points:
            raise ValueError(f"No type-curve points found in '{context}'.")

        return points


def filter_curves_by_method(
    curves: Iterable[TypeCurve],
    method: RTATypeCurveMethod | str,
) -> list[TypeCurve]:
    """Return curves for one RTA method."""
    method_value = RTATypeCurveMethod(method)
    return [curve for curve in curves if curve.method == method_value]