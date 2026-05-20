from pathlib import Path

import pandas as pd
import pytest

from src.services.rta_overlay_points_service import (
    build_overlay_points_from_dataframe,
    list_positive_numeric_columns,
    load_history_for_overlay,
)


def test_load_history_for_overlay_reads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    pd.DataFrame(
        [
            {"date": "2024-01-01", "x": 1.0, "y": 10.0},
            {"date": "2024-01-02", "x": 2.0, "y": 20.0},
        ]
    ).to_csv(csv_path, index=False)

    dataframe = load_history_for_overlay(csv_path)

    assert len(dataframe) == 2
    assert {"date", "x", "y"}.issubset(dataframe.columns)


def test_load_history_for_overlay_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="History CSV not found"):
        load_history_for_overlay(tmp_path / "missing.csv")


def test_list_positive_numeric_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "positive": [1.0, 2.0],
            "mixed": [-1.0, 3.0],
            "zero_only": [0.0, 0.0],
            "text": ["a", "b"],
        }
    )

    columns = list_positive_numeric_columns(dataframe)

    assert "positive" in columns
    assert "mixed" in columns
    assert "zero_only" not in columns
    assert "text" not in columns


def test_build_overlay_points_from_dataframe_filters_invalid_log_points() -> None:
    dataframe = pd.DataFrame(
        [
            {"date": "2024-01-01", "x": 1.0, "y": 10.0},
            {"date": "2024-01-02", "x": 0.0, "y": 20.0},
            {"date": "2024-01-03", "x": 3.0, "y": -5.0},
            {"date": "2024-01-04", "x": 4.0, "y": 40.0},
        ]
    )

    points = build_overlay_points_from_dataframe(
        dataframe=dataframe,
        x_column="x",
        y_column="y",
        label_column=None,
        date_column="date",
    )

    assert len(points) == 2
    assert points[0].x == 1.0
    assert points[0].y == 10.0
    assert points[0].date == "2024-01-01"
    assert points[1].x == 4.0
    assert points[1].y == 40.0
    assert points[1].date == "2024-01-04"


def test_build_overlay_points_raises_for_missing_columns() -> None:
    dataframe = pd.DataFrame([{"x": 1.0, "y": 2.0}])

    with pytest.raises(ValueError, match="x_column not found"):
        build_overlay_points_from_dataframe(
            dataframe=dataframe,
            x_column="missing",
            y_column="y",
        )

    with pytest.raises(ValueError, match="y_column not found"):
        build_overlay_points_from_dataframe(
            dataframe=dataframe,
            x_column="x",
            y_column="missing",
        )


def test_build_overlay_points_raises_when_no_positive_points() -> None:
    dataframe = pd.DataFrame(
        [
            {"x": 0.0, "y": 10.0},
            {"x": -1.0, "y": 20.0},
        ]
    )

    with pytest.raises(ValueError, match="No valid positive overlay points"):
        build_overlay_points_from_dataframe(
            dataframe=dataframe,
            x_column="x",
            y_column="y",
        )