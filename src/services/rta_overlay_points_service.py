"""Service helpers to build RTA overlay points from tabular history data.

This module does not compute RTA dimensionless variables.
It only maps existing positive numeric columns into overlay points so the M4
visual matching workflow can be tested before validated RTA transforms exist.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.rta_type_curves.overlay import RTAOverlayPoint


def load_history_for_overlay(path: Path | str) -> pd.DataFrame:
    """Load an enriched history CSV for M4 visual overlay."""
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"History CSV not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)

    if dataframe.empty:
        raise ValueError(f"History CSV is empty: {csv_path}")

    return dataframe


def list_positive_numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return columns that can be used as positive log-log axes."""
    numeric_columns: list[str] = []

    for column in dataframe.columns:
        numeric = pd.to_numeric(dataframe[column], errors="coerce")

        if numeric.notna().any() and (numeric.dropna() > 0).any():
            numeric_columns.append(column)

    return numeric_columns


def build_overlay_points_from_dataframe(
    *,
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    label_column: str | None = None,
    date_column: str | None = "date",
) -> list[RTAOverlayPoint]:
    """Build positive RTA overlay points from selected dataframe columns.

    Rows with missing, zero or negative X/Y values are excluded because the
    overlay plot uses log-log axes.
    """
    if x_column not in dataframe.columns:
        raise ValueError(f"x_column not found in dataframe: {x_column}")

    if y_column not in dataframe.columns:
        raise ValueError(f"y_column not found in dataframe: {y_column}")

    if label_column is not None and label_column not in dataframe.columns:
        raise ValueError(f"label_column not found in dataframe: {label_column}")

    if date_column is not None and date_column not in dataframe.columns:
        date_column = None

    working = dataframe.copy()
    working[x_column] = pd.to_numeric(working[x_column], errors="coerce")
    working[y_column] = pd.to_numeric(working[y_column], errors="coerce")

    working = working.dropna(subset=[x_column, y_column])
    working = working[(working[x_column] > 0) & (working[y_column] > 0)]

    if working.empty:
        raise ValueError(
            f"No valid positive overlay points found for x='{x_column}' "
            f"and y='{y_column}'."
        )

    points: list[RTAOverlayPoint] = []

    for index, row in working.iterrows():
        if label_column is not None:
            label = str(row[label_column])
        else:
            label = str(index)

        date_value = str(row[date_column]) if date_column is not None else None

        points.append(
            RTAOverlayPoint(
                x=float(row[x_column]),
                y=float(row[y_column]),
                label=label,
                date=date_value,
            )
        )

    return points