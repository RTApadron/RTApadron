"""Adapter for M1 well geometry and survey inputs.

This adapter reads the UI-generated geometry/survey files and enriches the
history dataframe with geometry columns consumed by the Pwf estimator.

It does not estimate Pwf directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class WellGeometryContext:
    """Normalized M1 geometry/survey context for one well."""

    well_id: str
    geometry: dict[str, Any]
    survey: pd.DataFrame | None
    geometry_source: str | None
    survey_source: str | None


def load_well_geometry_context(
    *,
    well_id: str,
    geometry_json: Path | None,
    survey_csv: Path | None,
) -> WellGeometryContext | None:
    """Load geometry/survey context if at least one source exists."""
    geometry: dict[str, Any] = {}
    survey: pd.DataFrame | None = None
    geometry_source: str | None = None
    survey_source: str | None = None

    if geometry_json is not None and geometry_json.exists():
        with geometry_json.open("r", encoding="utf-8") as file:
            raw_geometry = json.load(file)

        if not isinstance(raw_geometry, dict):
            msg = f"Geometry JSON must contain an object: {geometry_json}"
            raise ValueError(msg)

        geometry = raw_geometry
        geometry_source = str(geometry_json)

    if survey_csv is not None and survey_csv.exists():
        survey = pd.read_csv(survey_csv)
        survey = _normalize_survey(survey)
        survey_source = str(survey_csv)

    if not geometry and survey is None:
        return None

    geometry_well_id = str(geometry.get("well_id", well_id)).strip()
    if geometry_well_id and geometry_well_id != well_id:
        msg = (
            "El well_id del archivo de geometría no coincide con --well-id: "
            f"{geometry_well_id!r} != {well_id!r}"
        )
        raise ValueError(msg)

    return WellGeometryContext(
        well_id=well_id,
        geometry=geometry,
        survey=survey,
        geometry_source=geometry_source,
        survey_source=survey_source,
    )


def apply_geometry_context_to_history(
    history_df: pd.DataFrame,
    context: WellGeometryContext | None,
) -> pd.DataFrame:
    """Add geometry/survey columns used by Pwf estimation.

    Output columns added/updated:
    - tvd_perf_ft
    - perforation_mid_tvd_ft
    - tubing_id_in
    - length_ft
    - geometry_source
    - survey_source
    - sensor_depth_md_ft
    - pump_depth_md_ft
    - perforation_top_md_ft
    - perforation_bottom_md_ft
    """
    out = history_df.copy()

    if context is None:
        _ensure_geometry_trace_columns(out)
        return out

    geometry = context.geometry
    survey = context.survey

    perf_top_md = _positive_float_or_none(geometry.get("perforation_top_md_ft"))
    perf_bottom_md = _positive_float_or_none(geometry.get("perforation_bottom_md_ft"))
    perf_mid_md = _midpoint(perf_top_md, perf_bottom_md)

    explicit_perf_mid_tvd = _positive_float_or_none(
        geometry.get("perforation_mid_tvd_ft")
    )
    survey_perf_mid_tvd = _interpolate_tvd_from_survey(survey, perf_mid_md)

    perf_mid_tvd = (
        explicit_perf_mid_tvd
        if explicit_perf_mid_tvd is not None
        else survey_perf_mid_tvd
    )

    tubing_id = _positive_float_or_none(geometry.get("tubing_id_in"))
    sensor_depth_md = _positive_float_or_none(geometry.get("sensor_depth_md_ft"))
    pump_depth_md = _positive_float_or_none(geometry.get("pump_depth_md_ft"))

    flow_length = _first_positive(
        sensor_depth_md,
        pump_depth_md,
        perf_mid_md,
        perf_mid_tvd,
    )

    _ensure_geometry_trace_columns(out)

    if perf_mid_tvd is not None:
        out["tvd_perf_ft"] = perf_mid_tvd
        out["perforation_mid_tvd_ft"] = perf_mid_tvd

    if tubing_id is not None:
        out["tubing_id_in"] = tubing_id

    if flow_length is not None:
        out["length_ft"] = flow_length

    out["geometry_source"] = context.geometry_source or pd.NA
    out["survey_source"] = context.survey_source or pd.NA
    out["sensor_depth_md_ft"] = sensor_depth_md
    out["pump_depth_md_ft"] = pump_depth_md
    out["perforation_top_md_ft"] = perf_top_md
    out["perforation_bottom_md_ft"] = perf_bottom_md

    return out


def _normalize_survey(survey_df: pd.DataFrame) -> pd.DataFrame:
    required = {"md_ft", "tvd_ft"}
    missing = required.difference(survey_df.columns)
    if missing:
        msg = f"Survey CSV missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    out = survey_df.copy()

    for column in ("md_ft", "tvd_ft", "inclination_deg", "azimuth_deg"):
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=["md_ft", "tvd_ft"])
    out = out.sort_values("md_ft").drop_duplicates(subset=["md_ft"], keep="last")

    return out.reset_index(drop=True)


def _ensure_geometry_trace_columns(df: pd.DataFrame) -> None:
    for column in (
        "tvd_perf_ft",
        "perforation_mid_tvd_ft",
        "tubing_id_in",
        "length_ft",
        "geometry_source",
        "survey_source",
        "sensor_depth_md_ft",
        "pump_depth_md_ft",
        "perforation_top_md_ft",
        "perforation_bottom_md_ft",
    ):
        if column not in df.columns:
            df[column] = pd.NA


def _positive_float_or_none(value: object) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None

    parsed_float = float(parsed)
    return parsed_float if parsed_float > 0 else None


def _first_positive(*values: float | None) -> float | None:
    for value in values:
        if value is not None and value > 0:
            return value
    return None


def _midpoint(top: float | None, bottom: float | None) -> float | None:
    if top is None or bottom is None:
        return None
    return (top + bottom) / 2.0


def _interpolate_tvd_from_survey(
    survey: pd.DataFrame | None,
    md_ft: float | None,
) -> float | None:
    if survey is None or md_ft is None or survey.empty:
        return None

    survey = survey.dropna(subset=["md_ft", "tvd_ft"]).sort_values("md_ft")
    if survey.empty:
        return None

    md_values = survey["md_ft"].astype(float)
    tvd_values = survey["tvd_ft"].astype(float)

    if md_ft <= float(md_values.min()):
        return float(tvd_values.iloc[0])

    if md_ft >= float(md_values.max()):
        return float(tvd_values.iloc[-1])

    interpolated = pd.Series(
        tvd_values.to_numpy(),
        index=md_values.to_numpy(),
        dtype="float64",
    ).reindex(
        sorted([*md_values.to_list(), md_ft])
    ).interpolate(method="index")

    value = interpolated.loc[md_ft]
    return None if pd.isna(value) else float(value)