"""Canonical models for RTA type curves.

This module intentionally does not generate analytical type curves.
Curves must come from validated CSV/table sources or clearly marked
internal demo data.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RTATypeCurveMethod(StrEnum):
    """Supported RTA type-curve methods."""

    FETKOVICH = "fetkovich"
    PALACIO_BLASINGAME = "palacio_blasingame"
    AGARWAL_GARDNER = "agarwal_gardner"


class CurveDataStatus(StrEnum):
    """Validation status for curve data."""

    VALIDATED = "validated"
    DIGITIZED_PENDING_QC = "digitized_pending_qc"
    DEMO = "demo"


class TypeCurvePoint(BaseModel):
    """Single point in a type curve."""

    model_config = ConfigDict(extra="forbid")

    method: RTATypeCurveMethod
    curve_id: str = Field(min_length=1)
    curve_family: str = Field(min_length=1)
    x: float = Field(gt=0)
    y: float = Field(gt=0)
    x_label: str = Field(min_length=1)
    y_label: str = Field(min_length=1)
    source: str = Field(min_length=1)
    status: CurveDataStatus = CurveDataStatus.DEMO
    notes: str = ""

    @field_validator("curve_id", "curve_family", "x_label", "y_label", "source", "notes")
    @classmethod
    def strip_text(cls, value: str) -> str:
        """Normalize string fields."""
        return value.strip()


class TypeCurve(BaseModel):
    """A complete type curve grouped by method and curve_id."""

    model_config = ConfigDict(extra="forbid")

    method: RTATypeCurveMethod
    curve_id: str = Field(min_length=1)
    curve_family: str = Field(min_length=1)
    x_label: str = Field(min_length=1)
    y_label: str = Field(min_length=1)
    source: str = Field(min_length=1)
    status: CurveDataStatus = CurveDataStatus.DEMO
    notes: str = ""
    points: list[TypeCurvePoint] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_consistency(self) -> "TypeCurve":
        """Validate all points belong to the same curve definition."""
        for point in self.points:
            if point.method != self.method:
                raise ValueError("All curve points must have the same method.")
            if point.curve_id != self.curve_id:
                raise ValueError("All curve points must have the same curve_id.")
            if point.curve_family != self.curve_family:
                raise ValueError("All curve points must have the same curve_family.")
            if point.x_label != self.x_label:
                raise ValueError("All curve points must have the same x_label.")
            if point.y_label != self.y_label:
                raise ValueError("All curve points must have the same y_label.")
        return self

    def to_records(self) -> list[dict[str, Any]]:
        """Return the curve as plain dictionaries."""
        return [point.model_dump(mode="json") for point in self.points]