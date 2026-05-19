"""Domain models for ecoRTA M4 diagnostic RTA inputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class RTAConfig(BaseModel):
    """User-editable RTA configuration for diagnostic preparation.

    The defaults are placeholders for early M4 workflow testing. They must be
    reviewed and adjusted by the interpreter for each well before technical use.
    """

    well_id: str = Field(..., min_length=1)
    pi_psia: float = Field(3500.0, gt=0.0, description="Initial reservoir pressure.")
    ct_1psi: float = Field(
        1.2e-5,
        gt=0.0,
        description="Total compressibility in 1/psi.",
    )
    phi_frac: float = Field(0.18, gt=0.0, le=1.0, description="Porosity fraction.")
    h_ft: float = Field(50.0, gt=0.0, description="Net pay thickness.")
    rw_ft: float = Field(0.328, gt=0.0, description="Wellbore radius.")
    area_acres: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional drainage area.",
    )
    swi_frac: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="Optional initial water saturation.",
    )
    notes: str | None = Field(
        default="Defaults iniciales M4. Revisar con datos reales del pozo.",
    )
    rta_model_version: str = "m4-rta-config-0.1"

    @field_validator("well_id")
    @classmethod
    def strip_well_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            msg = "well_id no puede estar vacío."
            raise ValueError(msg)
        return cleaned

    def as_repeated_columns(self) -> dict[str, Any]:
        """Return config fields that should be repeated in diagnostic rows."""
        return {
            "pi_psia": self.pi_psia,
            "ct_1psi": self.ct_1psi,
            "phi_frac": self.phi_frac,
            "h_ft": self.h_ft,
            "rw_ft": self.rw_ft,
            "area_acres": self.area_acres,
            "swi_frac": self.swi_frac,
            "rta_model_version": self.rta_model_version,
        }


def default_rta_config(well_id: str) -> RTAConfig:
    """Build default, user-editable RTA configuration for a well."""
    return RTAConfig(well_id=well_id)
