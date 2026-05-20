"""Domain models for ecoRTA M4 diagnostic RTA inputs."""

from __future__ import annotations

from typing import Any, Literal

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
    re_ft: float | None = Field(
        default=None,
        gt=0.0,
        description="Drainage radius (ft). Leave blank if area_acres is provided.",
    )
    area_acres: float | None = Field(
        default=None,
        gt=0.0,
        description="Drainage area (acres). Used by Agarwal-Gardner (tDA).",
    )
    Bo_rb_stb: float = Field(
        1.20,
        gt=0.0,
        description="Average oil formation volume factor (RB/STB).",
    )
    mu_o_cp: float = Field(
        2.0,
        gt=0.0,
        description="Average oil viscosity (cp).",
    )
    CA: float = Field(
        31.62,
        gt=0.0,
        description=(
            "Dietz shape factor. Default 31.62 for circular drainage (Table B-1, "
            "Earlougher 1977). Used in Blasingame bpss calculation."
        ),
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
    rta_model_version: str = "m4-rta-config-0.2"

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
            "re_ft": self.re_ft,
            "area_acres": self.area_acres,
            "Bo_rb_stb": self.Bo_rb_stb,
            "mu_o_cp": self.mu_o_cp,
            "CA": self.CA,
            "swi_frac": self.swi_frac,
            "rta_model_version": self.rta_model_version,
        }


def default_rta_config(well_id: str) -> RTAConfig:
    """Build default, user-editable RTA configuration for a well."""
    return RTAConfig(well_id=well_id)


RTAMethod = Literal["fetkovich", "palacio_blasingame", "agarwal_gardner"]
RTAMatchMode = Literal["multiplier", "point_to_target"]

RTA_X_COLUMN_OPTIONS = (
    "elapsed_days",
    "material_balance_time_days",
)

RTA_Y_COLUMN_OPTIONS = (
    "normalized_rate_stb_d_psi",
    "qo_stb_d",
)


class RTAMatchConfig(BaseModel):
    """User-editable manual matching controls for type-curve overlay preparation.

    This model does not estimate reservoir parameters by itself. It only stores
    the interpreter's selected diagnostic axes and scale multipliers so the
    measured well points can be moved consistently on log-log type-curve plots.
    """

    well_id: str = Field(..., min_length=1)
    method: RTAMethod = Field(
        "fetkovich",
        description="Selected RTA type-curve family.",
    )
    x_column: str = Field(
        "material_balance_time_days",
        description="Diagnostic x-axis column from rta_diagnostics.csv.",
    )
    y_column: str = Field(
        "normalized_rate_stb_d_psi",
        description="Diagnostic y-axis column from rta_diagnostics.csv.",
    )
    x_multiplier: float = Field(
        1.0,
        gt=0.0,
        description="Manual x-axis multiplier for log-log overlay.",
    )
    y_multiplier: float = Field(
        1.0,
        gt=0.0,
        description="Manual y-axis multiplier for log-log overlay.",
    )
    match_mode: RTAMatchMode = Field(
        "multiplier",
        description=(
            "Manual match workflow. 'multiplier' uses x/y multipliers directly; "
            "'point_to_target' derives multipliers from an anchor point and target point."
        ),
    )
    anchor_x_raw: float | None = Field(
        default=None,
        gt=0.0,
        description="Selected raw x-coordinate used as matching anchor.",
    )
    anchor_y_raw: float | None = Field(
        default=None,
        gt=0.0,
        description="Selected raw y-coordinate used as matching anchor.",
    )
    target_x: float | None = Field(
        default=None,
        gt=0.0,
        description="Target x-coordinate for the selected anchor point.",
    )
    target_y: float | None = Field(
        default=None,
        gt=0.0,
        description="Target y-coordinate for the selected anchor point.",
    )
    x_label: str = "tD / tiempo equivalente"
    y_label: str = "qD / tasa normalizada"
    match_name: str = "manual_match_001"
    notes: str | None = Field(
        default=(
            "Matching manual preliminar. No calcula parámetros finales de "
            "yacimiento hasta incorporar curvas tipo validadas."
        ),
    )
    match_model_version: str = "m4-rta-manual-match-0.3"

    @field_validator("well_id")
    @classmethod
    def strip_match_well_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            msg = "well_id no puede estar vacío."
            raise ValueError(msg)
        return cleaned

    @field_validator("x_column")
    @classmethod
    def validate_x_column(cls, value: str) -> str:
        if value not in RTA_X_COLUMN_OPTIONS:
            msg = f"x_column no soportada: {value}. Opciones: {RTA_X_COLUMN_OPTIONS}"
            raise ValueError(msg)
        return value

    @field_validator("y_column")
    @classmethod
    def validate_y_column(cls, value: str) -> str:
        if value not in RTA_Y_COLUMN_OPTIONS:
            msg = f"y_column no soportada: {value}. Opciones: {RTA_Y_COLUMN_OPTIONS}"
            raise ValueError(msg)
        return value


    def effective_multipliers(self) -> tuple[float, float]:
        """Return x/y multipliers used to move the diagnostic cloud.

        In multiplier mode, the stored multipliers are used directly. In
        point-to-target mode, the multipliers are derived from the selected
        anchor point and the desired target point.
        """
        if self.match_mode == "multiplier":
            return self.x_multiplier, self.y_multiplier

        required_values = {
            "anchor_x_raw": self.anchor_x_raw,
            "anchor_y_raw": self.anchor_y_raw,
            "target_x": self.target_x,
            "target_y": self.target_y,
        }
        missing = [
            name
            for name, value in required_values.items()
            if value is None or value <= 0
        ]
        if missing:
            msg = (
                "match_mode='point_to_target' requiere valores positivos para: "
                f"{missing}"
            )
            raise ValueError(msg)

        assert self.anchor_x_raw is not None
        assert self.anchor_y_raw is not None
        assert self.target_x is not None
        assert self.target_y is not None
        return self.target_x / self.anchor_x_raw, self.target_y / self.anchor_y_raw


def default_rta_match_config(well_id: str) -> RTAMatchConfig:
    """Build default, user-editable RTA manual match configuration."""
    return RTAMatchConfig(well_id=well_id)
