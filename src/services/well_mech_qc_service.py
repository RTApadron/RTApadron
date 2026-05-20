"""QC checks for well mechanical state (M1).

Validates dimensional compatibility of casing strings, tubing, and
perforations before any Pwf or RTA calculation is attempted.

Each check returns one or more MechQCResult with severity ok/warning/error.

Checks implemented
------------------
PERF_ORDER           — perfs_bottom > perfs_top
CASING_WALL          — ID < OD for every casing string
TUBING_WALL          — tubing ID < OD
CASING_SHOE_ORDER    — each inner casing shoe deeper than outer
CASING_CONCENTRICITY — each inner casing fits inside the next outer (OD < ID_outer)
TUBING_FITS_CASING   — tubing OD < innermost casing ID
PERFS_WITHIN_CASING  — perfs bottom depth ≤ casing shoe
TUBING_VS_PERFS      — tubing shoe relative to perforation interval
ESP_CONFIG           — ESP intake within tubing and above perfs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

Severity = Literal["ok", "warning", "error"]


@dataclass(frozen=True)
class MechQCResult:
    """Single QC check result for the well mechanical state."""

    code: str
    severity: Severity
    title: str
    detail: str


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------

@dataclass
class CasingString:
    """A single casing or liner string."""

    name: str           # "Superficie", "Intermedio", "Producción", etc.
    od_in: float        # outer diameter [in]
    id_in: float        # inner diameter / drift [in]
    shoe_depth_ft: float  # MD depth of casing shoe [ft]


@dataclass
class TubingString:
    """Production tubing string."""

    od_in: float        # outer diameter [in]
    id_in: float        # inner diameter [in]
    set_depth_ft: float  # MD depth of tubing shoe [ft]


@dataclass
class WellMechConfig:
    """Well mechanical state configuration."""

    well_id: str
    casings: list[CasingString]   # ordered outermost → innermost (Surface first)
    tubing: TubingString
    perfs_top_ft: float
    perfs_bottom_ft: float
    has_esp: bool = False
    esp_intake_depth_ft: float | None = None
    total_depth_ft: float | None = None

    @property
    def innermost_casing(self) -> CasingString | None:
        return self.casings[-1] if self.casings else None

    @property
    def effective_total_depth(self) -> float:
        candidates = [self.perfs_bottom_ft + 200.0]
        if self.casings:
            candidates.append(self.casings[-1].shoe_depth_ft + 100.0)
        if self.total_depth_ft:
            candidates.append(self.total_depth_ft)
        return max(candidates)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_MIN_CLEARANCE_CASING_IN = 0.250   # min diametral clearance between adjacent casings [in]
_MIN_CLEARANCE_TUBING_IN = 0.125   # min diametral clearance tubing OD vs casing ID [in]
_PERF_NEAR_SHOE_FT       = 100.0   # warn if perfs bottom is within this of casing shoe [ft]
_TUBING_FAR_FROM_PERFS   = 500.0   # warn if tubing set depth is this far above perfs_top [ft]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_perf_order(
    perfs_top_ft: float,
    perfs_bottom_ft: float,
) -> list[MechQCResult]:
    if perfs_bottom_ft <= perfs_top_ft:
        return [MechQCResult(
            code="PERF_ORDER",
            severity="error",
            title="Intervalo de perforaciones inválido",
            detail=(
                f"Tope perforado ({perfs_top_ft:.0f} ft) ≥ "
                f"fondo perforado ({perfs_bottom_ft:.0f} ft). "
                "El tope debe ser más somero que el fondo."
            ),
        )]
    thickness = perfs_bottom_ft - perfs_top_ft
    return [MechQCResult(
        code="PERF_ORDER",
        severity="ok",
        title="Intervalo de perforaciones",
        detail=(
            f"{perfs_top_ft:.0f}–{perfs_bottom_ft:.0f} ft MD "
            f"| espesor = {thickness:.0f} ft"
        ),
    )]


def _check_casing_wall_integrity(casings: list[CasingString]) -> list[MechQCResult]:
    results: list[MechQCResult] = []
    for c in casings:
        if c.id_in <= 0:
            results.append(MechQCResult(
                code="CASING_WALL",
                severity="error",
                title=f"ID inválido — {c.name}",
                detail=f"{c.name}: ID = {c.id_in:.3f}\" ≤ 0.",
            ))
        elif c.id_in >= c.od_in:
            results.append(MechQCResult(
                code="CASING_WALL",
                severity="error",
                title=f"Diámetro inválido — {c.name}",
                detail=(
                    f"{c.name}: ID = {c.id_in:.3f}\" ≥ OD = {c.od_in:.3f}\". "
                    "El diámetro interno debe ser menor que el externo."
                ),
            ))
        else:
            wall_in = (c.od_in - c.id_in) / 2.0
            results.append(MechQCResult(
                code="CASING_WALL",
                severity="ok",
                title=f"Pared válida — {c.name}",
                detail=(
                    f"{c.name}: OD = {c.od_in:.3f}\" | ID = {c.id_in:.3f}\" "
                    f"| espesor pared = {wall_in:.3f}\""
                ),
            ))
    return results


def _check_tubing_wall_integrity(tubing: TubingString) -> list[MechQCResult]:
    if tubing.id_in <= 0:
        return [MechQCResult(
            code="TUBING_WALL",
            severity="error",
            title="Tubing ID inválido",
            detail=f"Tubing ID = {tubing.id_in:.3f}\" ≤ 0.",
        )]
    if tubing.id_in >= tubing.od_in:
        return [MechQCResult(
            code="TUBING_WALL",
            severity="error",
            title="Diámetro tubing inválido",
            detail=(
                f"Tubing ID = {tubing.id_in:.3f}\" ≥ OD = {tubing.od_in:.3f}\". "
                "El ID debe ser menor que el OD."
            ),
        )]
    wall_in = (tubing.od_in - tubing.id_in) / 2.0
    return [MechQCResult(
        code="TUBING_WALL",
        severity="ok",
        title="Tubing — pared válida",
        detail=(
            f"OD = {tubing.od_in:.3f}\" | ID = {tubing.id_in:.3f}\" "
            f"| espesor pared = {wall_in:.3f}\""
        ),
    )]


def _check_casing_shoe_order(casings: list[CasingString]) -> list[MechQCResult]:
    """Inner casing shoe must be deeper than the outer casing shoe."""
    results: list[MechQCResult] = []
    for i in range(1, len(casings)):
        outer = casings[i - 1]
        inner = casings[i]
        if inner.shoe_depth_ft <= outer.shoe_depth_ft:
            results.append(MechQCResult(
                code="CASING_SHOE_ORDER",
                severity="error",
                title=f"Zapatos fuera de orden — {inner.name} vs {outer.name}",
                detail=(
                    f"{inner.name} shoe = {inner.shoe_depth_ft:.0f} ft ≤ "
                    f"{outer.name} shoe = {outer.shoe_depth_ft:.0f} ft. "
                    "El revestimiento interior debe asentarse más profundo que el exterior."
                ),
            ))
        else:
            results.append(MechQCResult(
                code="CASING_SHOE_ORDER",
                severity="ok",
                title=f"Zapatos en orden — {inner.name} / {outer.name}",
                detail=(
                    f"{outer.name} shoe = {outer.shoe_depth_ft:.0f} ft "
                    f"→ {inner.name} shoe = {inner.shoe_depth_ft:.0f} ft ✓"
                ),
            ))
    return results


def _check_casing_concentricity(casings: list[CasingString]) -> list[MechQCResult]:
    """Each inner casing OD must be smaller than the outer casing ID."""
    results: list[MechQCResult] = []
    for i in range(1, len(casings)):
        outer = casings[i - 1]
        inner = casings[i]
        clearance = outer.id_in - inner.od_in  # diametral clearance [in]
        if clearance <= 0:
            results.append(MechQCResult(
                code="CASING_CONCENTRICITY",
                severity="error",
                title=f"{inner.name} no cabe dentro de {outer.name}",
                detail=(
                    f"{inner.name} OD = {inner.od_in:.3f}\" ≥ "
                    f"{outer.name} ID = {outer.id_in:.3f}\". "
                    "Imposible bajar el revestimiento interior."
                ),
            ))
        elif clearance < _MIN_CLEARANCE_CASING_IN:
            results.append(MechQCResult(
                code="CASING_CONCENTRICITY",
                severity="warning",
                title=f"Holgura ajustada — {inner.name} en {outer.name}",
                detail=(
                    f"Holgura diametral = {clearance:.3f}\" "
                    f"(mín. recomendado: {_MIN_CLEARANCE_CASING_IN:.3f}\"). "
                    "Puede dificultar la cementación e instalación."
                ),
            ))
        else:
            results.append(MechQCResult(
                code="CASING_CONCENTRICITY",
                severity="ok",
                title=f"Concentricidad — {inner.name} en {outer.name}",
                detail=(
                    f"{inner.name} OD = {inner.od_in:.3f}\" | "
                    f"{outer.name} ID = {outer.id_in:.3f}\" | "
                    f"holgura = {clearance:.3f}\""
                ),
            ))
    return results


def _check_tubing_fits_casing(
    tubing: TubingString,
    innermost_casing: CasingString | None,
) -> list[MechQCResult]:
    if innermost_casing is None:
        return []
    clearance = innermost_casing.id_in - tubing.od_in
    if clearance <= 0:
        return [MechQCResult(
            code="TUBING_FITS_CASING",
            severity="error",
            title="Tubing no cabe en el casing de producción",
            detail=(
                f"Tubing OD = {tubing.od_in:.3f}\" ≥ "
                f"{innermost_casing.name} ID = {innermost_casing.id_in:.3f}\". "
                "Imposible bajar el tubing. Reducir OD o cambiar el revestimiento."
            ),
        )]
    if clearance < _MIN_CLEARANCE_TUBING_IN:
        return [MechQCResult(
            code="TUBING_FITS_CASING",
            severity="warning",
            title="Holgura tubing–casing ajustada",
            detail=(
                f"Holgura diametral = {clearance:.3f}\" "
                f"(mín. recomendado: {_MIN_CLEARANCE_TUBING_IN:.3f}\"). "
                "Puede dificultar el desplazamiento del tubing y trabajos de cable."
            ),
        )]
    return [MechQCResult(
        code="TUBING_FITS_CASING",
        severity="ok",
        title="Tubing dentro del casing de producción",
        detail=(
            f"Tubing OD = {tubing.od_in:.3f}\" | "
            f"{innermost_casing.name} ID = {innermost_casing.id_in:.3f}\" | "
            f"holgura = {clearance:.3f}\""
        ),
    )]


def _check_perfs_within_casing(
    perfs_top_ft: float,
    perfs_bottom_ft: float,
    innermost_casing: CasingString | None,
) -> list[MechQCResult]:
    if innermost_casing is None:
        return []
    shoe = innermost_casing.shoe_depth_ft
    if perfs_bottom_ft > shoe:
        return [MechQCResult(
            code="PERFS_WITHIN_CASING",
            severity="error",
            title="Perforaciones por debajo del zapato del casing",
            detail=(
                f"Fondo perforado ({perfs_bottom_ft:.0f} ft) > "
                f"{innermost_casing.name} shoe ({shoe:.0f} ft). "
                "Las perforaciones quedan en hoyo abierto — verificar profundidades."
            ),
        )]
    if perfs_bottom_ft > shoe - _PERF_NEAR_SHOE_FT:
        dist = shoe - perfs_bottom_ft
        return [MechQCResult(
            code="PERFS_WITHIN_CASING",
            severity="warning",
            title="Perforaciones cerca del zapato del casing",
            detail=(
                f"Fondo perforado a solo {dist:.0f} ft del zapato del "
                f"{innermost_casing.name} ({shoe:.0f} ft). "
                "Riesgo de comunicación con el hoyo abierto."
            ),
        )]
    return [MechQCResult(
        code="PERFS_WITHIN_CASING",
        severity="ok",
        title="Perforaciones dentro del casing",
        detail=(
            f"{perfs_top_ft:.0f}–{perfs_bottom_ft:.0f} ft dentro del "
            f"{innermost_casing.name} (shoe = {shoe:.0f} ft)."
        ),
    )]


def _check_tubing_vs_perfs(
    tubing: TubingString,
    perfs_top_ft: float,
    perfs_bottom_ft: float,
) -> list[MechQCResult]:
    sd = tubing.set_depth_ft
    if sd > perfs_bottom_ft:
        return [MechQCResult(
            code="TUBING_VS_PERFS",
            severity="warning",
            title="Tubing asentado debajo de todo el intervalo perforado",
            detail=(
                f"Zapato tubing = {sd:.0f} ft > fondo perforado = {perfs_bottom_ft:.0f} ft. "
                "Diseño inusual — verificar completamiento."
            ),
        )]
    if sd > perfs_top_ft:
        return [MechQCResult(
            code="TUBING_VS_PERFS",
            severity="warning",
            title="Tubing dentro del intervalo perforado",
            detail=(
                f"Zapato tubing = {sd:.0f} ft está dentro del intervalo perforado "
                f"({perfs_top_ft:.0f}–{perfs_bottom_ft:.0f} ft). "
                "Verificar diseño de completamiento."
            ),
        )]
    dist = perfs_top_ft - sd
    if dist > _TUBING_FAR_FROM_PERFS:
        return [MechQCResult(
            code="TUBING_VS_PERFS",
            severity="warning",
            title="Tubing lejos de las perforaciones",
            detail=(
                f"Zapato tubing = {sd:.0f} ft — {dist:.0f} ft sobre el tope perforado "
                f"({perfs_top_ft:.0f} ft). Volumen muerto grande puede afectar la Pwf medida."
            ),
        )]
    return [MechQCResult(
        code="TUBING_VS_PERFS",
        severity="ok",
        title="Posición tubing vs perforaciones",
        detail=(
            f"Zapato tubing = {sd:.0f} ft | "
            f"{dist:.0f} ft sobre tope perforado ({perfs_top_ft:.0f} ft)."
        ),
    )]


def _check_esp(
    has_esp: bool,
    esp_intake_depth_ft: float | None,
    tubing_set_depth_ft: float,
    perfs_top_ft: float,
) -> list[MechQCResult]:
    if not has_esp:
        return []
    if esp_intake_depth_ft is None:
        return [MechQCResult(
            code="ESP_CONFIG",
            severity="warning",
            title="ESP habilitado sin profundidad de intake",
            detail="Ingrese la profundidad del intake del ESP.",
        )]
    if esp_intake_depth_ft > tubing_set_depth_ft:
        return [MechQCResult(
            code="ESP_CONFIG",
            severity="error",
            title="Intake del ESP por debajo del zapato del tubing",
            detail=(
                f"Intake ESP = {esp_intake_depth_ft:.0f} ft > "
                f"zapato tubing = {tubing_set_depth_ft:.0f} ft. "
                "El ESP debe estar dentro del tubing."
            ),
        )]
    if esp_intake_depth_ft >= perfs_top_ft:
        return [MechQCResult(
            code="ESP_CONFIG",
            severity="warning",
            title="Intake del ESP en zona perforada o más profundo",
            detail=(
                f"Intake ESP = {esp_intake_depth_ft:.0f} ft ≥ "
                f"tope perforado = {perfs_top_ft:.0f} ft. "
                "Diseño inusual — verificar que el ESP no esté en el intervalo perforado."
            ),
        )]
    dist = perfs_top_ft - esp_intake_depth_ft
    return [MechQCResult(
        code="ESP_CONFIG",
        severity="ok",
        title="Configuración ESP",
        detail=(
            f"Intake ESP = {esp_intake_depth_ft:.0f} ft | "
            f"{dist:.0f} ft sobre tope perforado ({perfs_top_ft:.0f} ft)."
        ),
    )]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_mech_qc(config: WellMechConfig) -> list[MechQCResult]:
    """Run all mechanical state QC checks in order of priority.

    Args:
        config: Well mechanical configuration.

    Returns:
        Ordered list of MechQCResult — one entry per check (never empty when
        data is present for that check).
    """
    results: list[MechQCResult] = []
    results.extend(_check_perf_order(config.perfs_top_ft, config.perfs_bottom_ft))
    results.extend(_check_casing_wall_integrity(config.casings))
    results.extend(_check_tubing_wall_integrity(config.tubing))
    results.extend(_check_casing_shoe_order(config.casings))
    results.extend(_check_casing_concentricity(config.casings))
    results.extend(_check_tubing_fits_casing(config.tubing, config.innermost_casing))
    results.extend(_check_perfs_within_casing(
        config.perfs_top_ft, config.perfs_bottom_ft, config.innermost_casing
    ))
    results.extend(_check_tubing_vs_perfs(
        config.tubing, config.perfs_top_ft, config.perfs_bottom_ft
    ))
    results.extend(_check_esp(
        config.has_esp,
        config.esp_intake_depth_ft,
        config.tubing.set_depth_ft,
        config.perfs_top_ft,
    ))
    return results


def mech_severity_level(results: list[MechQCResult]) -> Severity:
    """Return worst severity across all QC results."""
    if any(r.severity == "error" for r in results):
        return "error"
    if any(r.severity == "warning" for r in results):
        return "warning"
    return "ok"
