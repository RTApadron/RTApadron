"""Servicio de validación M5 — tabla comparativa ecoRTA vs software comercial.

Construye una lista de ComparisonRow comparando parámetros clave
de ecoRTA (M3 DCA + M4 RTA) contra valores ingresados manualmente
desde cualquier software comercial de referencia.

Semáforo de concordancia (|Δ%|):
    match   — < 5 %
    close   — 5 % a < 20 %
    diverge — ≥ 20 %
    missing — alguno de los dos valores es None
"""

from __future__ import annotations

import json
from pathlib import Path

from src.domain.m5_models import (
    ComparisonRow,
    ComparisonStatus,
    ExternalSoftwareResult,
    WellResultsSummary,
)

# Umbrales de concordancia (porcentaje de diferencia relativa absoluta)
_THRESHOLD_MATCH = 5.0    # |Δ%| < 5  → match
_THRESHOLD_CLOSE = 20.0   # |Δ%| < 20 → close; ≥ 20 → diverge


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _status(rel_diff_pct: float | None) -> ComparisonStatus:
    if rel_diff_pct is None:
        return "missing"
    abs_pct = abs(rel_diff_pct)
    if abs_pct < _THRESHOLD_MATCH:
        return "match"
    if abs_pct < _THRESHOLD_CLOSE:
        return "close"
    return "diverge"


def _row(
    parameter: str,
    units: str,
    ecorta: float | None,
    external: float | None,
    note: str = "",
) -> ComparisonRow:
    """Construye una ComparisonRow calculando diferencias."""
    if ecorta is None or external is None or external == 0.0:
        return ComparisonRow(
            parameter=parameter,
            units=units,
            ecorta_value=ecorta,
            external_value=external,
            abs_diff=None,
            rel_diff_pct=None,
            status="missing",
            note=note,
        )

    abs_diff = abs(ecorta - external)
    rel_diff_pct = (ecorta - external) / external * 100.0

    return ComparisonRow(
        parameter=parameter,
        units=units,
        ecorta_value=ecorta,
        external_value=external,
        abs_diff=abs_diff,
        rel_diff_pct=rel_diff_pct,
        status=_status(rel_diff_pct),
        note=note,
    )


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def build_comparison_table(
    summary: WellResultsSummary,
    external: ExternalSoftwareResult,
    dca_model: str = "best",
) -> list[ComparisonRow]:
    """Genera la tabla comparativa ecoRTA vs software comercial.

    Args:
        summary:    WellResultsSummary con resultados M3 + M4.
        external:   Valores de referencia del software comercial.
        dca_model:  Qué modelo DCA usar para EUR. "best" selecciona el de
                    mayor R²; también acepta "exponential", "hyperbolic",
                    "harmonic".

    Returns:
        Lista de ComparisonRow lista para renderizar y exportar.
    """
    rows: list[ComparisonRow] = []

    # ── DCA: EUR ─────────────────────────────────────────────────────────────
    ecorta_eur: float | None = None
    eur_label = "EUR DCA"

    if summary.dca and summary.dca.models:
        if dca_model == "best":
            best = max(summary.dca.models, key=lambda m: m.r2)
            ecorta_eur = best.eur_stb
            eur_label = f"EUR DCA ({best.model.capitalize()}, mejor R²)"
        else:
            match = next((m for m in summary.dca.models if m.model == dca_model), None)
            if match:
                ecorta_eur = match.eur_stb
                eur_label = f"EUR DCA ({dca_model.capitalize()})"

    rows.append(_row(
        eur_label, "STB",
        ecorta_eur,
        external.eur_stb,
    ))

    # EUR en MM STB (para lectura más cómoda)
    rows.append(_row(
        eur_label.replace("EUR", "EUR").replace("STB", ""),
        "MM STB",
        ecorta_eur / 1e6 if ecorta_eur is not None else None,
        external.eur_stb / 1e6 if external.eur_stb is not None else None,
    ))

    # ── DCA: qi, Di, b (si el usuario los ingresó) ───────────────────────────
    if summary.dca and summary.dca.models and (
        external.qi_stb_d is not None
        or external.di_nominal_d is not None
        or external.b_factor is not None
    ):
        src_model = (
            max(summary.dca.models, key=lambda m: m.r2)
            if dca_model == "best"
            else next((m for m in summary.dca.models if m.model == dca_model), None)
        )
        if src_model:
            rows.append(_row(
                "qi inicial", "STB/d",
                src_model.qi_stb_d,
                external.qi_stb_d,
            ))
            rows.append(_row(
                "Di nominal", "1/d",
                src_model.di_nominal_d,
                external.di_nominal_d,
            ))
            rows.append(_row(
                "b (Arps)", "adim.",
                src_model.b,
                external.b_factor,
            ))

    # ── RTA: kh, k, OOIP ─────────────────────────────────────────────────────
    rta_note = "DEMO — curvas tipo no validadas" if (summary.rta and summary.rta.status == "demo") else ""

    if summary.rta or external.kh_md_ft is not None:
        rows.append(_row(
            "kh", "mD·ft",
            summary.rta.kh_md_ft if summary.rta else None,
            external.kh_md_ft,
            note=rta_note,
        ))

    if summary.rta or external.k_md is not None:
        rows.append(_row(
            "k (permeabilidad efectiva)", "mD",
            summary.rta.k_md if summary.rta else None,
            external.k_md,
            note=rta_note,
        ))

    if summary.rta or external.n_vol_stb is not None:
        rows.append(_row(
            "OOIP volumétrico", "MM STB",
            summary.rta.n_vol_stb / 1e6 if (summary.rta and summary.rta.n_vol_stb) else None,
            external.n_vol_stb / 1e6 if external.n_vol_stb is not None else None,
            note=rta_note,
        ))

    # ── Factor de recobro derivado ────────────────────────────────────────────
    rf_ecorta: float | None = None
    rf_ext: float | None = None

    if ecorta_eur and summary.rta and summary.rta.n_vol_stb:
        rf_ecorta = ecorta_eur / summary.rta.n_vol_stb * 100.0
    if external.eur_stb and external.n_vol_stb:
        rf_ext = external.eur_stb / external.n_vol_stb * 100.0

    if rf_ecorta is not None or rf_ext is not None:
        rows.append(_row(
            "Factor de recobro (EUR/OOIP)", "%",
            rf_ecorta,
            rf_ext,
            note="Derivado: EUR DCA / OOIP volumétrico",
        ))

    return rows


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def save_external_result(
    result: ExternalSoftwareResult,
    output_dir: Path | str,
    well_id: str,
) -> Path:
    """Guarda los valores del software de referencia como JSON en output/."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{well_id}_external_reference.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_external_result(
    output_dir: Path | str,
    well_id: str,
) -> ExternalSoftwareResult | None:
    """Carga los valores del software de referencia desde output/ si existen."""
    path = Path(output_dir) / f"{well_id}_external_reference.json"
    if not path.exists():
        return None
    try:
        return ExternalSoftwareResult.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Exportar tabla a CSV bytes
# ---------------------------------------------------------------------------

def comparison_table_to_csv_bytes(
    rows: list[ComparisonRow],
    summary: WellResultsSummary,
    external: ExternalSoftwareResult,
) -> bytes:
    """Genera un CSV de la tabla comparativa listo para descarga."""
    import io
    import csv

    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow([
        f"ecoRTA vs {external.software_label} — Pozo {summary.well_id}",
    ])
    writer.writerow([])
    writer.writerow([
        "Parámetro", "Unidades",
        "ecoRTA", external.software_label,
        "Δ absoluto", "Δ relativo (%)", "Concordancia", "Nota",
    ])

    for r in rows:
        writer.writerow([
            r.parameter,
            r.units,
            f"{r.ecorta_value:.4g}" if r.ecorta_value is not None else "—",
            f"{r.external_value:.4g}" if r.external_value is not None else "—",
            f"{r.abs_diff:.4g}" if r.abs_diff is not None else "—",
            f"{r.rel_diff_pct:+.2f}" if r.rel_diff_pct is not None else "—",
            r.status,
            r.note,
        ])

    return buf.getvalue().encode("utf-8")
