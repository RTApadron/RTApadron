"""Servicio agregador M5 — construye WellResultsSummary desde archivos output/*.

Lee los artefactos generados por M1-M4 y los unifica en un modelo canónico
sin recalcular nada. Tolerante a módulos ausentes (solo advierte).

Archivos esperados (todos opcionales salvo history_enriched):
    output/<well_id>_history_enriched.csv   → WellInfoSummary + PVTSummary
    output/<well_id>_qc_report.json         → warnings M1
    output/<well_id>_dca_fit_results.csv    → DCASummary
    output/<well_id>_rta_match_summary.json → RTASummary
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.domain.m5_models import (
    DCASummary,
    DCAModelSummary,
    PVTSummary,
    RTASummary,
    WellInfoSummary,
    WellResultsSummary,
)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _parse_date(value: object) -> date | None:
    if value is None:
        return None
    try:
        return pd.to_datetime(str(value)).date()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Constructores por módulo
# ---------------------------------------------------------------------------

def _build_well_info(
    history_df: pd.DataFrame,
    qc_report: dict,
    well_id: str,
) -> WellInfoSummary:
    warnings: list[str] = []

    # warnings del QC M1
    for key in ("warnings", "qc_warnings", "issues"):
        if isinstance(qc_report.get(key), list):
            warnings.extend(str(w) for w in qc_report[key])

    if history_df.empty:
        return WellInfoSummary(
            well_id=well_id,
            qc_warnings=warnings,
        )

    df = history_df.copy()
    date_col = next((c for c in ("date", "fecha", "Date") if c in df.columns), None)

    date_from: date | None = None
    date_to: date | None = None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        valid_dates = df[date_col].dropna()
        if not valid_dates.empty:
            date_from = valid_dates.min().date()
            date_to = valid_dates.max().date()

    qo_col = next((c for c in ("qo_stb_d", "qo", "Qo") if c in df.columns), None)
    qo_avg = float(df[qo_col].dropna().mean()) if qo_col else None
    qo_max = float(df[qo_col].dropna().max()) if qo_col else None

    pwf_source_counts: dict[str, int] = {}
    if "pwf_source" in df.columns:
        pwf_source_counts = df["pwf_source"].value_counts().to_dict()

    # meta del QC report
    api = qc_report.get("api_gravity") or qc_report.get("api")
    temp = qc_report.get("temp_f") or qc_report.get("temperature_f")
    well_name = qc_report.get("well_name")
    field_name = qc_report.get("field_name")

    return WellInfoSummary(
        well_id=well_id,
        well_name=well_name,
        field_name=field_name,
        api_gravity=float(api) if api is not None else None,
        temp_f=float(temp) if temp is not None else None,
        history_points=len(df),
        date_from=date_from,
        date_to=date_to,
        qo_avg_stb_d=qo_avg,
        qo_max_stb_d=qo_max,
        pwf_source_counts=pwf_source_counts,
        qc_warnings=warnings,
    )


def _build_pvt_summary(history_df: pd.DataFrame) -> PVTSummary | None:
    """Extrae estadísticas PVT de la historia enriquecida M2."""
    if history_df.empty:
        return None

    bo_col = next((c for c in ("bo", "bo_rb_stb", "Bo") if c in history_df.columns), None)
    rs_col = next((c for c in ("rs", "rs_scf_stb", "Rs") if c in history_df.columns), None)
    mu_col = next((c for c in ("mu_o_cp", "mu_o", "viscosity") if c in history_df.columns), None)
    pb_col = next((c for c in ("pb_psia", "pb", "Pb") if c in history_df.columns), None)
    corr_col = next((c for c in ("oil_corr",) if c in history_df.columns), None)
    cal_col = next((c for c in ("calibrated_flag",) if c in history_df.columns), None)

    # si no hay columnas PVT la historia no fue enriquecida con M2
    if not any([bo_col, rs_col, mu_col]):
        return None

    avg_bo = float(history_df[bo_col].dropna().mean()) if bo_col else None
    avg_rs = float(history_df[rs_col].dropna().mean()) if rs_col else None
    avg_mu = float(history_df[mu_col].dropna().mean()) if mu_col else None
    pb = float(history_df[pb_col].dropna().median()) if pb_col else None
    corr = str(history_df[corr_col].dropna().iloc[0]) if corr_col else None
    calibrated = bool(history_df[cal_col].dropna().any()) if cal_col else False

    # Trazabilidad: si hay datos de laboratorio (calibrated) → "measured"; si no → "estimated"
    pvt_source = "lab" if calibrated else "correlation"
    pvt_status = "measured" if calibrated else "estimated"

    return PVTSummary(
        oil_corr=corr,
        calibrated=calibrated,
        pvt_source=pvt_source,
        avg_bo_rb_stb=avg_bo,
        avg_rs_scf_stb=avg_rs,
        avg_mu_o_cp=avg_mu,
        pb_psia=pb,
        status=pvt_status,
    )


def _build_dca_summary(dca_df: pd.DataFrame) -> DCASummary | None:
    if dca_df.empty:
        return None

    models: list[DCAModelSummary] = []
    eur_by_model: dict[str, float] = {}

    for _, row in dca_df.iterrows():
        model_name = str(row.get("model", ""))
        eur = float(row["eur_stb"]) if pd.notna(row.get("eur_stb")) else 0.0
        eur_by_model[model_name] = eur

        models.append(DCAModelSummary(
            model=model_name,
            qi_stb_d=float(row.get("qi_stb_d", 0.0) or 0.0),
            di_nominal_d=float(row.get("di_nominal_d", 0.0) or 0.0),
            b=float(row.get("b", 0.0) or 0.0),
            eur_stb=eur,
            r2=float(row.get("r2", 0.0) or 0.0),
            rmse_stb_d=float(row.get("rmse_stb_d", 0.0) or 0.0),
            forecast_days=int(row.get("forecast_days", 0) or 0),
            n_points=int(row.get("n_points", 0) or 0),
        ))

    best_model = max(models, key=lambda m: m.r2).model if models else None

    return DCASummary(
        best_model=best_model,
        models=models,
        eur_exponential_stb=eur_by_model.get("exponential"),
        eur_harmonic_stb=eur_by_model.get("harmonic"),
        eur_hyperbolic_stb=eur_by_model.get("hyperbolic"),
    )


def _build_rta_summary(match_summary: dict) -> RTASummary | None:
    if not match_summary:
        return None

    # New format: results nested under "results" key; multipliers under "match" key.
    # Legacy format: flat dict with "match_params" key. Support both.
    results = (
        match_summary.get("results")
        or match_summary.get("match_params")
        or match_summary
    )
    match_mults = match_summary.get("match", {})

    qc_raw = match_summary.get("warnings", match_summary.get("qc_warnings", []))
    if isinstance(qc_raw, dict):
        qc_warnings = [str(v) for v in qc_raw.values() if v]
    else:
        qc_warnings = [str(w) for w in qc_raw if w]

    def _f(key: str, src: dict = results) -> float | None:
        v = src.get(key)
        return float(v) if v is not None else None

    def _f_mult(key: str) -> float | None:
        # New format: match["x_multiplier"]; Legacy: match_params["effective_x_multiplier"]
        v = match_mults.get(key)
        if v is None:
            v = results.get(f"effective_{key}") or results.get(key)
        return float(v) if v is not None else None

    return RTASummary(
        method=match_summary.get("method"),
        kh_md_ft=_f("kh_md_ft"),
        k_md=_f("k_md"),
        n_vol_stb=_f("n_vol_stb"),
        re_ft=_f("re_ft"),
        area_acres=_f("area_acres"),
        x_multiplier=_f_mult("x_multiplier"),
        y_multiplier=_f_mult("y_multiplier"),
        status=match_summary.get("status", "demo"),
        qc_warnings=qc_warnings,
    )


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def build_well_results(
    *,
    well_id: str,
    output_dir: Path | str,
) -> WellResultsSummary:
    """Construye WellResultsSummary desde los artefactos output/* de M1-M4.

    Args:
        well_id:    Identificador del pozo (prefijo de los archivos).
        output_dir: Directorio donde M1-M4 guardan sus artefactos.

    Returns:
        WellResultsSummary listo para dashboard y exportación.
    """
    out = Path(output_dir)
    warnings: list[str] = []

    # M1-M2: historia enriquecida
    history_path = out / f"{well_id}_history_enriched.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path, low_memory=False)
    else:
        history_df = pd.DataFrame()
        warnings.append(f"M1: {history_path.name} no encontrado — sin historia enriquecida.")

    qc_report = _load_json(out / f"{well_id}_qc_report.json")

    # M3: resultados DCA — prefer the interactive model summary over the pipeline time-series CSV
    dca_summary_path = out / f"{well_id}_dca_model_summary.csv"
    dca_path = out / f"{well_id}_dca_fit_results.csv"
    if dca_summary_path.exists():
        dca_df = pd.read_csv(dca_summary_path)
    elif dca_path.exists():
        _raw = pd.read_csv(dca_path)
        if "model" in _raw.columns and "eur_stb" in _raw.columns:
            dca_df = _raw
        else:
            dca_df = pd.DataFrame()
            warnings.append(
                f"M3: {dca_path.name} contiene datos de serie de tiempo, no resumen por modelo. "
                "Use '💾 Guardar DCA para M5' en M3."
            )
    else:
        dca_df = pd.DataFrame()
        warnings.append(f"M3: {dca_summary_path.name} no encontrado — sin resultados DCA.")

    # M4: match summary RTA
    rta_path = out / f"{well_id}_rta_match_summary.json"
    rta_data = _load_json(rta_path)
    if not rta_data:
        warnings.append(f"M4: {rta_path.name} no encontrado — sin resultados RTA.")

    well_info = _build_well_info(history_df, qc_report, well_id)
    pvt = _build_pvt_summary(history_df)
    dca = _build_dca_summary(dca_df)
    rta = _build_rta_summary(rta_data)

    return WellResultsSummary(
        well_id=well_id,
        well_info=well_info,
        pvt=pvt,
        dca=dca,
        rta=rta,
        consolidated_warnings=warnings,
    )
