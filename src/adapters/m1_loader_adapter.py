"""Adaptador conservador para Módulo 1: historia de pozo y Pwf.

No reemplaza src/well_mod/run_estimator.py. Este adaptador normaliza CSVs y
crea las columnas mínimas requeridas por el contrato común M1-M2.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CANONICAL_COLUMNS = [
    "well_id",
    "date",
    "qo_stb_d",
    "qg_mscf_d",
    "qw_stb_d",
    "whp_psia",
    "t_wh_f",
    "pwf_measured_psia",
    "pwf_estimated_psia",
]


COLUMN_ALIASES = {
    # Identificación
    "well": "well_id",
    "well_name": "well_id",
    "pozo": "well_id",
    "nombre_pozo": "well_id",
    # Fecha
    "fecha": "date",
    "timestamp": "date",
    "datetime": "date",
    # Caudal de aceite
    "qo": "qo_stb_d",
    "q_o": "qo_stb_d",
    "q_oil": "qo_stb_d",
    "oil_rate": "qo_stb_d",
    "oil_stb_d": "qo_stb_d",
    "qo_bopd": "qo_stb_d",
    "bopd": "qo_stb_d",
    # Caudal de gas
    "qg": "qg_mscf_d",
    "q_g": "qg_mscf_d",
    "q_gas": "qg_mscf_d",
    "gas_rate": "qg_mscf_d",
    "gas_mscf_d": "qg_mscf_d",
    "qg_mscfd": "qg_mscf_d",
    "mscfd": "qg_mscf_d",
    # Caudal de agua
    "qw": "qw_stb_d",
    "q_w": "qw_stb_d",
    "q_water": "qw_stb_d",
    "water_rate": "qw_stb_d",
    "water_stb_d": "qw_stb_d",
    "qw_bwpd": "qw_stb_d",
    "bwpd": "qw_stb_d",
    # Presión de cabeza
    "whp": "whp_psia",
    "thp": "whp_psia",
    "wellhead_pressure": "whp_psia",
    "whp_psi": "whp_psia",
    "whp_psig": "whp_psia",
    "presion_cabeza": "whp_psia",
    "presion_cabeza_psia": "whp_psia",
    # Temperatura en cabeza
    "temp_wh": "t_wh_f",
    "temperature_wh": "t_wh_f",
    "twh": "t_wh_f",
    "t_wh": "t_wh_f",
    "t_wh_deg_f": "t_wh_f",
    "temperatura_cabeza": "t_wh_f",
    # Pwf medida
    "pwf": "pwf_measured_psia",
    "pwf_psia": "pwf_measured_psia",
    "pwf_psi": "pwf_measured_psia",
    "pwf_measured": "pwf_measured_psia",
    "pwf_measured_psi": "pwf_measured_psia",
    "pwf_measured_psia": "pwf_measured_psia",
    "bottomhole_pressure": "pwf_measured_psia",
    "bottomhole_pressure_psia": "pwf_measured_psia",
    "presion_fondo": "pwf_measured_psia",
    "presion_fondo_psia": "pwf_measured_psia",
    # Pwf estimada
    "pwf_estimated": "pwf_estimated_psia",
    "pwf_estimated_psi": "pwf_estimated_psia",
    "pwf_estimated_psia": "pwf_estimated_psia",
    "pwf_calculated": "pwf_estimated_psia",
    "pwf_calculated_psia": "pwf_estimated_psia",
    "pwf_estimada": "pwf_estimated_psia",
    "pwf_estimada_psia": "pwf_estimated_psia",
}


def load_history_csv(
    path: str | Path,
    *,
    well_id: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> pd.DataFrame:
    """Carga y normaliza historia de producción/presión desde CSV.

    Args:
        path: Ruta del CSV de historia.
        well_id: Filtro opcional por pozo.
        from_date: Fecha inicial opcional en formato YYYY-MM-DD.
        to_date: Fecha final opcional en formato YYYY-MM-DD.

    Returns:
        DataFrame con columnas canónicas mínimas.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        msg = f"No existe el archivo de historia: {csv_path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(csv_path)
    if df.empty:
        msg = f"El archivo de historia está vacío: {csv_path}"
        raise ValueError(msg)

    df = _normalize_columns(df)

    if "well_id" not in df.columns:
        if well_id is None:
            msg = "La historia no tiene columna well_id y no se pasó --well-id."
            raise ValueError(msg)
        df["well_id"] = well_id

    if "date" not in df.columns:
        msg = "La historia debe contener una columna date/fecha/timestamp."
        raise ValueError(msg)

    df["well_id"] = df["well_id"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    invalid_dates = int(df["date"].isna().sum())
    if invalid_dates:
        msg = f"La historia contiene {invalid_dates} fechas inválidas."
        raise ValueError(msg)

    if well_id is not None:
        df = df[df["well_id"] == well_id].copy()

    if from_date is not None:
        df = df[df["date"] >= pd.Timestamp(from_date)].copy()

    if to_date is not None:
        df = df[df["date"] <= pd.Timestamp(to_date)].copy()

    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    numeric_columns = [
        "qo_stb_d",
        "qg_mscf_d",
        "qw_stb_d",
        "whp_psia",
        "t_wh_f",
        "pwf_measured_psia",
        "pwf_estimated_psia",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values(["well_id", "date"]).reset_index(drop=True)
    return df[CANONICAL_COLUMNS]


def apply_pwf_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica regla base: Pwf medida válida tiene prioridad sobre estimada."""

    required = {"pwf_measured_psia", "pwf_estimated_psia"}
    missing = required.difference(df.columns)
    if missing:
        msg = f"Faltan columnas para aplicar regla Pwf: {sorted(missing)}"
        raise ValueError(msg)

    out = df.copy()

    measured_valid = out["pwf_measured_psia"].notna() & (out["pwf_measured_psia"] > 0)
    estimated_valid = out["pwf_estimated_psia"].notna() & (out["pwf_estimated_psia"] > 0)

    out["pwf_used_psia"] = pd.NA
    out["pwf_source"] = "missing"

    out.loc[measured_valid, "pwf_used_psia"] = out.loc[measured_valid, "pwf_measured_psia"]
    out.loc[measured_valid, "pwf_source"] = "measured"

    use_estimated = ~measured_valid & estimated_valid
    out.loc[use_estimated, "pwf_used_psia"] = out.loc[use_estimated, "pwf_estimated_psia"]
    out.loc[use_estimated, "pwf_source"] = "estimated"

    out["pwf_used_psia"] = pd.to_numeric(out["pwf_used_psia"], errors="coerce")
    return out


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas y aplica alias conocidos."""

    renamed: dict[str, str] = {}

    for column in df.columns:
        clean = str(column).strip().lower()
        clean = clean.replace(" ", "_").replace("-", "_")
        clean = clean.replace("(", "").replace(")", "")
        clean = clean.replace("[", "").replace("]", "")
        clean = clean.replace("/", "_")

        renamed[column] = COLUMN_ALIASES.get(clean, clean)

    out = df.rename(columns=renamed)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out