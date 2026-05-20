"""Standalone Streamlit UI for M4 type-curve visual overlay.

Run from the project root with:

    python -m streamlit run src/ui/m4_type_curve_overlay.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.rta_type_curves.loader import TypeCurveLoader
from src.rta_type_curves.overlay import ManualMatchConfig, build_overlay
from src.rta_type_curves.registry import TypeCurveRegistry
from src.services.rta_overlay_points_service import (
    build_overlay_points_from_dataframe,
    list_positive_numeric_columns,
    load_history_for_overlay,
)


def _load_registry() -> TypeCurveRegistry:
    """Load type curves from CSV files or demo fallback."""
    loader = TypeCurveLoader()
    return TypeCurveRegistry(loader.load_available(allow_demo_fallback=True))


def _plot_overlay_streamlit(overlay_result: Any) -> None:
    """Render overlay in Streamlit using matplotlib."""
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.loglog(
        overlay_result.type_curve.x,
        overlay_result.type_curve.y,
        label=overlay_result.type_curve.label,
    )

    ax.loglog(
        overlay_result.rta_points_raw.x,
        overlay_result.rta_points_raw.y,
        linestyle="",
        marker="o",
        label=overlay_result.rta_points_raw.label,
    )

    ax.loglog(
        overlay_result.rta_points_matched.x,
        overlay_result.rta_points_matched.y,
        linestyle="",
        marker="x",
        label=overlay_result.rta_points_matched.label,
    )

    ax.set_xlabel(overlay_result.x_label)
    ax.set_ylabel(overlay_result.y_label)
    ax.set_title(f"{overlay_result.method} - {overlay_result.curve_id}")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    """Read an uploaded CSV through a temporary file."""
    with NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    try:
        return load_history_for_overlay(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    """Run the standalone M4 overlay screen."""
    st.set_page_config(page_title="M4 RTA Type-Curve Overlay", layout="wide")

    st.title("M4 - Overlay visual datos RTA vs curvas tipo")

    st.success("Pantalla M4 cargada correctamente.")

    st.info(
        "Esta pantalla permite superponer puntos del pozo sobre curvas tipo "
        "cargadas desde CSV/tablas internas. No calcula todavía kh, skin, "
        "volumen contactado ni OOIP."
    )

    try:
        registry = _load_registry()
    except Exception as exc:
        st.error(f"No fue posible cargar el registro de curvas tipo: {exc}")
        return

    available_methods = registry.list_methods()

    if not available_methods:
        st.error("No hay curvas tipo disponibles.")
        return

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("Curva tipo")

        method = st.selectbox(
            "Método",
            options=available_methods,
            format_func=lambda value: value.value,
        )

        curve_ids = registry.list_curve_ids(method)

        if not curve_ids:
            st.error(f"No hay curvas disponibles para el método {method.value}.")
            return

        curve_id = st.selectbox(
            "Curva",
            options=curve_ids,
        )

        type_curve = registry.get(method, curve_id)

        st.caption(f"Familia: {type_curve.curve_family}")
        st.caption(f"Estado de datos: {type_curve.status.value}")
        st.caption(f"Fuente: {type_curve.source}")

        if type_curve.status.value == "demo":
            st.warning(
                "Esta curva está marcada como demo. Debe reemplazarse por "
                "curvas digitalizadas/validadas antes de interpretación técnica."
            )

        st.subheader("Datos del pozo")

        uploaded_csv = st.file_uploader(
            "Cargar historia enriquecida CSV",
            type=["csv"],
        )

        if uploaded_csv is None:
            st.info("Carga un CSV enriquecido para generar el overlay.")
            return

        try:
            history_df = _read_uploaded_csv(uploaded_csv)
        except Exception as exc:
            st.error(f"No fue posible leer el CSV cargado: {exc}")
            return

        numeric_columns = list_positive_numeric_columns(history_df)

        if not numeric_columns:
            st.error("El CSV no tiene columnas numéricas positivas para log-log.")
            return

        default_x_index = 0
        default_y_index = 1 if len(numeric_columns) > 1 else 0

        x_column = st.selectbox(
            "Columna X para puntos RTA",
            options=numeric_columns,
            index=default_x_index,
        )

        y_column = st.selectbox(
            "Columna Y para puntos RTA",
            options=numeric_columns,
            index=default_y_index,
        )

        label_column = "date" if "date" in history_df.columns else None

        st.subheader("Matching manual")

        x_multiplier = st.number_input(
            "x_multiplier",
            min_value=1e-12,
            value=1.0,
            format="%.6g",
        )

        y_multiplier = st.number_input(
            "y_multiplier",
            min_value=1e-12,
            value=1.0,
            format="%.6g",
        )

        use_anchor = st.checkbox("Usar ancla y target")

        anchor_data_x = None
        anchor_data_y = None
        target_curve_x = None
        target_curve_y = None

        if use_anchor:
            anchor_data_x = st.number_input(
                "anchor_data_x",
                min_value=1e-12,
                value=1.0,
                format="%.6g",
            )

            anchor_data_y = st.number_input(
                "anchor_data_y",
                min_value=1e-12,
                value=1.0,
                format="%.6g",
            )

            target_curve_x = st.number_input(
                "target_curve_x",
                min_value=1e-12,
                value=1.0,
                format="%.6g",
            )

            target_curve_y = st.number_input(
                "target_curve_y",
                min_value=1e-12,
                value=1.0,
                format="%.6g",
            )

    with right_col:
        st.subheader("Overlay log-log")

        try:
            rta_points = build_overlay_points_from_dataframe(
                dataframe=history_df,
                x_column=x_column,
                y_column=y_column,
                label_column=label_column,
                date_column="date",
            )

            match_config = ManualMatchConfig(
                x_multiplier=x_multiplier,
                y_multiplier=y_multiplier,
                anchor_data_x=anchor_data_x,
                anchor_data_y=anchor_data_y,
                target_curve_x=target_curve_x,
                target_curve_y=target_curve_y,
            )

            overlay_result = build_overlay(
                type_curve=type_curve,
                rta_points=rta_points,
                match_config=match_config,
            )

            _plot_overlay_streamlit(overlay_result)

            st.subheader("Multiplicadores efectivos")

            st.json(
                {
                    "x_multiplier_input": match_config.x_multiplier,
                    "y_multiplier_input": match_config.y_multiplier,
                    "effective_x_multiplier": match_config.effective_x_multiplier,
                    "effective_y_multiplier": match_config.effective_y_multiplier,
                    "points_count": len(rta_points),
                    "method": method.value,
                    "curve_id": curve_id,
                    "x_column": x_column,
                    "y_column": y_column,
                }
            )

            with st.expander("Vista previa de datos"):
                st.dataframe(history_df.head(50), width="stretch")

        except Exception as exc:
            st.error(f"No fue posible construir el overlay: {exc}")


if __name__ == "__main__":
    main()