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
import matplotlib.ticker as mticker
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

MIN_MULTIPLIER = 1e-12
MAX_MULTIPLIER = 1e12


def _inject_arcade_css() -> None:
    """Inject lightweight arcade-inspired CSS for the joystick section."""
    st.markdown(
        """
        <style>
        .arcade-panel {
            background: linear-gradient(180deg, #17112b 0%, #0d0a1a 100%);
            border: 2px solid #ff4fd8;
            border-radius: 14px;
            padding: 14px 16px 10px 16px;
            box-shadow:
                0 0 10px rgba(255, 79, 216, 0.35),
                0 0 18px rgba(0, 255, 255, 0.18);
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }

        .arcade-title {
            color: #00f7ff;
            font-weight: 800;
            letter-spacing: 1px;
            font-size: 1.05rem;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
            text-shadow: 0 0 8px rgba(0, 247, 255, 0.6);
        }

        .arcade-subtitle {
            color: #ffd166;
            font-size: 0.92rem;
            margin-bottom: 0.7rem;
        }

        div[data-testid="stButton"] > button {
            border-radius: 12px;
            border: 2px solid #00f7ff;
            background: linear-gradient(180deg, #2a1e56 0%, #191233 100%);
            color: #ffffff;
            font-weight: 800;
            box-shadow:
                0 0 8px rgba(0, 247, 255, 0.20),
                inset 0 0 6px rgba(255, 79, 216, 0.10);
        }

        div[data-testid="stButton"] > button:hover {
            border-color: #ffd166;
            color: #ffd166;
            box-shadow:
                0 0 12px rgba(255, 209, 102, 0.30),
                inset 0 0 8px rgba(255, 79, 216, 0.14);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_registry() -> TypeCurveRegistry:
    """Load type curves from CSV files or demo fallback."""
    loader = TypeCurveLoader()
    return TypeCurveRegistry(loader.load_available(allow_demo_fallback=True))


def _plot_overlay_streamlit(overlay_result: Any) -> None:
    """Render overlay in Streamlit using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 6.5))

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

    # Fix: use plain-text tick labels to avoid Matplotlib ParseException
    # when log-scale values reach extremes like 1e-12 or 1e+12.
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda val, _: f"{val:g}" if val != 0 else "0"
    ))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda val, _: f"{val:g}" if val != 0 else "0"
    ))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    try:
        fig.tight_layout()
    except Exception:
        pass  # tight_layout can fail with extreme log-scale ranges; plot is still valid

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


def _clamp_multiplier(value: float) -> float:
    """Keep manual match multipliers inside Streamlit-safe bounds."""
    return min(max(float(value), MIN_MULTIPLIER), MAX_MULTIPLIER)


def _init_match_state() -> None:
    """Initialize session-state values for manual matching."""
    defaults = {
        "x_multiplier": 1.0,
        "y_multiplier": 1.0,
        "match_sensitivity_decades": 0.1,
        "use_anchor": False,
        "anchor_data_x": 1.0,
        "anchor_data_y": 1.0,
        "target_curve_x": 1.0,
        "target_curve_y": 1.0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Clamp only when outside bounds — avoids overwriting valid widget edits.
    for key in ("x_multiplier", "y_multiplier"):
        current = st.session_state[key]
        clamped = _clamp_multiplier(current)
        if clamped != current:
            st.session_state[key] = clamped


def _current_step_factor() -> float:
    """Return multiplicative step factor from log sensitivity."""
    step_decades = float(st.session_state["match_sensitivity_decades"])
    return 10**step_decades


def _move_left() -> None:
    """Shift matched points left in log scale."""
    st.session_state["x_multiplier"] = _clamp_multiplier(
        st.session_state["x_multiplier"] / _current_step_factor()
    )


def _move_right() -> None:
    """Shift matched points right in log scale."""
    st.session_state["x_multiplier"] = _clamp_multiplier(
        st.session_state["x_multiplier"] * _current_step_factor()
    )


def _move_up() -> None:
    """Shift matched points up in log scale."""
    st.session_state["y_multiplier"] = _clamp_multiplier(
        st.session_state["y_multiplier"] * _current_step_factor()
    )


def _move_down() -> None:
    """Shift matched points down in log scale."""
    st.session_state["y_multiplier"] = _clamp_multiplier(
        st.session_state["y_multiplier"] / _current_step_factor()
    )


def _reset_match() -> None:
    """Reset manual matching controls."""
    st.session_state["x_multiplier"] = 1.0
    st.session_state["y_multiplier"] = 1.0
    st.session_state["match_sensitivity_decades"] = 0.1
    st.session_state["use_anchor"] = False
    st.session_state["anchor_data_x"] = 1.0
    st.session_state["anchor_data_y"] = 1.0
    st.session_state["target_curve_x"] = 1.0
    st.session_state["target_curve_y"] = 1.0


def main() -> None:
    """Run the standalone M4 overlay screen."""
    st.set_page_config(page_title="M4 RTA Type-Curve Overlay", layout="wide")
    _init_match_state()
    _inject_arcade_css()

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

        st.markdown('<div class="arcade-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="arcade-title">Arcade Match Control</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="arcade-subtitle">'
            "Joystick logarítmico para mover la nube de puntos"
            "</div>",
            unsafe_allow_html=True,
        )

        st.select_slider(
            "Sensibilidad logarítmica (décadas por paso)",
            options=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
            key="match_sensitivity_decades",
            format_func=lambda value: f"{value:g} décadas/paso",
        )

        multiplier_col1, multiplier_col2 = st.columns(2)

        with multiplier_col1:
            st.number_input(
                "x_multiplier",
                min_value=MIN_MULTIPLIER,
                max_value=MAX_MULTIPLIER,
                format="%.6g",
                key="x_multiplier",
            )

        with multiplier_col2:
            st.number_input(
                "y_multiplier",
                min_value=MIN_MULTIPLIER,
                max_value=MAX_MULTIPLIER,
                format="%.6g",
                key="y_multiplier",
            )

        st.caption(
            "Cada pulsación aplica un factor multiplicativo de "
            f"{_current_step_factor():.6g} "
            f"(10^{st.session_state['match_sensitivity_decades']})."
        )

        up_cols = st.columns([1, 1, 1])
        with up_cols[1]:
            st.button("⬆ UP", width="stretch", on_click=_move_up)

        mid_cols = st.columns([1, 1, 1])
        with mid_cols[0]:
            st.button("⬅ LEFT", width="stretch", on_click=_move_left)
        with mid_cols[1]:
            st.button("RESET", width="stretch", on_click=_reset_match)
        with mid_cols[2]:
            st.button("RIGHT ➡", width="stretch", on_click=_move_right)

        down_cols = st.columns([1, 1, 1])
        with down_cols[1]:
            st.button("⬇ DOWN", width="stretch", on_click=_move_down)

        st.markdown("</div>", unsafe_allow_html=True)

        st.checkbox("Usar ancla y target", key="use_anchor")

        if st.session_state["use_anchor"]:
            anchor_col1, anchor_col2 = st.columns(2)

            with anchor_col1:
                st.number_input(
                    "anchor_data_x",
                    min_value=MIN_MULTIPLIER,
                    max_value=MAX_MULTIPLIER,
                    format="%.6g",
                    key="anchor_data_x",
                )
                st.number_input(
                    "anchor_data_y",
                    min_value=MIN_MULTIPLIER,
                    max_value=MAX_MULTIPLIER,
                    format="%.6g",
                    key="anchor_data_y",
                )

            with anchor_col2:
                st.number_input(
                    "target_curve_x",
                    min_value=MIN_MULTIPLIER,
                    max_value=MAX_MULTIPLIER,
                    format="%.6g",
                    key="target_curve_x",
                )
                st.number_input(
                    "target_curve_y",
                    min_value=MIN_MULTIPLIER,
                    max_value=MAX_MULTIPLIER,
                    format="%.6g",
                    key="target_curve_y",
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

            anchor_data_x = (
                float(st.session_state["anchor_data_x"])
                if st.session_state["use_anchor"]
                else None
            )
            anchor_data_y = (
                float(st.session_state["anchor_data_y"])
                if st.session_state["use_anchor"]
                else None
            )
            target_curve_x = (
                float(st.session_state["target_curve_x"])
                if st.session_state["use_anchor"]
                else None
            )
            target_curve_y = (
                float(st.session_state["target_curve_y"])
                if st.session_state["use_anchor"]
                else None
            )

            match_config = ManualMatchConfig(
                x_multiplier=_clamp_multiplier(st.session_state["x_multiplier"]),
                y_multiplier=_clamp_multiplier(st.session_state["y_multiplier"]),
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
                    "min_multiplier": MIN_MULTIPLIER,
                    "max_multiplier": MAX_MULTIPLIER,
                    "sensitivity_decades": st.session_state[
                        "match_sensitivity_decades"
                    ],
                    "step_factor": _current_step_factor(),
                    "points_count": len(rta_points),
                    "method": method.value,
                    "curve_id": curve_id,
                    "x_column": x_column,
                    "y_column": y_column,
                    "anchor_enabled": st.session_state["use_anchor"],
                }
            )

            with st.expander("Vista previa de datos"):
                st.dataframe(history_df.head(50), width="stretch")

        except Exception as exc:
            st.error(f"No fue posible construir el overlay: {exc}")


if __name__ == "__main__":
    main()