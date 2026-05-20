"""Standalone Streamlit UI for M4 type-curve visual overlay.

Run from the project root with:

    python -m streamlit run src/ui/m4_type_curve_overlay.py
"""

from __future__ import annotations

import io
import json
import math
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

from src.rta.models import RTAConfig
from src.rta_type_curves.loader import TypeCurveLoader
from src.rta_type_curves.overlay import ManualMatchConfig, RTAOverlayPoint, build_overlay
from src.rta_type_curves.registry import TypeCurveRegistry
from src.services.rta_overlay_points_service import (
    build_overlay_points_from_dataframe,
    list_positive_numeric_columns,
    load_history_for_overlay,
)
from src.services.rta_export_service import (
    build_match_summary,
    save_match_summary,
    save_overlay_png,
)
from src.services.rta_match_params_service import compute_match_params
from src.services.rta_scenario_service import (
    load_rta_scenario,
    save_rta_scenario,
    scenario_path,
)
from src.services.rta_transform_service import (
    RTATransformPoint,
    compute_rta_transforms,
)

MIN_MULTIPLIER = 1e-12
MAX_MULTIPLIER = 1e12

OUTPUT_DIR = PROJECT_ROOT / "output"

_RTA_TRANSFORM_REQUIRED_COLS = {"well_id", "date", "qo_stb_d", "pwf_used_psia"}


def _has_rta_transform_columns(df: pd.DataFrame) -> bool:
    return _RTA_TRANSFORM_REQUIRED_COLS.issubset(set(df.columns))


def _inject_arcade_css() -> None:
    """Inject arcade-inspired CSS for the joystick section."""
    st.markdown(
        """
        <style>
        .arcade-panel {
            background: #07050f;
            border: 1.5px solid #5b21b6;
            border-radius: 18px;
            padding: 16px 18px 14px 18px;
            box-shadow:
                0 0 0 1px rgba(91, 33, 182, 0.4),
                0 0 24px rgba(91, 33, 182, 0.22),
                0 0 48px rgba(0, 210, 255, 0.06),
                inset 0 1px 0 rgba(255, 255, 255, 0.04);
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }

        .arcade-title {
            color: #a78bfa;
            font-weight: 700;
            letter-spacing: 2.5px;
            font-size: 0.78rem;
            text-transform: uppercase;
            margin-bottom: 0.15rem;
            text-shadow: 0 0 12px rgba(167, 139, 250, 0.7);
        }

        .arcade-subtitle {
            color: rgba(200, 190, 230, 0.45);
            font-size: 0.78rem;
            margin-bottom: 0.9rem;
            letter-spacing: 0.3px;
        }

        div[data-testid="stButton"] > button {
            border-radius: 10px;
            border: 1.5px solid rgba(167, 139, 250, 0.5);
            background: linear-gradient(160deg, #1e1333 0%, #110c22 100%);
            color: #c4b5fd;
            font-weight: 700;
            font-size: 1.1rem;
            letter-spacing: 0.5px;
            transition: all 0.12s ease;
            box-shadow:
                0 0 6px rgba(139, 92, 246, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }

        div[data-testid="stButton"] > button:hover {
            border-color: #a78bfa;
            color: #ede9fe;
            background: linear-gradient(160deg, #2d1f50 0%, #1a1238 100%);
            box-shadow:
                0 0 14px rgba(167, 139, 250, 0.35),
                inset 0 1px 0 rgba(255, 255, 255, 0.08);
            transform: translateY(-1px);
        }

        div[data-testid="stButton"] > button:active {
            transform: translateY(1px);
            box-shadow: 0 0 4px rgba(139, 92, 246, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _transform_points_to_overlay(
    points: list[RTATransformPoint],
) -> list[RTAOverlayPoint]:
    return [p.to_overlay_point() for p in points]


def _init_reservoir_config_state(config: RTAConfig | None) -> None:
    """Populate session state with RTAConfig values (only if not already set)."""
    defaults = config or RTAConfig(well_id="W-001")
    _ss_defaults: dict[str, object] = {
        "rta_well_id": defaults.well_id,
        "rta_pi_psia": defaults.pi_psia,
        "rta_phi_frac": defaults.phi_frac,
        "rta_h_ft": defaults.h_ft,
        "rta_ct_1psi": defaults.ct_1psi,
        "rta_rw_ft": defaults.rw_ft,
        "rta_re_ft": defaults.re_ft or 0.0,
        "rta_area_acres": defaults.area_acres or 0.0,
        "rta_Bo_rb_stb": defaults.Bo_rb_stb,
        "rta_mu_o_cp": defaults.mu_o_cp,
        "rta_CA": defaults.CA,
    }
    for key, value in _ss_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _current_rta_config() -> RTAConfig:
    """Build RTAConfig from current session state values."""
    re_ft_val = float(st.session_state["rta_re_ft"])
    area_val = float(st.session_state["rta_area_acres"])
    return RTAConfig(
        well_id=str(st.session_state["rta_well_id"]).strip() or "W-001",
        pi_psia=float(st.session_state["rta_pi_psia"]),
        phi_frac=float(st.session_state["rta_phi_frac"]),
        h_ft=float(st.session_state["rta_h_ft"]),
        ct_1psi=float(st.session_state["rta_ct_1psi"]),
        rw_ft=float(st.session_state["rta_rw_ft"]),
        re_ft=re_ft_val if re_ft_val > 0 else None,
        area_acres=area_val if area_val > 0 else None,
        Bo_rb_stb=float(st.session_state["rta_Bo_rb_stb"]),
        mu_o_cp=float(st.session_state["rta_mu_o_cp"]),
        CA=float(st.session_state["rta_CA"]),
    )


_FT2_PER_ACRE = 43_560.0


def _on_re_ft_change() -> None:
    """Sync area when user edits drainage radius."""
    re = float(st.session_state.get("rta_re_ft", 0.0))
    if re > 0:
        st.session_state["rta_area_acres"] = round(
            math.pi * re ** 2 / _FT2_PER_ACRE, 4
        )


def _on_area_change() -> None:
    """Sync radius when user edits drainage area."""
    area = float(st.session_state.get("rta_area_acres", 0.0))
    if area > 0:
        st.session_state["rta_re_ft"] = round(
            math.sqrt(area * _FT2_PER_ACRE / math.pi), 2
        )


def _render_reservoir_config() -> RTAConfig:
    """Render the reservoir/fluid configuration panel and return current config."""
    with st.expander("Parámetros de yacimiento / fluidos", expanded=False):
        st.caption(
            "Estos valores alimentan el cálculo de variables RTA físicas "
            "(Δp, tasa normalizada, MBT) y, en el próximo commit, los parámetros "
            "de yacimiento desde el match point (kh, k, OOIP)."
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.text_input("ID del pozo", key="rta_well_id")
            st.number_input(
                "pi — Presión inicial (psia)",
                min_value=100.0,
                max_value=20000.0,
                step=50.0,
                format="%.1f",
                key="rta_pi_psia",
            )
            st.number_input(
                "φ — Porosidad (fracción)",
                min_value=0.001,
                max_value=0.999,
                step=0.005,
                format="%.4f",
                key="rta_phi_frac",
            )
            st.number_input(
                "h — Espesor neto (ft)",
                min_value=0.1,
                max_value=5000.0,
                step=1.0,
                format="%.1f",
                key="rta_h_ft",
            )
            st.number_input(
                "ct — Compresibilidad total (1/psi)",
                min_value=1e-7,
                max_value=1e-2,
                step=1e-6,
                format="%.2e",
                key="rta_ct_1psi",
            )

        with col_b:
            st.number_input(
                "rw — Radio de pozo (ft)",
                min_value=0.01,
                max_value=5.0,
                step=0.01,
                format="%.4f",
                key="rta_rw_ft",
            )
            st.number_input(
                "re — Radio de drene (ft)",
                min_value=0.0,
                max_value=50000.0,
                step=10.0,
                format="%.1f",
                key="rta_re_ft",
                on_change=_on_re_ft_change,
                help="Al cambiar re, el área se recalcula automáticamente.",
            )
            st.number_input(
                "Área de drene (acres)",
                min_value=0.0,
                max_value=100000.0,
                step=1.0,
                format="%.3f",
                key="rta_area_acres",
                on_change=_on_area_change,
                help="Al cambiar el área, re se recalcula automáticamente.",
            )
            st.number_input(
                "Bo — Factor volumétrico de aceite (RB/STB)",
                min_value=0.5,
                max_value=5.0,
                step=0.01,
                format="%.4f",
                key="rta_Bo_rb_stb",
            )
            st.number_input(
                "μo — Viscosidad de aceite (cp)",
                min_value=0.1,
                max_value=1000.0,
                step=0.1,
                format="%.3f",
                key="rta_mu_o_cp",
            )
            st.number_input(
                "CA — Factor de forma Dietz",
                min_value=0.1,
                max_value=100.0,
                step=0.1,
                format="%.4f",
                help="31.62 = drene circular (default). Ver Tabla B-1, Earlougher 1977.",
                key="rta_CA",
            )

        try:
            config = _current_rta_config()
        except Exception as exc:
            st.error(f"Configuración inválida: {exc}")
            config = RTAConfig(well_id="W-001")

        save_col, status_col = st.columns([1, 3])
        with save_col:
            if st.button("Guardar escenario", type="primary"):
                try:
                    saved_path = save_rta_scenario(config, output_dir=OUTPUT_DIR)
                    st.session_state["rta_scenario_saved"] = str(saved_path)
                except Exception as exc:
                    st.error(f"No se pudo guardar: {exc}")

        with status_col:
            saved = st.session_state.get("rta_scenario_saved")
            if saved:
                st.success(f"Guardado: `{Path(saved).name}`")
            else:
                saved_file = scenario_path(config.well_id, output_dir=OUTPUT_DIR)
                if saved_file.exists():
                    st.info(f"Escenario previo en disco: `{saved_file.name}`")

    return config


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


def _plot_all_curves_streamlit(
    type_curves: list[Any],
    raw_points: list[RTAOverlayPoint],
    x_multiplier: float,
    y_multiplier: float,
    method_label: str,
    selected_curve_id: str | None = None,
) -> bytes:
    """Plot the full type-curve family + data cloud on one log-log figure."""
    from src.rta_type_curves.models import TypeCurve  # local to avoid circular

    fig, ax = plt.subplots(figsize=(11, 7))

    bdf_curves = [c for c in type_curves if c.curve_family != "transient_stem"]
    transient_curves = [c for c in type_curves if c.curve_family == "transient_stem"]

    # --- BDF stems (solid) ---
    bdf_colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for i, curve in enumerate(bdf_curves):
        xs = [p.x for p in curve.points]
        ys = [p.y for p in curve.points]
        is_selected = curve.curve_id == selected_curve_id
        ax.loglog(
            xs, ys,
            color=bdf_colors[i % len(bdf_colors)],
            linewidth=2.0 if is_selected else 1.2,
            linestyle="-",
            label=curve.curve_id,
            zorder=3,
            alpha=1.0 if is_selected else 0.75,
        )

    # --- Transient stems (dashed, grouped color) ---
    n_trans = len(transient_curves)
    for i, curve in enumerate(transient_curves):
        xs = [p.x for p in curve.points]
        ys = [p.y for p in curve.points]
        frac = i / max(n_trans - 1, 1)
        color = plt.cm.autumn(0.15 + 0.7 * frac)  # type: ignore[attr-defined]
        re_rw_label = curve.curve_id.split("re_rw_")[-1] if "re_rw_" in curve.curve_id else curve.curve_id
        ax.loglog(
            xs, ys,
            color=color,
            linewidth=0.9,
            linestyle="--",
            label=f"re/rw={re_rw_label}" if i == 0 or i == n_trans - 1 else "_nolegend_",
            zorder=2,
            alpha=0.65,
        )

    # --- Raw data points ---
    if raw_points:
        raw_x = [p.x for p in raw_points]
        raw_y = [p.y for p in raw_points]
        ax.loglog(
            raw_x, raw_y,
            linestyle="", marker="o", color="#f97316",
            markersize=4, label="Datos raw", zorder=5, alpha=0.55,
        )

        # --- Matched (shifted) data ---
        matched_x = [v * x_multiplier for v in raw_x]
        matched_y = [v * y_multiplier for v in raw_y]
        ax.loglog(
            matched_x, matched_y,
            linestyle="", marker="o", color="#22d3ee",
            markersize=4, label="Datos ajustados", zorder=6, alpha=0.85,
        )

    # Axis labels from first curve
    x_lbl = type_curves[0].x_label if type_curves else "tDd"
    y_lbl = type_curves[0].y_label if type_curves else "qDd"
    ax.set_xlabel(x_lbl, fontsize=10)
    ax.set_ylabel(y_lbl, fontsize=10)
    ax.set_title(f"{method_label} — familia completa de curvas tipo", fontsize=11)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)

    # Compact legend
    ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.85)

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
        pass

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    png_bytes = buf.getvalue()

    st.pyplot(fig, width="stretch")
    plt.close(fig)
    return png_bytes


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

    # Pre-load saved scenario (if any) to seed reservoir config defaults.
    if "rta_scenario_loaded" not in st.session_state:
        existing = load_rta_scenario("W-001", output_dir=OUTPUT_DIR)
        _init_reservoir_config_state(existing)
        st.session_state["rta_scenario_loaded"] = True

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

    reservoir_config = _render_reservoir_config()

    _latest_png_bytes: bytes | None = None

    left_col, right_col, params_col = st.columns([1, 2.2, 0.9])

    with left_col:
        st.subheader("Curva tipo")

        method = st.selectbox(
            "Método",
            options=available_methods,
            format_func=lambda value: value.value,
        )

        all_method_curves = registry.get_by_method(method)

        if not all_method_curves:
            st.error(f"No hay curvas disponibles para el método {method.value}.")
            return

        bdf_curve_ids = [
            c.curve_id for c in all_method_curves
            if c.curve_family != "transient_stem"
        ]

        st.caption(
            f"{len(all_method_curves)} curvas cargadas "
            f"({len(bdf_curve_ids)} BDF + "
            f"{len(all_method_curves) - len(bdf_curve_ids)} transientes)"
        )
        st.warning(
            "Curvas DEMO — analíticamente correctas pero no digitalizadas "
            "desde los papers de referencia. No usar para interpretación final."
        )

        with st.expander("Curva de referencia (para kh / k)"):
            st.caption(
                "Selecciona la curva BDF sobre la que visualmente cae la nube "
                "de puntos ajustados. Se usa solo para calcular kh y k."
            )
            ref_curve_id = st.selectbox(
                "Curva BDF de match",
                options=bdf_curve_ids,
                label_visibility="collapsed",
            )
            type_curve = registry.get(method, ref_curve_id)
            st.caption(f"Familia: {type_curve.curve_family} · Fuente: {type_curve.source}")

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

        # --- Determine overlay source ---
        # If the CSV has qo_stb_d + pwf_used_psia, compute physically-meaningful
        # RTA transforms using pi_psia from the reservoir config panel.
        # Otherwise fall back to arbitrary column selection.
        use_rta_transforms = _has_rta_transform_columns(history_df)

        rta_transform_points: list[RTATransformPoint] = []
        x_column: str = ""
        y_column: str = ""
        label_column: str | None = "date" if "date" in history_df.columns else None

        if use_rta_transforms:
            st.success(
                "CSV con columnas RTA detectadas (`qo_stb_d`, `pwf_used_psia`). "
                "Usando pi = **{:.0f} psia** del panel de configuración.".format(
                    reservoir_config.pi_psia
                )
            )
            try:
                rta_transform_points = compute_rta_transforms(
                    dataframe=history_df,
                    pi_psia=reservoir_config.pi_psia,
                )
            except Exception as exc:
                st.error(f"Error al calcular variables RTA: {exc}")
                return
        else:
            st.info(
                "CSV sin columnas RTA estándar. Selecciona columnas X/Y manualmente. "
                "Para usar variables físicas (MBT, tasa normalizada), el CSV debe "
                "incluir `well_id`, `date`, `qo_stb_d`, `pwf_used_psia`."
            )
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

        st.subheader("Matching manual")

        st.markdown('<div class="arcade-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="arcade-title">Match Control</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="arcade-subtitle">'
            "joystick logarítmico · mueve la nube de puntos sobre la curva tipo"
            "</div>",
            unsafe_allow_html=True,
        )

        st.select_slider(
            "Sensibilidad",
            options=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
            key="match_sensitivity_decades",
            format_func=lambda value: f"{value:g}",
        )

        multiplier_col1, multiplier_col2 = st.columns(2)

        with multiplier_col1:
            st.number_input(
                "Escala X",
                min_value=MIN_MULTIPLIER,
                max_value=MAX_MULTIPLIER,
                format="%.6g",
                key="x_multiplier",
            )

        with multiplier_col2:
            st.number_input(
                "Escala Y",
                min_value=MIN_MULTIPLIER,
                max_value=MAX_MULTIPLIER,
                format="%.6g",
                key="y_multiplier",
            )

        st.caption(
            f"Paso actual: ×{_current_step_factor():.6g} por pulsación"
        )

        up_cols = st.columns([1, 1, 1])
        with up_cols[1]:
            st.button("▲", width="stretch", on_click=_move_up)

        mid_cols = st.columns([1, 1, 1])
        with mid_cols[0]:
            st.button("◀", width="stretch", on_click=_move_left)
        with mid_cols[1]:
            st.button("⟳", width="stretch", on_click=_reset_match)
        with mid_cols[2]:
            st.button("▶", width="stretch", on_click=_move_right)

        down_cols = st.columns([1, 1, 1])
        with down_cols[1]:
            st.button("▼", width="stretch", on_click=_move_down)

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
            if use_rta_transforms and rta_transform_points:
                from src.rta_type_curves.models import RTATypeCurveMethod
                method_map = {m.value: m for m in RTATypeCurveMethod}
                selected_method_enum = method_map.get(method.value, method)
                method_points = [
                    p for p in rta_transform_points
                    if p.method == selected_method_enum
                ]
                if not method_points:
                    st.warning(f"Sin puntos para el método {method.value}.")
                    return
                rta_points = _transform_points_to_overlay(method_points)
                st.caption(
                    f"Ejes: **{method_points[0].x_label}** (X) · "
                    f"**{method_points[0].y_label}** (Y) · "
                    f"{len(method_points)} puntos válidos"
                )
            else:
                rta_points = build_overlay_points_from_dataframe(
                    dataframe=history_df,
                    x_column=x_column,
                    y_column=y_column,
                    label_column=label_column,
                    date_column="date",
                )

            match_config = ManualMatchConfig(
                x_multiplier=_clamp_multiplier(st.session_state["x_multiplier"]),
                y_multiplier=_clamp_multiplier(st.session_state["y_multiplier"]),
                anchor_data_x=(
                    float(st.session_state["anchor_data_x"])
                    if st.session_state["use_anchor"] else None
                ),
                anchor_data_y=(
                    float(st.session_state["anchor_data_y"])
                    if st.session_state["use_anchor"] else None
                ),
                target_curve_x=(
                    float(st.session_state["target_curve_x"])
                    if st.session_state["use_anchor"] else None
                ),
                target_curve_y=(
                    float(st.session_state["target_curve_y"])
                    if st.session_state["use_anchor"] else None
                ),
            )

            _latest_png_bytes = _plot_all_curves_streamlit(
                type_curves=all_method_curves,
                raw_points=rta_points,
                x_multiplier=match_config.effective_x_multiplier,
                y_multiplier=match_config.effective_y_multiplier,
                method_label=method.value,
                selected_curve_id=type_curve.curve_id,
            )

            st.subheader("Multiplicadores efectivos")

            match_info: dict[str, object] = {
                "x_multiplier_input": match_config.x_multiplier,
                "y_multiplier_input": match_config.y_multiplier,
                "effective_x_multiplier": match_config.effective_x_multiplier,
                "effective_y_multiplier": match_config.effective_y_multiplier,
                "sensitivity_decades": st.session_state["match_sensitivity_decades"],
                "step_factor": _current_step_factor(),
                "points_count": len(rta_points),
                "method": method.value,
                "curve_id": ref_curve_id,
                "anchor_enabled": st.session_state["use_anchor"],
            }
            if use_rta_transforms:
                match_info["data_source"] = "rta_transform_service"
                match_info["pi_psia"] = reservoir_config.pi_psia
            else:
                match_info["data_source"] = "manual_column_selection"
                match_info["x_column"] = x_column
                match_info["y_column"] = y_column
            st.json(match_info)

            with st.expander("Vista previa de datos"):
                st.dataframe(history_df.head(50), width="stretch")

        except Exception as exc:
            st.error(f"No fue posible construir el overlay: {exc}")

    # --- Parameters panel — always rendered next to the plot ---
    with params_col:
        st.subheader("Parámetros")
        st.caption("DEMO")

        def _fmt(val: float | None, dec: int = 2) -> str:
            return f"{val:.{dec}f}" if val is not None else "—"

        _mp = None
        try:
            _mp = compute_match_params(
                config=reservoir_config,
                effective_x_multiplier=_clamp_multiplier(
                    st.session_state.get("x_multiplier", 1.0)
                ),
                effective_y_multiplier=_clamp_multiplier(
                    st.session_state.get("y_multiplier", 1.0)
                ),
                method=method.value,
            )
            st.metric("kh (mD·ft)", _fmt(_mp.kh_md_ft, 1))
            st.metric("k (mD)", _fmt(_mp.k_md, 3))
            st.metric(
                "N vol. (MM STB)",
                f"{_mp.n_vol_stb / 1e6:.3f}" if _mp.n_vol_stb else "—",
            )
            st.divider()
            st.metric("re (ft)", _fmt(_mp.re_ft, 0))
            st.metric("Área (acres)", _fmt(_mp.area_acres, 1))
            st.metric("ln(re/rw)−½", _fmt(_mp.ln_re_rw_term, 3))
            st.divider()
            st.caption(f"Escala X: {st.session_state.get('x_multiplier', 1.0):.4g}")
            st.caption(f"Escala Y: {st.session_state.get('y_multiplier', 1.0):.4g}")

            if _mp.warnings:
                with st.expander("⚠ Advertencias", expanded=False):
                    for w in _mp.warnings:
                        st.warning(w, icon="⚠")

            with st.expander("JSON", expanded=False):
                st.json(_mp.as_dict())

        except Exception as exc:
            st.error(str(exc))

        # --- Export section ---
        if _mp is not None and _latest_png_bytes is not None:
            st.divider()
            with st.expander("Exportar", expanded=False):
                _summary = build_match_summary(
                    match_params=_mp,
                    ref_curve_id=ref_curve_id,
                    config=reservoir_config,
                )
                _summary_bytes = json.dumps(
                    _summary, indent=2, ensure_ascii=False
                ).encode("utf-8")

                st.download_button(
                    "⬇ JSON",
                    data=_summary_bytes,
                    file_name=f"{reservoir_config.well_id}_rta_match_summary.json",
                    mime="application/json",
                    use_container_width=True,
                )
                st.download_button(
                    "⬇ PNG",
                    data=_latest_png_bytes,
                    file_name=f"{reservoir_config.well_id}_rta_overlay.png",
                    mime="image/png",
                    use_container_width=True,
                )
                if st.button("💾 Guardar en output/", use_container_width=True):
                    try:
                        saved_json = save_match_summary(_summary, output_dir=OUTPUT_DIR)
                        saved_png = save_overlay_png(
                            _latest_png_bytes, reservoir_config.well_id, output_dir=OUTPUT_DIR
                        )
                        st.success(
                            f"`{saved_json.name}`  \n`{saved_png.name}`"
                        )
                    except Exception as exc:
                        st.error(f"Error al exportar: {exc}")


if __name__ == "__main__":
    main()