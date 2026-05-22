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
from src.services.rta_qc_service import (
    run_rta_qc,
    qc_severity_level,
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


def _init_reservoir_config_state(
    config: RTAConfig | None,
    well_id: str | None = None,
    pvt_json_path: Path | None = None,
) -> None:
    """Populate session state with RTAConfig values (only if not already set).

    Args:
        config: Saved RTAConfig scenario (or None to use defaults).
        well_id: Hub well ID — seeds rta_well_id if not yet in session state.
        pvt_json_path: Path to ``{well_id}_pvt_config_ui.json``; when present,
            Bo and μo are read from it to pre-populate the PVT fields.
    """
    defaults = config or RTAConfig(well_id=well_id or "W-001")

    # Read Bo / μo from M2 PVT JSON when available.
    bo_from_pvt: float | None = None
    mu_from_pvt: float | None = None
    if pvt_json_path is not None and pvt_json_path.exists():
        try:
            _pvt = json.loads(pvt_json_path.read_text(encoding="utf-8"))
            _bo = _pvt.get("bo_rb_stb")
            _mu = _pvt.get("mu_o_cp")
            if _bo is not None:
                bo_from_pvt = float(_bo)
            if _mu is not None:
                mu_from_pvt = float(_mu)
        except Exception:
            pass  # If PVT file is unreadable, fall back to scenario / defaults.

    _ss_defaults: dict[str, object] = {
        "rta_well_id": well_id or defaults.well_id,
        "rta_pi_psia": defaults.pi_psia,
        "rta_phi_frac": defaults.phi_frac,
        "rta_h_ft": defaults.h_ft,
        "rta_ct_1psi": defaults.ct_1psi,
        "rta_rw_ft": defaults.rw_ft,
        "rta_re_ft": defaults.re_ft or 0.0,
        "rta_area_acres": defaults.area_acres or 0.0,
        "rta_Bo_rb_stb": bo_from_pvt if bo_from_pvt is not None else defaults.Bo_rb_stb,
        "rta_mu_o_cp": mu_from_pvt if mu_from_pvt is not None else defaults.mu_o_cp,
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
    with st.expander("Parámetros de yacimiento / fluidos", expanded=True):
        st.caption(
            "Bo y μo se pre-cargan desde M2. Pi con advertencia si Pi < Pwf máx."
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
    auxiliary_series: list[tuple[str, list[RTAOverlayPoint]]] | None = None,
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

    # --- Auxiliary Blasingame series (qDdi, qDdid) ---
    if auxiliary_series:
        _aux_colors = ["#4ade80", "#f472b6"]   # green=integral, pink=derivative
        _aux_markers = ["s", "^"]
        for idx, (series_label, series_pts) in enumerate(auxiliary_series):
            if not series_pts:
                continue
            ax_x = [p.x * x_multiplier for p in series_pts]
            ax_y = [p.y * y_multiplier for p in series_pts]
            ax.loglog(
                ax_x, ax_y,
                linestyle="", marker=_aux_markers[idx % 2],
                color=_aux_colors[idx % 2],
                markersize=4, label=series_label, zorder=6, alpha=0.75,
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


_SENSITIVITY_LEVELS: dict[str, float] = {
    "Grueso": 0.5,   # ×3.2 per click  — posicionamiento inicial rápido
    "Medio":  0.1,   # ×1.26 per click — ajuste general (default)
    "Fino":   0.02,  # ×1.05 per click — refinamiento de precisión
}


def _init_match_state() -> None:
    """Initialize session-state values for manual matching."""
    defaults = {
        "x_multiplier": 1.0,
        "y_multiplier": 1.0,
        "match_sensitivity_label": "Medio",
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
    st.session_state["match_sensitivity_label"] = "Medio"
    st.session_state["match_sensitivity_decades"] = 0.1
    st.session_state["use_anchor"] = False
    st.session_state["anchor_data_x"] = 1.0
    st.session_state["anchor_data_y"] = 1.0
    st.session_state["target_curve_x"] = 1.0
    st.session_state["target_curve_y"] = 1.0


def _run_m4_overlay(
    *,
    well_id: str = "W-001",
    output_dir: Path = OUTPUT_DIR,
    show_title: bool = True,
    show_anchor: bool = True,
) -> None:
    """Core M4 overlay UI — shared by standalone main() and render_m4_joystick_embedded()."""
    from src.rta_type_curves.models import RTATypeCurveMethod as _RTAMethod

    _scenario_key = f"rta_scenario_loaded_{well_id}"
    if _scenario_key not in st.session_state:
        existing = load_rta_scenario(well_id, output_dir=output_dir)
        _pvt_json_path = PROJECT_ROOT / "data" / "ui_uploads" / f"{well_id}_pvt_config_ui.json"
        _init_reservoir_config_state(
            existing,
            well_id=well_id,
            pvt_json_path=_pvt_json_path if _pvt_json_path.exists() else None,
        )
        st.session_state[_scenario_key] = True

    _inject_arcade_css()

    if show_title:
        st.title("M4 — Overlay visual RTA vs curvas tipo")

    # Load type-curve registry
    try:
        registry = _load_registry()
    except Exception as exc:
        st.error(f"No fue posible cargar el registro de curvas tipo: {exc}")
        return

    # Reservoir config (expander expandido por defecto)
    reservoir_config = _render_reservoir_config()

    # Auto-load history ONLY from output/{well_id}_history_enriched.csv
    _auto_history_path = output_dir / f"{well_id}_history_enriched.csv"
    if not _auto_history_path.exists():
        st.info(
            "No hay historia enriquecida disponible. "
            "Ejecuta primero el **Paso 4 (M1-M2)** para generarla."
        )
        return

    try:
        history_df = pd.read_csv(_auto_history_path)
    except Exception as exc:
        st.error(f"Error leyendo historia enriquecida: {exc}")
        return

    st.caption(f"Historia: `{_auto_history_path.name}` — {len(history_df)} registros")

    # Pi warning: shown when Pi < Pwf máx
    if "pwf_used_psia" in history_df.columns:
        _max_pwf = float(history_df["pwf_used_psia"].dropna().max())
        if _max_pwf > reservoir_config.pi_psia:
            st.warning(
                f"Pi ({reservoir_config.pi_psia:.0f} psia) < Pwf máx "
                f"({_max_pwf:.0f} psia) — Delta-p negativo: curvas tipo no serán visibles. "
                "Ajusta Pi en los parámetros de yacimiento."
            )

    # Require standard RTA columns — no manual column selection
    if not _has_rta_transform_columns(history_df):
        st.warning(
            "El CSV de historia no tiene columnas RTA estándar "
            "(`well_id`, `date`, `qo_stb_d`, `pwf_used_psia`). "
            "Ejecuta M1-M2 para generar el archivo enriquecido."
        )
        return

    try:
        rta_transform_points = compute_rta_transforms(
            dataframe=history_df,
            pi_psia=reservoir_config.pi_psia,
        )
    except Exception as exc:
        st.error(f"Error calculando variables RTA: {exc}")
        return

    if not rta_transform_points:
        st.warning("Sin puntos RTA válidos (verifica Pi y datos de producción).")
        return

    # ---- Method tabs ----
    _TAB_LABELS  = ["🔬 Fetkovich", "📊 Palacio-Blasingame", "📈 Agarwal-Gardner"]
    _TAB_METHODS = [_RTAMethod.FETKOVICH, _RTAMethod.PALACIO_BLASINGAME, _RTAMethod.AGARWAL_GARDNER]
    _tabs = st.tabs(_TAB_LABELS)

    for _tab_idx, _tab in enumerate(_tabs):
        method = _TAB_METHODS[_tab_idx]
        _mval  = method.value

        with _tab:
            # Per-method session-state keys (avoids duplicate widget keys across tabs)
            _xk   = f"x_multiplier_{_mval}"
            _yk   = f"y_multiplier_{_mval}"
            _slk  = f"match_sensitivity_label_{_mval}"
            _snk  = f"match_sensitivity_decades_{_mval}"

            for _k, _v in [(_xk, 1.0), (_yk, 1.0), (_slk, "Medio"), (_snk, 0.1)]:
                if _k not in st.session_state:
                    st.session_state[_k] = _v

            # Callback closures for this method's joystick
            def _make_cbs(_mv: str, _xkey: str, _ykey: str, _skey: str):
                def _step() -> float:
                    return 10 ** float(st.session_state.get(_skey, 0.1))
                def _left():  st.session_state[_xkey] = _clamp_multiplier(st.session_state[_xkey] / _step())
                def _right(): st.session_state[_xkey] = _clamp_multiplier(st.session_state[_xkey] * _step())
                def _up():    st.session_state[_ykey] = _clamp_multiplier(st.session_state[_ykey] * _step())
                def _down():  st.session_state[_ykey] = _clamp_multiplier(st.session_state[_ykey] / _step())
                def _reset():
                    st.session_state[_xkey] = 1.0
                    st.session_state[_ykey] = 1.0
                    st.session_state[f"match_sensitivity_label_{_mv}"] = "Medio"
                    st.session_state[f"match_sensitivity_decades_{_mv}"] = 0.1
                return _left, _right, _up, _down, _reset

            _cb_left, _cb_right, _cb_up, _cb_down, _cb_reset = _make_cbs(_mval, _xk, _yk, _snk)

            # Curves for this method
            all_curves = registry.get_by_method(method)
            if not all_curves:
                st.error(f"Sin curvas para {_mval}.")
                continue

            _bdf_families = ("arps_bdf", "radial_bdf")
            bdf_ids = [c.curve_id for c in all_curves if c.curve_family in _bdf_families]

            st.caption(
                f"{len(all_curves)} curvas — {len(bdf_ids)} BDF + "
                f"{len(all_curves) - len(bdf_ids)} transientes"
            )

            ref_id = st.selectbox(
                "Curva BDF de referencia (para kh/k)",
                options=bdf_ids,
                key=f"ref_curve_{_mval}",
            )

            # 2-column layout: chart (wide) | joystick+results (narrow)
            _chart_col, _joy_col = st.columns([3, 1.2])

            # Current multipliers (read once; updated by callbacks on next rerun)
            _x_eff = _clamp_multiplier(st.session_state[_xk])
            _y_eff = _clamp_multiplier(st.session_state[_yk])

            # ---- Joystick / results column ----
            with _joy_col:
                st.markdown('<div class="arcade-panel">', unsafe_allow_html=True)
                st.markdown('<div class="arcade-title">Match Control</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="arcade-subtitle">joystick log · mueve la nube sobre la curva</div>',
                    unsafe_allow_html=True,
                )

                _sens = st.radio(
                    "Sensibilidad",
                    options=list(_SENSITIVITY_LEVELS.keys()),
                    horizontal=True,
                    key=_slk,
                )
                st.session_state[_snk] = _SENSITIVITY_LEVELS[_sens]
                _sens_desc = {
                    "Grueso": "x3.2/click",
                    "Medio":  "x1.26/click",
                    "Fino":   "x1.05/click",
                }
                st.caption(_sens_desc[_sens])

                _j1, _j2 = st.columns(2)
                with _j1:
                    st.number_input(
                        "X", min_value=MIN_MULTIPLIER, max_value=MAX_MULTIPLIER,
                        format="%.4g", key=_xk,
                    )
                with _j2:
                    st.number_input(
                        "Y", min_value=MIN_MULTIPLIER, max_value=MAX_MULTIPLIER,
                        format="%.4g", key=_yk,
                    )

                _uc = st.columns([1, 1, 1])
                with _uc[1]:
                    st.button("▲", use_container_width=True, on_click=_cb_up,    key=f"jup_{_mval}")
                _mc = st.columns([1, 1, 1])
                with _mc[0]:
                    st.button("◀", use_container_width=True, on_click=_cb_left,  key=f"jleft_{_mval}")
                with _mc[1]:
                    st.button("⟳", use_container_width=True, on_click=_cb_reset, key=f"jreset_{_mval}")
                with _mc[2]:
                    st.button("▶", use_container_width=True, on_click=_cb_right, key=f"jright_{_mval}")
                _dc = st.columns([1, 1, 1])
                with _dc[1]:
                    st.button("▼", use_container_width=True, on_click=_cb_down,  key=f"jdown_{_mval}")

                st.markdown("</div>", unsafe_allow_html=True)

                # Match parameters
                _mp = None
                def _fmt(v: float | None, d: int = 2) -> str:
                    return f"{v:.{d}f}" if v is not None else "—"

                try:
                    _mp = compute_match_params(
                        config=reservoir_config,
                        effective_x_multiplier=_x_eff,
                        effective_y_multiplier=_y_eff,
                        method=_mval,
                    )
                    st.metric("kh (mD·ft)", _fmt(_mp.kh_md_ft, 1))
                    st.metric("k (mD)",     _fmt(_mp.k_md, 3))
                    st.metric(
                        "N (MM STB)",
                        f"{_mp.n_vol_stb / 1e6:.3f}" if _mp.n_vol_stb else "—",
                    )
                    st.divider()
                    st.metric("re (ft)",      _fmt(_mp.re_ft, 0))
                    st.metric("Área (acres)", _fmt(_mp.area_acres, 1))
                    st.caption(f"X: {_x_eff:.4g}  |  Y: {_y_eff:.4g}")
                    if _mp.warnings:
                        for _w in _mp.warnings:
                            st.warning(_w, icon="⚠")
                except Exception as exc:
                    st.error(str(exc))

                # Save match
                if _mp is not None:
                    st.divider()
                    if st.button(
                        "📌 Guardar match",
                        use_container_width=True,
                        key=f"save_match_{_mval}",
                        help="Guarda kh/k/N de este método para la tabla comparativa",
                    ):
                        st.session_state[f"_saved_match_{_mval}"] = {
                            "method": _mval,
                            "ref_curve_id": ref_id,
                            "kh_md_ft": _mp.kh_md_ft,
                            "k_md": _mp.k_md,
                            "n_vol_stb": _mp.n_vol_stb,
                            "re_ft": _mp.re_ft,
                            "area_acres": _mp.area_acres,
                            "x_mult": _x_eff,
                            "y_mult": _y_eff,
                            "warnings": len(_mp.warnings),
                        }
                        st.success("Match guardado")

            # ---- Chart column ----
            with _chart_col:
                # QC warnings — visible (no expander)
                _method_pts = [p for p in rta_transform_points if p.method == method]
                if _method_pts:
                    _qc_results = run_rta_qc(
                        points=_method_pts,
                        effective_x_multiplier=_x_eff,
                        effective_y_multiplier=_y_eff,
                    )
                    _qc_level = qc_severity_level(_qc_results)
                    if _qc_level == "error":
                        for _r in _qc_results:
                            if _r.severity == "error":
                                st.error(f"**{_r.title}**  \n{_r.detail}", icon="🔴")
                    elif _qc_level == "warning":
                        for _r in _qc_results:
                            if _r.severity == "warning":
                                st.warning(f"**{_r.title}**  \n{_r.detail}", icon="⚠️")

                st.subheader("Overlay log-log")

                try:
                    _method_pts_chart = [p for p in rta_transform_points if p.method == method]
                    if not _method_pts_chart:
                        st.warning(f"Sin puntos RTA para {_mval}.")
                    else:
                        _rta_pts = _transform_points_to_overlay(_method_pts_chart)

                        # Blasingame auxiliary series (qDdi, qDdid)
                        _auxiliary: list[tuple[str, list[RTAOverlayPoint]]] = []
                        if _mval == "palacio_blasingame":
                            _pb_pts = [
                                p for p in rta_transform_points
                                if p.method.value == "palacio_blasingame"
                            ]
                            _int_pts = [
                                RTAOverlayPoint(
                                    x=p.material_balance_time,
                                    y=p.blasingame_integral,
                                    label=p.well_id,
                                    date=p.date,
                                )
                                for p in _pb_pts if p.blasingame_integral is not None
                            ]
                            _drv_pts = [
                                RTAOverlayPoint(
                                    x=p.material_balance_time,
                                    y=p.blasingame_derivative,
                                    label=p.well_id,
                                    date=p.date,
                                )
                                for p in _pb_pts if p.blasingame_derivative is not None
                            ]
                            if _int_pts:
                                _auxiliary.append(("qDdi (integral norm.)", _int_pts))
                            if _drv_pts:
                                _auxiliary.append(("qDdid (deriv. integral)", _drv_pts))

                        _match_cfg = ManualMatchConfig(
                            x_multiplier=_x_eff,
                            y_multiplier=_y_eff,
                        )
                        _png = _plot_all_curves_streamlit(
                            type_curves=all_curves,
                            raw_points=_rta_pts,
                            x_multiplier=_match_cfg.effective_x_multiplier,
                            y_multiplier=_match_cfg.effective_y_multiplier,
                            method_label=_mval,
                            selected_curve_id=ref_id,
                            auxiliary_series=_auxiliary or None,
                        )
                        st.image(_png, use_container_width=True)
                        st.caption(
                            f"{len(_method_pts_chart)} puntos  |  "
                            f"X: {_x_eff:.4g}  |  Y: {_y_eff:.4g}  |  "
                            f"Curva: {ref_id}"
                        )

                        # Export row (under chart)
                        if _mp is not None:
                            _e1, _e2, _e3 = st.columns(3)
                            _summary = build_match_summary(
                                match_params=_mp,
                                ref_curve_id=ref_id,
                                config=reservoir_config,
                            )
                            _summary_bytes = json.dumps(
                                _summary, indent=2, ensure_ascii=False
                            ).encode("utf-8")
                            with _e1:
                                st.download_button(
                                    "⬇ JSON",
                                    data=_summary_bytes,
                                    file_name=f"{reservoir_config.well_id}_rta_{_mval}.json",
                                    mime="application/json",
                                    use_container_width=True,
                                    key=f"dl_json_{_mval}",
                                )
                            with _e2:
                                st.download_button(
                                    "⬇ PNG",
                                    data=_png,
                                    file_name=f"{reservoir_config.well_id}_rta_{_mval}.png",
                                    mime="image/png",
                                    use_container_width=True,
                                    key=f"dl_png_{_mval}",
                                )
                            with _e3:
                                if st.button(
                                    "💾 Guardar",
                                    use_container_width=True,
                                    key=f"save_disk_{_mval}",
                                ):
                                    try:
                                        save_match_summary(_summary, output_dir=output_dir)
                                        save_overlay_png(
                                            _png, reservoir_config.well_id,
                                            output_dir=output_dir,
                                        )
                                        st.success("Guardado en output/")
                                    except Exception as exc:
                                        st.error(str(exc))

                except Exception as exc:
                    st.error(f"Error construyendo overlay: {exc}")

    # ---- Comparison table (full-width, below tabs) ----
    _METHOD_LABELS = {
        "fetkovich": "Fetkovich (SPE-4629)",
        "palacio_blasingame": "Palacio-Blasingame (SPE-25909)",
        "agarwal_gardner": "Agarwal-Gardner (SPE-49222)",
    }
    _saved = {
        m: st.session_state[f"_saved_match_{m}"]
        for m in ("fetkovich", "palacio_blasingame", "agarwal_gardner")
        if f"_saved_match_{m}" in st.session_state
    }

    if _saved:
        st.divider()
        st.subheader("Comparación de métodos")
        st.caption(
            "Cada fila corresponde al match guardado con 📌. "
            "Convergencia de kh entre métodos indica consistencia del match visual."
        )

        def _fmtc(val: float | None, dec: int = 2) -> str:
            return f"{val:.{dec}f}" if val is not None else "—"

        _cmp_rows = []
        for _m_val, _data in _saved.items():
            _kh = _data.get("kh_md_ft")
            _k  = _data.get("k_md")
            _n  = _data.get("n_vol_stb")
            _cmp_rows.append({
                "Método": _METHOD_LABELS.get(_m_val, _m_val),
                "kh (mD·ft)": _fmtc(_kh, 1),
                "k (mD)": _fmtc(_k, 3),
                "N vol. (MM STB)": f"{_n / 1e6:.3f}" if _n else "—",
                "re (ft)": _fmtc(_data.get("re_ft"), 0),
                "Área (acres)": _fmtc(_data.get("area_acres"), 1),
                "X": f"{_data['x_mult']:.4g}",
                "Y": f"{_data['y_mult']:.4g}",
                "Curva ref.": _data.get("ref_curve_id", "—"),
                "⚠": str(_data.get("warnings", 0)) if _data.get("warnings") else "",
            })

        st.dataframe(pd.DataFrame(_cmp_rows), use_container_width=True, hide_index=True)

        if st.button("🗑 Limpiar comparación"):
            for _m_val in ("fetkovich", "palacio_blasingame", "agarwal_gardner"):
                st.session_state.pop(f"_saved_match_{_m_val}", None)
            st.rerun()


def main() -> None:
    """Run the standalone M4 overlay screen."""
    st.set_page_config(page_title="M4 RTA Type-Curve Overlay", layout="wide")
    _run_m4_overlay(well_id="W-001", output_dir=OUTPUT_DIR, show_title=True, show_anchor=True)


def render_m4_joystick_embedded(well_id: str, output_dir: Path) -> None:
    """Render M4 joystick overlay embedded in the hub (no set_page_config, no anchor UI)."""
    _run_m4_overlay(well_id=well_id, output_dir=output_dir, show_title=False, show_anchor=False)


if __name__ == "__main__":
    main()