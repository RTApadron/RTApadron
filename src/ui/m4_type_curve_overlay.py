"""Standalone Streamlit UI for M4 type-curve visual overlay.

Run from the project root with:

    python -m streamlit run src/ui/m4_type_curve_overlay.py
"""

from __future__ import annotations

import base64
import io
import json
import math
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import streamlit.components.v1 as _st_components

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import plotly.graph_objects as go
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

# Standard matplotlib tab10 palette (matches _generate_overlay_png color order)
_TAB10_HEX = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

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
    _model_defaults = RTAConfig(well_id=well_id or "W-001")

    # Sanity-filter scenario values: if a saved value equals the widget minimum
    # (meaning it was saved accidentally at minimum), fall back to model defaults.
    def _safe(saved: float | None, lo: float, hi: float, default: float) -> float:
        if saved is None or not (lo < saved <= hi):
            return default
        return saved

    # Determine Pi first — needed to evaluate PVT at reservoir pressure.
    _pi = (
        _safe(config.pi_psia, 500.0, 20000.0, _model_defaults.pi_psia)
        if config is not None
        else _model_defaults.pi_psia
    )

    # Read Bo/μo from M2 PVT JSON — always takes priority over saved scenario.
    bo_from_pvt: float | None = None
    mu_from_pvt: float | None = None
    if pvt_json_path is not None and pvt_json_path.exists():
        try:
            _pvt = json.loads(pvt_json_path.read_text(encoding="utf-8"))
            # Direct saved values (present when user saved via M2 panel)
            _bo = _pvt.get("bo_rb_stb")
            _mu = _pvt.get("mu_o_cp")
            if _bo is not None:
                bo_from_pvt = float(_bo)
            if _mu is not None:
                mu_from_pvt = float(_mu)
            # Fall back: compute from correlation at Pi using the M2 correlation inputs
            if (bo_from_pvt is None or mu_from_pvt is None) and all(
                _pvt.get(k) for k in ("api", "gamma_g", "temp_f", "rsb_scf_stb")
            ):
                from src.services.pvt_service import PVTTableInput, compute_pvt_table
                _pvt_inp = PVTTableInput(
                    api=float(_pvt["api"]),
                    gamma_g=float(_pvt["gamma_g"]),
                    t_f=float(_pvt["temp_f"]),
                    rsb_scf_stb=float(_pvt["rsb_scf_stb"]),
                    p_min_psia=14.7,
                    p_max_psia=max(float(_pi), 5000.0),
                    n_points=10,
                    correlation=str(_pvt.get("oil_corr", "standing")),
                )
                _pb, _pts = compute_pvt_table(_pvt_inp)
                if _pts:
                    _pt_pi = min(_pts, key=lambda p: abs(p.p_psia - _pi))
                    if bo_from_pvt is None:
                        bo_from_pvt = _pt_pi.bo_rb_stb
                    if mu_from_pvt is None:
                        mu_from_pvt = _pt_pi.mu_o_cp
        except Exception:
            pass  # If PVT file is unreadable, fall back to scenario / defaults.

    if config is not None:
        # _pi already computed above
        _phi   = _safe(config.phi_frac,  0.005,  1.0,     _model_defaults.phi_frac)
        _h     = _safe(config.h_ft,      0.5,    5000.0,  _model_defaults.h_ft)
        _ct    = _safe(config.ct_1psi,   1e-7,   1e-2,    _model_defaults.ct_1psi)
        _rw    = _safe(config.rw_ft,     0.05,   5.0,     _model_defaults.rw_ft)
        _ca    = _safe(config.CA,        1.0,    200.0,   _model_defaults.CA)
        _re    = config.re_ft if (config.re_ft and config.re_ft > 1.0) else None
        _area  = config.area_acres if (config.area_acres and config.area_acres > 0.001) else None
        _swi   = config.swi_frac if config.swi_frac is not None else 0.0
        # Bo and μo: M2 PVT takes priority; only fall back to scenario if PVT unavailable
        _bo    = bo_from_pvt if bo_from_pvt is not None else _safe(config.Bo_rb_stb, 0.7, 5.0, _model_defaults.Bo_rb_stb)
        _mu    = mu_from_pvt if mu_from_pvt is not None else _safe(config.mu_o_cp,  0.05, 1000.0, _model_defaults.mu_o_cp)
    else:
        _phi   = _model_defaults.phi_frac
        _h     = _model_defaults.h_ft
        _ct    = _model_defaults.ct_1psi
        _rw    = _model_defaults.rw_ft
        _ca    = _model_defaults.CA          # 31.62 circular drainage default
        _re    = None
        _area  = None
        _swi   = 0.0
        _bo    = bo_from_pvt if bo_from_pvt is not None else _model_defaults.Bo_rb_stb
        _mu    = mu_from_pvt if mu_from_pvt is not None else _model_defaults.mu_o_cp

    _ss_defaults: dict[str, object] = {
        "rta_well_id": well_id or (config.well_id if config else "W-001"),
        "rta_pi_psia": _pi,
        "rta_phi_frac": _phi,
        "rta_h_ft": _h,
        "rta_ct_1psi": _ct,
        "rta_rw_ft": _rw,
        "rta_re_ft": _re or 0.0,
        "rta_area_acres": _area or 0.0,
        "rta_swi_frac": _swi,
        "rta_Bo_rb_stb": _bo,
        "rta_mu_o_cp": _mu,
        "rta_CA": _ca,
    }
    for key, value in _ss_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Always force Bo/μo from M2 PVT (even if session_state already had a value)
    if bo_from_pvt is not None:
        st.session_state["rta_Bo_rb_stb"] = bo_from_pvt
    if mu_from_pvt is not None:
        st.session_state["rta_mu_o_cp"] = mu_from_pvt
    # Track the Pi used for this PVT computation so we can detect future changes
    st.session_state["rta_pvt_last_pi"] = _pi


def _current_rta_config() -> RTAConfig:
    """Build RTAConfig from current session state values."""
    re_ft_val = float(st.session_state["rta_re_ft"])
    area_val = float(st.session_state["rta_area_acres"])
    swi_val = float(st.session_state["rta_swi_frac"])
    return RTAConfig(
        well_id=str(st.session_state["rta_well_id"]).strip() or "W-001",
        pi_psia=float(st.session_state["rta_pi_psia"]),
        phi_frac=float(st.session_state["rta_phi_frac"]),
        h_ft=float(st.session_state["rta_h_ft"]),
        ct_1psi=float(st.session_state["rta_ct_1psi"]),
        rw_ft=float(st.session_state["rta_rw_ft"]),
        re_ft=re_ft_val if re_ft_val > 0 else None,
        area_acres=area_val if area_val > 0 else None,
        swi_frac=swi_val if swi_val > 0 else None,
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


def _render_reservoir_config(hub_well_id: str = "") -> RTAConfig:
    """Render the reservoir/fluid configuration panel and return current config."""
    # Always force rta_well_id to match the hub's well_id so saved files are
    # consistently named and M5 can find them.
    if hub_well_id:
        st.session_state["rta_well_id"] = hub_well_id

    with st.expander("Parámetros de yacimiento / fluidos", expanded=True):
        st.caption(
            "Bo y μo se pre-cargan desde M2. Pi con advertencia si Pi < Pwf máx."
        )

        # If Pi changed since the last PVT evaluation, recompute Bo/μo at new Pi.
        _cur_pi = float(st.session_state.get("rta_pi_psia", 0.0))
        if _cur_pi > 0 and st.session_state.get("rta_pvt_last_pi") != _cur_pi:
            _wid = hub_well_id or str(st.session_state.get("rta_well_id", ""))
            _pvt_p = PROJECT_ROOT / "data" / "ui_uploads" / f"{_wid}_pvt_config_ui.json"
            if _pvt_p.exists():
                try:
                    _pvt = json.loads(_pvt_p.read_text(encoding="utf-8"))
                    _api = _pvt.get("api"); _gg = _pvt.get("gamma_g")
                    _tf  = _pvt.get("temp_f"); _rsb = _pvt.get("rsb_scf_stb")
                    if all(v is not None for v in (_api, _gg, _tf, _rsb)):
                        from src.services.pvt_service import PVTTableInput, compute_pvt_table
                        _pvt_inp = PVTTableInput(
                            api=float(_api), gamma_g=float(_gg),
                            t_f=float(_tf), rsb_scf_stb=float(_rsb),
                            p_min_psia=14.7, p_max_psia=max(_cur_pi, 5000.0),
                            n_points=10,
                            correlation=str(_pvt.get("oil_corr", "standing")),
                        )
                        _pb2, _pts2 = compute_pvt_table(_pvt_inp)
                        if _pts2:
                            _pt = min(_pts2, key=lambda p: abs(p.p_psia - _cur_pi))
                            st.session_state["rta_Bo_rb_stb"] = _pt.bo_rb_stb
                            st.session_state["rta_mu_o_cp"]   = _pt.mu_o_cp
                            st.session_state["rta_pvt_last_pi"] = _cur_pi
                except Exception:
                    pass

        col_a, col_b = st.columns(2)

        with col_a:
            st.text_input("ID del pozo", key="rta_well_id", disabled=bool(hub_well_id))
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
            st.number_input(
                "Swi — Saturación de agua irreducible (fracción)",
                min_value=0.0,
                max_value=0.999,
                step=0.01,
                format="%.4f",
                help=(
                    "Corrige el OOIP volumétrico: N = φ·h·A·(1−Swi)/(5.615·Bo). "
                    "Deja en 0 si no tienes datos (sobreestima OOIP)."
                ),
                key="rta_swi_frac",
            )

        try:
            config = _current_rta_config()
        except Exception as exc:
            st.error(f"Configuración inválida: {exc}")
            config = RTAConfig(well_id="W-001")

        # Sanity check — warn when parameters look like widget minimums
        _phys_warnings = []
        if config.pi_psia < 500:
            _phys_warnings.append(f"⚠ Pi = {config.pi_psia:.0f} psia parece muy bajo — ¿está en mínimo del slider?")
        if config.phi_frac < 0.01:
            _phys_warnings.append(f"⚠ φ = {config.phi_frac:.4f} parece muy bajo — revisa la porosidad.")
        if config.h_ft < 1.0:
            _phys_warnings.append(f"⚠ h = {config.h_ft:.2f} ft parece muy bajo — revisa el espesor neto.")
        if config.Bo_rb_stb < 0.8:
            _phys_warnings.append(f"⚠ Bo = {config.Bo_rb_stb:.3f} RB/STB parece muy bajo.")
        if config.re_ft is None and config.area_acres is None:
            _phys_warnings.append("⚠ re y área no definidos — N volumétrico no se puede calcular.")
        if _phys_warnings:
            st.warning("\n\n".join(_phys_warnings))

        save_col, reset_col, status_col = st.columns([1, 1, 3])
        with save_col:
            if st.button("Guardar escenario", type="primary"):
                try:
                    saved_path = save_rta_scenario(config, output_dir=OUTPUT_DIR)
                    st.session_state["rta_scenario_saved"] = str(saved_path)
                except Exception as exc:
                    st.error(f"No se pudo guardar: {exc}")

        with reset_col:
            if st.button("🔄 Reset", help="Recarga el escenario desde disco y resetea los sliders."):
                # Clear all rta_* widget keys so _init_reservoir_config_state re-seeds them
                _keys_to_clear = [k for k in st.session_state if k.startswith("rta_")]
                for _k in _keys_to_clear:
                    del st.session_state[_k]
                # Also clear the scenario-loaded guard so init runs again
                _guard = f"rta_scenario_loaded_{config.well_id}"
                st.session_state.pop(_guard, None)
                st.session_state.pop("rta_scenario_saved", None)
                st.rerun()

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


def _generate_overlay_png(
    type_curves: list[Any],
    raw_points: list[RTAOverlayPoint],
    x_multiplier: float,
    y_multiplier: float,
    method_label: str,
    selected_curve_id: str | None = None,
    auxiliary_series: list[tuple[str, list[RTAOverlayPoint]]] | None = None,
) -> bytes:
    """Generate PNG bytes of the type-curve overlay (used for download buttons)."""
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

    # --- Auxiliary series (qDdi, qDdid, log-log derivative) ---
    if auxiliary_series:
        _aux_colors = ["#4ade80", "#f472b6", "#facc15"]   # green, pink, yellow
        _aux_markers = ["s", "^", "D"]
        for idx, (series_label, series_pts) in enumerate(auxiliary_series):
            if not series_pts:
                continue
            ax_x = [p.x * x_multiplier for p in series_pts]
            ax_y = [p.y * y_multiplier for p in series_pts]
            ax.loglog(
                ax_x, ax_y,
                linestyle="", marker=_aux_markers[idx % len(_aux_markers)],
                color=_aux_colors[idx % len(_aux_colors)],
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

    # Clamp y-axis to [1e-4, 200] — prevents the exponential BDF tail (b=0 → y≈1e-9)
    # from collapsing the visible range and making all curves look flat.
    _ymin, _ymax = ax.get_ylim()
    ax.set_ylim(bottom=max(_ymin, 1e-4), top=min(_ymax, 200.0))

    try:
        fig.tight_layout()
    except Exception:
        pass

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    png_bytes = buf.getvalue()

    plt.close(fig)
    return png_bytes


def _plot_all_curves_plotly(
    type_curves: list[Any],
    raw_points: list[RTAOverlayPoint],
    x_multiplier: float,
    y_multiplier: float,
    method_label: str,
    selected_curve_id: str | None = None,
    auxiliary_series: list[tuple[str, list[RTAOverlayPoint]]] | None = None,
) -> go.Figure:
    """Build an interactive Plotly log-log figure for the overlay (zoom/pan native)."""
    fig = go.Figure()

    bdf_curves = [c for c in type_curves if c.curve_family != "transient_stem"]
    transient_curves = [c for c in type_curves if c.curve_family == "transient_stem"]

    # --- BDF stems (solid, tab10 colors) ---
    for i, curve in enumerate(bdf_curves):
        xs = [p.x for p in curve.points]
        ys = [p.y for p in curve.points]
        is_selected = curve.curve_id == selected_curve_id
        color = _TAB10_HEX[i % len(_TAB10_HEX)]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            name=curve.curve_id,
            line=dict(color=color, width=2.5 if is_selected else 1.2, dash="solid"),
            opacity=1.0 if is_selected else 0.75,
            legendgroup="bdf",
        ))

    # --- Transient stems (dashed, autumn palette) ---
    n_trans = len(transient_curves)
    for i, curve in enumerate(transient_curves):
        xs = [p.x for p in curve.points]
        ys = [p.y for p in curve.points]
        frac = 0.15 + 0.7 * (i / max(n_trans - 1, 1))
        g = int(255 * frac)
        color = f"rgba(255, {g}, 0, 0.65)"
        re_rw_label = (
            curve.curve_id.split("re_rw_")[-1]
            if "re_rw_" in curve.curve_id else curve.curve_id
        )
        show_leg = i == 0 or i == n_trans - 1
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            name=f"re/rw={re_rw_label}" if show_leg else "_",
            line=dict(color=color, width=0.9, dash="dash"),
            legendgroup="transient",
            showlegend=show_leg,
        ))

    # --- Raw data points ---
    if raw_points:
        raw_x = [p.x for p in raw_points]
        raw_y = [p.y for p in raw_points]
        fig.add_trace(go.Scatter(
            x=raw_x, y=raw_y,
            mode="markers",
            name="Datos raw",
            marker=dict(color="#f97316", size=5, opacity=0.55),
        ))
        matched_x = [v * x_multiplier for v in raw_x]
        matched_y = [v * y_multiplier for v in raw_y]
        fig.add_trace(go.Scatter(
            x=matched_x, y=matched_y,
            mode="markers",
            name="Datos ajustados",
            marker=dict(color="#22d3ee", size=5, opacity=0.85),
        ))

    # --- Auxiliary series (Blasingame qDdi/qDdid + log-log diagnostic) ---
    if auxiliary_series:
        _aux_colors  = ["#4ade80", "#f472b6", "#facc15"]   # green, pink, yellow
        _aux_symbols = ["square", "triangle-up", "diamond"]
        for idx, (series_label, series_pts) in enumerate(auxiliary_series):
            if not series_pts:
                continue
            ax_x = [p.x * x_multiplier for p in series_pts]
            ax_y = [p.y * y_multiplier for p in series_pts]
            fig.add_trace(go.Scatter(
                x=ax_x, y=ax_y,
                mode="markers",
                name=series_label,
                marker=dict(
                    color=_aux_colors[idx % len(_aux_colors)],
                    size=5,
                    symbol=_aux_symbols[idx % len(_aux_symbols)],
                    opacity=0.75,
                ),
            ))

    x_lbl = type_curves[0].x_label if type_curves else "tDd"
    y_lbl = type_curves[0].y_label if type_curves else "qDd"

    fig.update_layout(
        title=dict(
            text=f"{method_label} — familia completa de curvas tipo",
            font=dict(size=13),
        ),
        xaxis=dict(
            title=x_lbl,
            type="log",
            showgrid=True,
            gridcolor="rgba(180,180,180,0.25)",
            minor=dict(showgrid=True, gridcolor="rgba(180,180,180,0.12)"),
            showline=True,
            linecolor="#aaa",
        ),
        yaxis=dict(
            title=y_lbl,
            type="log",
            range=[math.log10(1e-4), math.log10(200.0)],
            showgrid=True,
            gridcolor="rgba(180,180,180,0.25)",
            minor=dict(showgrid=True, gridcolor="rgba(180,180,180,0.12)"),
            showline=True,
            linecolor="#aaa",
        ),
        legend=dict(
            font=dict(size=8),
            orientation="v",
            x=1.01, y=1.0,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
        height=520,
        margin=dict(l=60, r=20, t=45, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
    )

    return fig


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


def _find_best_bdf_stem(
    bdf_curves: list[Any],
    raw_points: list[RTAOverlayPoint],
    x_multiplier: float,
    y_multiplier: float,
) -> str | None:
    """Return the BDF curve_id whose shape is closest to the shifted data.

    Uses mean of minimum log-log distances: for each shifted data point finds
    the nearest point on the curve (in log10 space), then averages those
    distances across all data points. The curve with the smallest average wins.
    """
    if not bdf_curves or not raw_points:
        return None

    shifted_log = [
        (math.log10(p.x * x_multiplier), math.log10(p.y * y_multiplier))
        for p in raw_points
        if p.x > 0 and p.y > 0
        and p.x * x_multiplier > 0 and p.y * y_multiplier > 0
    ]
    if not shifted_log:
        return None

    best_id: str | None = None
    best_score = math.inf

    for curve in bdf_curves:
        curve_log = [
            (math.log10(pt.x), math.log10(pt.y))
            for pt in curve.points
            if pt.x > 0 and pt.y > 0
        ]
        if not curve_log:
            continue

        total = sum(
            min(math.hypot(cx - dx, cy - dy) for cx, cy in curve_log)
            for dx, dy in shifted_log
        )
        score = total / len(shifted_log)
        if score < best_score:
            best_score = score
            best_id = curve.curve_id

    return best_id


# 7 sensitivity steps: 1=MIN (finest) … 7=MAX (coarsest)
# Each value is log10-decades per click → step_factor = 10^decades
_SENSITIVITY_STEPS: list[tuple[str, float]] = [
    ("1 MIN",  0.005),  # ×1.012 / click — ajuste ultrafino
    ("2",      0.01),   # ×1.023 / click
    ("3",      0.02),   # ×1.047 / click — antes "Fino"
    ("4",      0.05),   # ×1.122 / click
    ("5",      0.1),    # ×1.259 / click — antes "Medio" (default)
    ("6",      0.3),    # ×2.000 / click — antes "Grueso"
    ("7 MAX",  0.5),    # ×3.162 / click — posicionamiento rápido
]
_SENSITIVITY_LABELS = [s[0] for s in _SENSITIVITY_STEPS]
_SENSITIVITY_MAP:  dict[str, float] = {s[0]: s[1] for s in _SENSITIVITY_STEPS}
_SENSITIVITY_DESC: dict[str, str] = {
    s[0]: f"×{10**s[1]:.3f}/click" for s in _SENSITIVITY_STEPS
}
_SENSITIVITY_DEFAULT = "5"   # equivalent of old "Medio"

# Keep legacy dict for backward-compat with any remaining references
_SENSITIVITY_LEVELS: dict[str, float] = {s[0]: s[1] for s in _SENSITIVITY_STEPS}

# ---------------------------------------------------------------------------
# SNES controller custom component
# ---------------------------------------------------------------------------

_SNES_COMPONENT_PATH = PROJECT_ROOT / "src" / "ui" / "components" / "snes_controller"
_SNES_CONTROLLER_IMG = PROJECT_ROOT / "assets" / "snes_controller.png"

# Declare the bidirectional component (cached at module load)
_snes_ctrl_component = _st_components.declare_component(
    "snes_controller",
    path=str(_SNES_COMPONENT_PATH),
)


def _load_ctrl_b64() -> str:
    """Return the SNES controller image as a base-64 string (cached in session state)."""
    _cache_key = "_snes_ctrl_b64"
    if _cache_key not in st.session_state:
        if _SNES_CONTROLLER_IMG.exists():
            st.session_state[_cache_key] = base64.b64encode(
                _SNES_CONTROLLER_IMG.read_bytes()
            ).decode()
        else:
            st.session_state[_cache_key] = ""
    return st.session_state[_cache_key]


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


@st.cache_data
def _load_history_cached(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data
def _compute_rta_transforms_cached(
    history_df: pd.DataFrame,
    pi_psia: float,
) -> list:
    return compute_rta_transforms(dataframe=history_df, pi_psia=pi_psia)


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

    # Reservoir config — pass hub well_id so it stays locked to the correct ID
    reservoir_config = _render_reservoir_config(hub_well_id=well_id)

    # Auto-load history ONLY from output/{well_id}_history_enriched.csv
    _auto_history_path = output_dir / f"{well_id}_history_enriched.csv"
    if not _auto_history_path.exists():
        st.info(
            "No hay historia enriquecida disponible. "
            "Ejecuta primero el **Paso 4 (M1-M2)** para generarla."
        )
        return

    try:
        history_df = _load_history_cached(str(_auto_history_path))
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
        rta_transform_points = _compute_rta_transforms_cached(
            history_df,
            reservoir_config.pi_psia,
        )
    except Exception as exc:
        st.error(f"Error calculando variables RTA: {exc}")
        return

    if not rta_transform_points:
        st.warning("Sin puntos RTA válidos (verifica Pi y datos de producción).")
        return

    # ---- Method tabs ----
    _TAB_LABELS  = ["🔬 Fetkovich", "📊 Palacio-Blasingame", "📈 Agarwal-Gardner"]
    _TAB_METHODS = [_RTAMethod.FETKOVICH, _RTAMethod.BLASINGAME, _RTAMethod.AGARWAL_GARDNER]
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

            for _k, _v in [
                (_xk, 1.0), (_yk, 1.0),
                (_slk, _SENSITIVITY_DEFAULT), (_snk, _SENSITIVITY_MAP[_SENSITIVITY_DEFAULT]),
                # Checkbox series — persistidas explícitamente para sobrevivir st.rerun()
                (f"bl_qdd_{_mval}", True), (f"bl_qddi_{_mval}", True), (f"bl_qddid_{_mval}", True),
            ]:
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
                    st.session_state[f"match_sensitivity_label_{_mv}"] = _SENSITIVITY_DEFAULT
                    st.session_state[f"match_sensitivity_decades_{_mv}"] = _SENSITIVITY_MAP[_SENSITIVITY_DEFAULT]
                return _left, _right, _up, _down, _reset

            _cb_left, _cb_right, _cb_up, _cb_down, _cb_reset = _make_cbs(_mval, _xk, _yk, _snk)

            # Curves for this method
            all_curves = registry.get_by_method(method)
            if not all_curves:
                st.error(f"Sin curvas para {_mval}.")
                continue

            # Blasingame uses composite family; other methods use radial/arps BDF families
            if method == _RTAMethod.BLASINGAME:
                _bdf_curves = [c for c in all_curves if c.y_label == "qDd"]
            else:
                _bdf_families = ("arps_bdf", "radial_bdf")
                _bdf_curves   = [c for c in all_curves if c.curve_family in _bdf_families]
            bdf_ids = [c.curve_id for c in _bdf_curves]

            # Build a tab10 color map for BDF stems (matches _generate_overlay_png color order)
            import matplotlib as _mpl
            _tab10_colors = [_mpl.colormaps["tab10"](i) for i in range(len(_bdf_curves))]
            _bdf_color_hex = {
                cid: "#{:02x}{:02x}{:02x}".format(
                    int(r * 255), int(g * 255), int(b * 255)
                )
                for cid, (r, g, b, _) in zip(bdf_ids, _tab10_colors)
            }

            _n_trans = len(all_curves) - len(bdf_ids)
            st.caption(
                f"{len(all_curves)} curvas — {len(bdf_ids)} BDF + {_n_trans} transientes"
            )

            # Read multipliers here so the Auto button can use them
            _x_eff = _clamp_multiplier(st.session_state[_xk])
            _y_eff = _clamp_multiplier(st.session_state[_yk])

            # Selectbox + 🎯 Auto-selección del mejor stem
            # Apply pending auto-selection BEFORE instantiating the widget
            _auto_pending_key = f"auto_pending_{_mval}"
            if _auto_pending_key in st.session_state:
                _pending_id = st.session_state.pop(_auto_pending_key)
                if _pending_id and _pending_id in bdf_ids:
                    st.session_state[f"ref_curve_{_mval}"] = _pending_id

            _sel_col, _auto_col = st.columns([3, 1])
            with _sel_col:
                ref_id = st.selectbox(
                    "Curva BDF de referencia (para kh/k)",
                    options=bdf_ids,
                    key=f"ref_curve_{_mval}",
                )
            with _auto_col:
                # Blasingame tab reuses PB transform points (same physics/variables)
                _auto_method = _RTAMethod.PALACIO_BLASINGAME if method == _RTAMethod.BLASINGAME else method
                _method_pts_auto = [p for p in rta_transform_points if p.method == _auto_method]
                _rta_pts_auto = _transform_points_to_overlay(_method_pts_auto)
                st.write("")  # alineación vertical con selectbox
                if _rta_pts_auto and st.button(
                    "🎯 Auto",
                    key=f"auto_stem_{_mval}",
                    help=(
                        "Selecciona automáticamente la curva BDF cuya forma "
                        "minimiza la distancia log-log media a los datos ajustados."
                    ),
                    use_container_width=True,
                ):
                    _best_id = _find_best_bdf_stem(
                        _bdf_curves, _rta_pts_auto, _x_eff, _y_eff
                    )
                    if _best_id and _best_id in bdf_ids:
                        st.session_state[_auto_pending_key] = _best_id
                        st.rerun()

            # Color legend for BDF stems — shows which color corresponds to each curve
            if _bdf_color_hex:
                _legend_html = "<div style='display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 8px 0'>"
                for _cid, _hex in _bdf_color_hex.items():
                    _is_sel = _cid == ref_id
                    _border = "2px solid #fff" if _is_sel else "none"
                    _opacity = "1.0" if _is_sel else "0.65"
                    _legend_html += (
                        f"<span style='display:inline-flex;align-items:center;gap:3px;"
                        f"opacity:{_opacity}'>"
                        f"<span style='width:12px;height:12px;border-radius:2px;"
                        f"background:{_hex};outline:{_border};flex-shrink:0'></span>"
                        f"<span style='font-size:0.72rem;color:#cbd5e1'>{_cid}</span>"
                        f"</span>"
                    )
                _legend_html += "</div>"
                st.markdown(_legend_html, unsafe_allow_html=True)

            # Compute match params before columns so both can reference _mp
            _mp = None
            _mp_error: str | None = None

            def _fmt(v: float | None, d: int = 2) -> str:
                return f"{v:.{d}f}" if v is not None else "—"

            try:
                _mp = compute_match_params(
                    config=reservoir_config,
                    effective_x_multiplier=_x_eff,
                    effective_y_multiplier=_y_eff,
                    method=_mval,
                )
            except Exception as _mp_exc:
                _mp_error = str(_mp_exc)

            # ── Blasingame composite: checkboxes de serie ANTES de las columnas ──────
            # Deben renderizarse antes del with _joy_col: para que st.rerun() del
            # joystick no los encuentre "no registrados" y los resetee al default.
            if method == _RTAMethod.BLASINGAME:
                _ck1, _ck2, _ck3 = st.columns(3)
                with _ck1:
                    _show_qdd = st.checkbox(
                        "qDd", value=st.session_state[f"bl_qdd_{_mval}"],
                        key=f"bl_qdd_{_mval}",
                        help="Serie qDd — curvas tipo y nube de puntos",
                    )
                with _ck2:
                    _show_qddi = st.checkbox(
                        "qDdi", value=st.session_state[f"bl_qddi_{_mval}"],
                        key=f"bl_qddi_{_mval}",
                        help="Serie qDdi — curvas tipo y nube de puntos",
                    )
                with _ck3:
                    _show_qddid = st.checkbox(
                        "qDdid", value=st.session_state[f"bl_qddid_{_mval}"],
                        key=f"bl_qddid_{_mval}",
                        help="Serie qDdid — curvas tipo y nube de puntos",
                    )
                _sel_ylabels: list[str] = []
                if _show_qdd:   _sel_ylabels.append("qDd")
                if _show_qddi:  _sel_ylabels.append("qDdi")
                if _show_qddid: _sel_ylabels.append("qDdid")
                display_curves = [c for c in all_curves if c.y_label in _sel_ylabels] or all_curves
            else:
                _show_qdd = _show_qddi = _show_qddid = True
                display_curves = all_curves

            # QC warnings — fuera de las columnas para que joystick y gráfica queden alineados
            _chart_method_qc = _RTAMethod.PALACIO_BLASINGAME if method == _RTAMethod.BLASINGAME else method
            _method_pts_qc = [p for p in rta_transform_points if p.method == _chart_method_qc]
            if _method_pts_qc:
                _qc_results_pre = run_rta_qc(
                    points=_method_pts_qc,
                    effective_x_multiplier=_x_eff,
                    effective_y_multiplier=_y_eff,
                )
                _qc_level_pre = qc_severity_level(_qc_results_pre)
                if _qc_level_pre == "error":
                    for _r in _qc_results_pre:
                        if _r.severity == "error":
                            st.error(f"**{_r.title}**  \n{_r.detail}", icon="🔴")
                elif _qc_level_pre == "warning":
                    for _r in _qc_results_pre:
                        if _r.severity == "warning":
                            st.warning(f"**{_r.title}**  \n{_r.detail}", icon="⚠️")

            # 2-column layout: chart (wide) | joystick (narrow)
            _chart_col, _joy_col = st.columns([3, 1.2])

            # ---- SNES Controller column ----
            with _joy_col:
                # Current sensitivity step (1-based index into _SENSITIVITY_LABELS)
                _cur_lbl = st.session_state.get(_slk, _SENSITIVITY_DEFAULT)
                try:
                    _cur_step_num = _SENSITIVITY_LABELS.index(_cur_lbl) + 1
                except ValueError:
                    _cur_step_num = int(_SENSITIVITY_DEFAULT)

                # ── Render the SNES controller component ──────────────────
                _snes_result = _snes_ctrl_component(
                    image_b64=_load_ctrl_b64(),
                    sens_step=_cur_step_num,
                    total_steps=len(_SENSITIVITY_LABELS),
                    key=f"snes_{_mval}",
                )

                # ── Process action (dedup by seq number) ──────────────────
                _snes_seq_key = f"_snes_last_seq_{_mval}"
                if _snes_result and isinstance(_snes_result, dict):
                    _seq_val = _snes_result.get("seq", 0)
                    if _seq_val != st.session_state.get(_snes_seq_key, -1):
                        st.session_state[_snes_seq_key] = _seq_val
                        _act = _snes_result.get("action", "")
                        if   _act == "up":    _cb_up()
                        elif _act == "down":  _cb_down()
                        elif _act == "left":  _cb_left()
                        elif _act == "right": _cb_right()
                        elif _act == "reset": _cb_reset()
                        elif _act in ("sens_inc", "sens_dec"):
                            try:
                                _sidx = _SENSITIVITY_LABELS.index(_cur_lbl)
                            except ValueError:
                                _sidx = int(_SENSITIVITY_DEFAULT) - 1
                            _new_sidx = (
                                min(_sidx + 1, len(_SENSITIVITY_LABELS) - 1)
                                if _act == "sens_inc" else max(_sidx - 1, 0)
                            )
                            _new_lbl = _SENSITIVITY_LABELS[_new_sidx]
                            st.session_state[_slk] = _new_lbl
                            st.session_state[_snk] = _SENSITIVITY_MAP[_new_lbl]
                        elif _act == "auto":
                            # SNES AUTO button — same logic as the 🎯 Auto button in UI
                            _best_id = _find_best_bdf_stem(
                                _bdf_curves, _rta_pts_auto, _x_eff, _y_eff
                            )
                            if _best_id and _best_id in bdf_ids:
                                st.session_state[_auto_pending_key] = _best_id
                        elif _act == "save":
                            # Flag for processing AFTER _mp is computed
                            st.session_state[f"_pending_save_{_mval}"] = True
                        st.rerun()

                # ── Current position caption ──────────────────────────────
                st.caption(
                    f"X: {_x_eff:.4g} · Y: {_y_eff:.4g} · S{_cur_step_num} "
                    f"({_SENSITIVITY_DESC.get(_cur_lbl, '')})"
                )

                # ── Handle SAVE flagged by SNES controller ─────────────────
                if st.session_state.pop(f"_pending_save_{_mval}", False):
                    if _mp is not None:
                        st.session_state[f"_saved_match_{_mval}"] = {
                            "method":       _mval,
                            "ref_curve_id": ref_id,
                            "kh_md_ft":     _mp.kh_md_ft,
                            "k_md":         _mp.k_md,
                            "n_vol_stb":    _mp.n_vol_stb,
                            "re_ft":        _mp.re_ft,
                            "area_acres":   _mp.area_acres,
                            "x_mult":       _x_eff,
                            "y_mult":       _y_eff,
                            "warnings":     len(_mp.warnings),
                        }
                        st.success("Match guardado ✅")
                    else:
                        st.warning("Sin parámetros de match — mueve el joystick primero.")

            # ---- Chart column ----
            with _chart_col:
                _chart_method = _RTAMethod.PALACIO_BLASINGAME if method == _RTAMethod.BLASINGAME else method

                st.subheader("Overlay log-log")

                try:
                    _method_pts_chart = [p for p in rta_transform_points if p.method == _chart_method]
                    if not _method_pts_chart:
                        st.warning(f"Sin puntos RTA para {_mval}.")
                    else:
                        _rta_pts = _transform_points_to_overlay(_method_pts_chart)

                        _auxiliary: list[tuple[str, list[RTAOverlayPoint]]] = []

                        # Blasingame composite scatter — usa checkboxes unificados de arriba
                        if method == _RTAMethod.BLASINGAME:
                            _pb_src = [
                                p for p in rta_transform_points
                                if p.method.value == "palacio_blasingame"
                            ]
                            if _show_qddi:
                                _int_pts = [
                                    RTAOverlayPoint(
                                        x=p.material_balance_time,
                                        y=p.blasingame_integral,
                                        label=p.well_id,
                                        date=p.date,
                                    )
                                    for p in _pb_src if p.blasingame_integral is not None
                                ]
                                if _int_pts:
                                    _auxiliary.append(("qDdi (integral norm.)", _int_pts))
                            if _show_qddid:
                                _drv_pts = [
                                    RTAOverlayPoint(
                                        x=p.material_balance_time,
                                        y=p.blasingame_derivative,
                                        label=p.well_id,
                                        date=p.date,
                                    )
                                    for p in _pb_src if p.blasingame_derivative is not None
                                ]
                                if _drv_pts:
                                    _auxiliary.append(("qDdid (deriv. integral)", _drv_pts))
                            if not _show_qdd:
                                _rta_pts = []

                        # Log-log diagnostic derivative — opt-in, all three methods
                        _show_log_deriv = st.checkbox(
                            "Derivada log-log (diagnóstico de flujo)",
                            value=False,
                            key=f"show_log_deriv_{_mval}",
                            help=(
                                "Muestra -d(ln(q/Δp))/d(ln(MBT)) × (q/Δp) en las mismas unidades.\n"
                                "Pendiente ≈ -1 → BDF · -0.5 → flujo lineal · -0.25 → biflujo"
                            ),
                        )
                        if _show_log_deriv:
                            _ld_pts = []
                            for p in _method_pts_chart:
                                _ld = getattr(p, "log_derivative", None)
                                if _ld is not None and p.normalized_rate * _ld > 0:
                                    _ld_pts.append(RTAOverlayPoint(
                                        x=p.material_balance_time,
                                        y=p.normalized_rate * _ld,
                                        label=p.well_id,
                                        date=p.date,
                                    ))
                            if _ld_pts:
                                _auxiliary.append(("Deriv. log-log (diag.)", _ld_pts))

                        _match_cfg = ManualMatchConfig(
                            x_multiplier=_x_eff,
                            y_multiplier=_y_eff,
                        )
                        _xm = _match_cfg.effective_x_multiplier
                        _ym = _match_cfg.effective_y_multiplier

                        # Interactive Plotly chart (zoom/pan/hover nativo)
                        _fig = _plot_all_curves_plotly(
                            type_curves=display_curves,
                            raw_points=_rta_pts,
                            x_multiplier=_xm,
                            y_multiplier=_ym,
                            method_label=_mval,
                            selected_curve_id=ref_id,
                            auxiliary_series=_auxiliary or None,
                        )
                        # uirevision stable key → Plotly preserves zoom/pan on rerun
                        _fig.update_layout(uirevision=f"rta_overlay_{_mval}")
                        st.plotly_chart(_fig, use_container_width=True, key=f"plotly_{_mval}")

                        # PNG generado sólo para botones de descarga
                        _png = _generate_overlay_png(
                            type_curves=display_curves,
                            raw_points=_rta_pts,
                            x_multiplier=_xm,
                            y_multiplier=_ym,
                            method_label=_mval,
                            selected_curve_id=ref_id,
                            auxiliary_series=_auxiliary or None,
                        )
                        st.caption(
                            f"{len(_method_pts_chart)} puntos  |  "
                            f"X: {_x_eff:.4g}  |  Y: {_y_eff:.4g}  |  "
                            f"Curva: {ref_id}"
                        )

                        # Export row (under chart)
                        if _mp is not None:
                            _e1, _e2, _e3 = st.columns(3)
                            try:
                                _ref_curve_obj = registry.get(_mval, ref_id)
                                _curve_status = _ref_curve_obj.status.value
                            except Exception:
                                _curve_status = "demo"
                            _summary = build_match_summary(
                                match_params=_mp,
                                ref_curve_id=ref_id,
                                config=reservoir_config,
                                curve_status=_curve_status,
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
                                    "💾 Guardar para M5",
                                    use_container_width=True,
                                    key=f"save_disk_{_mval}",
                                    help="Guarda el match a disco — requerido para que M5 muestre los resultados RTA.",
                                ):
                                    try:
                                        _saved = save_match_summary(_summary, output_dir=output_dir)
                                        save_overlay_png(
                                            _png, reservoir_config.well_id,
                                            output_dir=output_dir,
                                        )
                                        # Also persist the config so next session loads correct params
                                        save_rta_scenario(reservoir_config, output_dir=output_dir)
                                        st.session_state[f"_m4_saved_to_disk_{_mval}"] = str(_saved.name)
                                        st.rerun()
                                    except Exception as exc:
                                        st.error(str(exc))
                                _disk_saved = st.session_state.get(f"_m4_saved_to_disk_{_mval}")
                                if _disk_saved:
                                    st.success(f"✅ `{_disk_saved}` listo para M5")

                except Exception as exc:
                    st.error(f"Error construyendo overlay: {exc}")

            # ── Métricas de match — fila horizontal debajo del overlay ──────
            if _mp_error:
                st.error(f"Error calculando parámetros: {_mp_error}")
            elif _mp is not None:
                _mc1, _mc2, _mc3, _mc4, _mc5, _mc6 = st.columns(6)
                _mc1.metric("kh (mD·ft)", _fmt(_mp.kh_md_ft, 1))
                _mc2.metric("k (mD)", _fmt(_mp.k_md, 3))

                # N vol.: volumétrico fijo (φ, h, A, Swi, Bo)
                def _n_str_fmt(n_stb: float | None) -> str:
                    if n_stb is not None and n_stb > 0:
                        _mm = n_stb / 1e6
                        return f"{_mm:.3f}" if _mm >= 0.0005 else f"~{n_stb:.0f} STB"
                    return "—"

                _mc3.metric(
                    "N vol. (MM STB)",
                    _n_str_fmt(_mp.n_vol_stb),
                    help=(
                        "OOIP volumétrico: φ·h·A·(1-Swi)/(5.615·Bo)\n"
                        f"φ={reservoir_config.phi_frac:.3f} · h={reservoir_config.h_ft:.1f} ft · "
                        f"A={_fmt(_mp.area_acres, 1)} acres · Swi={reservoir_config.swi_frac or 0:.2f} · "
                        f"Bo={reservoir_config.Bo_rb_stb:.3f} RB/STB\n"
                        "Fijo — no cambia con el joystick."
                    ),
                )

                # N match: dinámico desde posición del joystick (x_mult + y_mult)
                _mc4.metric(
                    "N match (MM STB)",
                    _n_str_fmt(getattr(_mp, "n_dyn_stb", None)),
                    help=(
                        "OOIP dinámico del match: C·(1-Swi)·kh / (Bo·μ·ct·x_mult·ln_term)\n"
                        "Actualiza con cada click del joystick.\n"
                        "Cuando N match ≈ N vol. → match consistente con la geometría."
                    ),
                )

                # Compute dynamic area/re directly in UI from n_dyn_stb
                # (avoids any stale-module issue with RTAMatchParams dataclass fields)
                _n_dyn_val = getattr(_mp, "n_dyn_stb", None)
                _re_dyn: float | None = None
                _a_dyn: float | None = None
                if _n_dyn_val is not None and _n_dyn_val > 0:
                    _swi_ui = reservoir_config.swi_frac or 0.0
                    _phi_ui = reservoir_config.phi_frac
                    _h_ui   = reservoir_config.h_ft
                    _bo_ui  = reservoir_config.Bo_rb_stb
                    _denom_ui = _phi_ui * _h_ui * (1.0 - _swi_ui)
                    if _denom_ui > 0 and _bo_ui > 0:
                        _a_ft2_ui = _n_dyn_val * 5.615 * _bo_ui / _denom_ui
                        if _a_ft2_ui > 0:
                            _a_dyn  = _a_ft2_ui / _FT2_PER_ACRE
                            _re_dyn = math.sqrt(_a_ft2_ui / math.pi)

                _both_moved = (_x_eff != 1.0 and _y_eff != 1.0)
                _mc5.metric(
                    "re match (ft)",
                    _fmt(_re_dyn, 0) if _re_dyn is not None else "—",
                    help=(
                        "Radio de drene dinámico del match.\n"
                        "Actualiza cuando ambos ejes del joystick (X e Y) difieren de 1.0.\n"
                        f"Estado: {'✅ listo' if _both_moved else '⏳ mueve X e Y para activar'}"
                    ),
                )
                _mc6.metric(
                    "Área match (acres)",
                    _fmt(_a_dyn, 1) if _a_dyn is not None else "—",
                    help=(
                        "Área de drene dinámica del match.\n"
                        "Actualiza cuando ambos ejes del joystick (X e Y) difieren de 1.0.\n"
                        f"Estado: {'✅ listo' if _both_moved else '⏳ mueve X e Y para activar'}"
                    ),
                )
                if _mp.warnings:
                    for _w in _mp.warnings:
                        st.warning(_w, icon="⚠")

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