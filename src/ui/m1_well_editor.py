"""M1 — Editor interactivo de estado mecánico del pozo.

Permite al usuario ingresar los diámetros y profundidades de los
revestimientos, tubing, ESP y perforaciones, y muestra en tiempo real:

  1. Esquema mecánico (cross-section log-log style)
  2. QC técnico de compatibilidad dimensional
  3. Estimación de Pwf (hidrostática + Darcy-Weisbach)

Run from project root:
    python -m streamlit run src/ui/m1_well_editor.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import streamlit as st

from src.services.well_mech_qc_service import (
    CasingString,
    MechQCResult,
    TubingString,
    WellMechConfig,
    mech_severity_level,
    run_mech_qc,
)
from src.well_mod.pwf import PwfInputs, estimate_pwf_v1, estimate_pwf_v2
from src.well_mod.schematic import draw_well_schematic, schematic_to_png_bytes

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ecoRTA M1 — Estado Mecánico",
    page_icon="🛢",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEVERITY_ICON = {"ok": "✅", "warning": "⚠️", "error": "🔴"}
_SEVERITY_FN   = {"ok": st.success, "warning": st.warning, "error": st.error}
_SEVERITY_KW   = {"ok": "✅", "warning": "⚠️", "error": "🔴"}

# Typical nominal tubing sizes [OD_in, ID_in]
_TUBING_SIZES = {
    "1.900\" (1.610\" ID)": (1.900, 1.610),
    "2-3/8\" (1.995\" ID)": (2.375, 1.995),
    "2-7/8\" (2.441\" ID)": (2.875, 2.441),
    "3-1/2\" (2.992\" ID)": (3.500, 2.992),
    "4\" (3.476\" ID)":     (4.000, 3.476),
    "4-1/2\" (3.958\" ID)": (4.500, 3.958),
    "Personalizado":         None,
}

# Typical nominal casing sizes [OD_in, ID_in] (N-80 / API)
_CASING_SIZES = {
    '4-1/2" (3.958" ID)':  (4.500, 3.958),
    '5" (4.276" ID)':      (5.000, 4.276),
    '5-1/2" (4.892" ID)':  (5.500, 4.892),
    '7" (6.276" ID)':      (7.000, 6.276),
    '7-5/8" (6.765" ID)':  (7.625, 6.765),
    '9-5/8" (8.681" ID)':  (9.625, 8.681),
    '10-3/4" (9.760" ID)': (10.750, 9.760),
    '13-3/8" (12.415" ID)':(13.375, 12.415),
    '20" (18.730" ID)':    (20.000, 18.730),
    'Personalizado':        None,
}

_DEFAULT_CASING_NAMES = ["Superficie", "Intermedio", "Producción"]
_DEFAULT_CASING_PRESETS = [
    ('13-3/8" (12.415" ID)', 2000.0),
    ('9-5/8" (8.681" ID)',   8500.0),
]
_DEFAULT_TUBING_PRESET  = '2-7/8" (2.441" ID)'


def _casing_inputs(idx: int) -> CasingString:
    """Render inputs for one casing string and return a CasingString."""
    default_name = _DEFAULT_CASING_NAMES[idx] if idx < 3 else f"Revestimiento {idx + 1}"
    default_preset, default_shoe = (
        _DEFAULT_CASING_PRESETS[idx] if idx < len(_DEFAULT_CASING_PRESETS)
        else ('9-5/8" (8.681" ID)', 9000.0 + idx * 1000)
    )

    name = st.text_input("Nombre", value=default_name, key=f"c_name_{idx}")
    preset = st.selectbox(
        "Tamaño nominal", options=list(_CASING_SIZES.keys()),
        index=list(_CASING_SIZES.keys()).index(default_preset),
        key=f"c_preset_{idx}",
    )
    preset_vals = _CASING_SIZES[preset]

    col_od, col_id = st.columns(2)
    od = col_od.number_input(
        "OD (in)", min_value=1.0, max_value=30.0,
        value=float(preset_vals[0]) if preset_vals else 9.625,
        step=0.001, format="%.3f", key=f"c_od_{idx}",
    )
    id_ = col_id.number_input(
        "ID / drift (in)", min_value=0.5, max_value=29.0,
        value=float(preset_vals[1]) if preset_vals else 8.681,
        step=0.001, format="%.3f", key=f"c_id_{idx}",
    )
    shoe = st.number_input(
        "Zapato MD (ft)", min_value=50.0, max_value=25_000.0,
        value=float(default_shoe),
        step=10.0, format="%.0f", key=f"c_shoe_{idx}",
    )
    return CasingString(name=name, od_in=od, id_in=id_, shoe_depth_ft=shoe)


# ---------------------------------------------------------------------------
# Embedded renderer (for use from the hub app.py)
# ---------------------------------------------------------------------------

def render_m1_editor_embedded(well_id: str) -> None:
    """Render the M1 well-mechanic editor inside the hub (no page config, no title).

    Displays casing / tubing / ESP / perforations inputs on the left and the
    well schematic + mechanical QC on the right.  A *Save* button writes the
    relevant geometry fields to ``output/{well_id}_well_geometry.json`` so the
    hub's Pwf pipeline can pick them up.
    """
    import json

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    geo_path = output_dir / f"{well_id}_well_geometry.json"

    # Load existing geometry for pre-filling defaults
    _existing_geo: dict = {}
    if geo_path.exists():
        try:
            _existing_geo = json.loads(geo_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    left_col, right_col = st.columns([0.40, 0.60], gap="large")

    # =========================================================================
    # LEFT COLUMN — inputs
    # =========================================================================
    with left_col:
        st.markdown("#### 🔩 Revestimientos")
        n_casings = st.number_input(
            "Número de revestimientos",
            min_value=1, max_value=3, value=2, step=1,
            key="m1e_n_casings",
        )
        casings: list[CasingString] = []
        for _ci in range(int(n_casings)):
            _default_name = _DEFAULT_CASING_NAMES[_ci] if _ci < 3 else f"Revestimiento {_ci + 1}"
            _def_preset, _def_shoe = (
                _DEFAULT_CASING_PRESETS[_ci] if _ci < len(_DEFAULT_CASING_PRESETS)
                else ('9-5/8" (8.681" ID)', 9000.0 + _ci * 1000)
            )
            with st.expander(
                f"Revestimiento {_ci + 1} — {_default_name}",
                expanded=(_ci == int(n_casings) - 1),
            ):
                _name = st.text_input("Nombre", value=_default_name, key=f"m1e_c_name_{_ci}")
                _preset = st.selectbox(
                    "Tamaño nominal", options=list(_CASING_SIZES.keys()),
                    index=list(_CASING_SIZES.keys()).index(_def_preset),
                    key=f"m1e_c_preset_{_ci}",
                )
                _pv = _CASING_SIZES[_preset]
                _cod, _cid = st.columns(2)
                _c_od = _cod.number_input(
                    "OD (in)", min_value=1.0, max_value=30.0,
                    value=float(_pv[0]) if _pv else 9.625,
                    step=0.001, format="%.3f", key=f"m1e_c_od_{_ci}",
                )
                _c_id = _cid.number_input(
                    "ID / drift (in)", min_value=0.5, max_value=29.0,
                    value=float(_pv[1]) if _pv else 8.681,
                    step=0.001, format="%.3f", key=f"m1e_c_id_{_ci}",
                )
                _c_shoe = st.number_input(
                    "Zapato MD (ft)", min_value=50.0, max_value=25_000.0,
                    value=float(_def_shoe), step=10.0, format="%.0f",
                    key=f"m1e_c_shoe_{_ci}",
                )
                casings.append(CasingString(
                    name=_name, od_in=_c_od, id_in=_c_id, shoe_depth_ft=_c_shoe
                ))

        st.divider()

        st.markdown("#### 🔵 Tubing")
        _t_preset = st.selectbox(
            "Tamaño nominal",
            options=list(_TUBING_SIZES.keys()),
            index=list(_TUBING_SIZES.keys()).index(_DEFAULT_TUBING_PRESET),
            key="m1e_t_preset",
        )
        _tpv = _TUBING_SIZES[_t_preset]
        _tc1, _tc2 = st.columns(2)
        _t_od = _tc1.number_input(
            "OD (in)", min_value=0.5, max_value=8.0,
            value=float(_tpv[0]) if _tpv else 2.875,
            step=0.001, format="%.3f", key="m1e_t_od",
        )
        _t_id = _tc2.number_input(
            "ID (in)", min_value=0.1, max_value=7.5,
            value=float(_tpv[1]) if _tpv else 2.441,
            step=0.001, format="%.3f", key="m1e_t_id",
        )
        _t_set = st.number_input(
            "Zapato tubing MD (ft)", min_value=50.0, max_value=25_000.0,
            value=float(_existing_geo.get("sensor_depth_md_ft", 7_200.0)),
            step=10.0, format="%.0f", key="m1e_t_set",
        )
        tubing = TubingString(od_in=_t_od, id_in=_t_id, set_depth_ft=_t_set)

        st.divider()

        st.markdown("#### ⚡ ESP")
        _has_esp = st.checkbox(
            "El pozo tiene ESP (bomba eléctrica sumergible)", value=True, key="m1e_has_esp"
        )
        _esp_intake: float | None = None
        if _has_esp:
            _esp_intake = float(st.number_input(
                "Profundidad intake ESP MD (ft)", min_value=50.0, max_value=25_000.0,
                value=float(_existing_geo.get("pump_depth_md_ft", 7_000.0)),
                step=10.0, format="%.0f", key="m1e_esp_intake",
            ))

        st.divider()

        st.markdown("#### 🔴 Perforaciones")
        _pc1, _pc2 = st.columns(2)
        _p_top = _pc1.number_input(
            "Tope MD (ft)", min_value=50.0, max_value=25_000.0,
            value=float(_existing_geo.get("perforation_top_md_ft", 7_400.0)),
            step=10.0, format="%.0f", key="m1e_p_top",
        )
        _p_bot = _pc2.number_input(
            "Fondo MD (ft)", min_value=50.0, max_value=25_000.0,
            value=float(_existing_geo.get("perforation_bottom_md_ft", 7_600.0)),
            step=10.0, format="%.0f", key="m1e_p_bot",
        )

        st.divider()

        # Save button — writes geometry fields to hub's JSON
        if st.button(
            "💾 Guardar estado mecánico",
            type="primary",
            use_container_width=True,
            key="m1e_save_btn",
        ):
            _geo_payload = {
                "well_id": well_id,
                "tubing_od_in": float(_t_od),
                "tubing_id_in": float(_t_id),
                "pump_depth_md_ft": float(_esp_intake) if _esp_intake is not None else None,
                "sensor_depth_md_ft": float(_t_set),
                "perforation_top_md_ft": float(_p_top),
                "perforation_bottom_md_ft": float(_p_bot),
                "perforation_mid_tvd_ft": float(_existing_geo.get(
                    "perforation_mid_tvd_ft", (_p_top + _p_bot) / 2
                )),
                "datum_name": str(_existing_geo.get("datum_name", "KB")),
                "datum_depth_tvd_ft": float(_existing_geo.get("datum_depth_tvd_ft", 0.0)),
                # Store casing summary (first casing = largest = shoe for Pwf)
                "casing_od_in": float(casings[0].od_in) if casings else 0.0,
                "casing_id_in": float(casings[0].id_in) if casings else 0.0,
            }
            try:
                geo_path.write_text(
                    json.dumps(_geo_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                st.success(f"✅ Estado mecánico guardado en `{geo_path.name}`.")
                _existing_geo = _geo_payload  # Update for downstream use
            except Exception as _save_exc:
                st.error(f"❌ No fue posible guardar: {_save_exc}")

    # =========================================================================
    # RIGHT COLUMN — schematic + QC
    # =========================================================================
    with right_col:
        config = WellMechConfig(
            well_id=well_id,
            casings=casings,
            tubing=tubing,
            perfs_top_ft=float(_p_top),
            perfs_bottom_ft=float(_p_bot),
            has_esp=_has_esp,
            esp_intake_depth_ft=_esp_intake,
        )

        # QC técnico
        qc_results = run_mech_qc(config)
        severity = mech_severity_level(qc_results)
        icon = _SEVERITY_ICON.get(severity, "")
        with st.expander(f"{icon} QC Estado Mecánico", expanded=(severity != "ok")):
            for _r in qc_results:
                _msg = f"**{_r.title}**  \n{_r.detail}"
                _SEVERITY_FN[_r.severity](_msg, icon=_SEVERITY_KW[_r.severity])

        # Well schematic
        st.markdown("**Esquema del pozo**")
        try:
            _fig = draw_well_schematic(config, qc_results)
            st.pyplot(_fig, use_container_width=True)
            plt.close(_fig)
        except Exception as _exc:
            st.error(f"Error al generar el esquema: {_exc}")

        # Download PNG
        try:
            _png = schematic_to_png_bytes(config)
            st.download_button(
                "⬇ Descargar esquema (PNG)",
                data=_png,
                file_name=f"{well_id}_estado_mecanico.png",
                mime="image/png",
                use_container_width=True,
                key="m1e_dl_png",
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🛢 ecoRTA M1 — Estado Mecánico del Pozo")
    st.caption(
        "Ingrese los diámetros y profundidades para generar el esquema y "
        "verificar la compatibilidad dimensional en tiempo real."
    )

    left_col, right_col = st.columns([0.40, 0.60], gap="large")

    # =========================================================================
    # LEFT COLUMN — inputs
    # =========================================================================
    with left_col:

        # ── Well ID ───────────────────────────────────────────────────────
        well_id = st.text_input("ID del pozo", value="W-001")

        st.divider()

        # ── Casing strings ────────────────────────────────────────────────
        st.subheader("🔩 Revestimientos")
        n_casings = st.number_input(
            "Número de revestimientos", min_value=1, max_value=3, value=2, step=1,
        )
        casings: list[CasingString] = []
        for i in range(int(n_casings)):
            with st.expander(
                f"Revestimiento {i + 1} — "
                f"{_DEFAULT_CASING_NAMES[i] if i < 3 else ''}",
                expanded=(i == int(n_casings) - 1),
            ):
                casings.append(_casing_inputs(i))

        st.divider()

        # ── Tubing ────────────────────────────────────────────────────────
        st.subheader("🔵 Tubing")
        t_preset = st.selectbox(
            "Tamaño nominal",
            options=list(_TUBING_SIZES.keys()),
            index=list(_TUBING_SIZES.keys()).index(_DEFAULT_TUBING_PRESET),
        )
        t_preset_vals = _TUBING_SIZES[t_preset]

        tc1, tc2 = st.columns(2)
        t_od = tc1.number_input(
            "OD (in)", min_value=0.5, max_value=8.0,
            value=float(t_preset_vals[0]) if t_preset_vals else 2.875,
            step=0.001, format="%.3f",
        )
        t_id = tc2.number_input(
            "ID (in)", min_value=0.1, max_value=7.5,
            value=float(t_preset_vals[1]) if t_preset_vals else 2.441,
            step=0.001, format="%.3f",
        )
        t_set = st.number_input(
            "Zapato MD (ft)", min_value=50.0, max_value=25_000.0,
            value=7_200.0, step=10.0, format="%.0f",
        )
        tubing = TubingString(od_in=t_od, id_in=t_id, set_depth_ft=t_set)

        st.divider()

        # ── ESP ───────────────────────────────────────────────────────────
        st.subheader("⚡ ESP")
        has_esp = st.checkbox("El pozo tiene ESP (bomba eléctrica sumergible)", value=True)
        esp_intake: float | None = None
        if has_esp:
            esp_intake = st.number_input(
                "Profundidad intake ESP MD (ft)", min_value=50.0, max_value=25_000.0,
                value=7_000.0, step=10.0, format="%.0f",
            )

        st.divider()

        # ── Perforaciones ─────────────────────────────────────────────────
        st.subheader("🔴 Perforaciones")
        pc1, pc2 = st.columns(2)
        p_top = pc1.number_input(
            "Tope MD (ft)", min_value=50.0, max_value=25_000.0,
            value=7_400.0, step=10.0, format="%.0f",
        )
        p_bot = pc2.number_input(
            "Fondo MD (ft)", min_value=50.0, max_value=25_000.0,
            value=7_600.0, step=10.0, format="%.0f",
        )

        st.divider()

        # ── Pwf inputs ────────────────────────────────────────────────────
        st.subheader("📐 Estimación Pwf")
        st.caption("Modelo: hidrostático + Darcy-Weisbach (fluido monofásico líquido)")

        pw1, pw2 = st.columns(2)
        qo = pw1.number_input("qo (STB/d)", min_value=0.0, value=500.0, step=10.0)
        qw = pw2.number_input("qw (STB/d)", min_value=0.0, value=50.0, step=10.0)
        pw3, pw4 = st.columns(2)
        api = pw3.number_input("°API", min_value=5.0, max_value=60.0, value=16.0, step=0.5)
        whp = pw4.number_input("WHP (psia)", min_value=0.0, value=100.0, step=5.0)
        pw5, pw6 = st.columns(2)
        mu_o = pw5.number_input("μo (cp)", min_value=0.1, max_value=2000.0, value=50.0, step=1.0)
        tvd_perf = pw6.number_input(
            "TVD perfs (ft)", min_value=50.0, max_value=25_000.0,
            value=(p_top + p_bot) / 2, step=10.0,
        )

    # =========================================================================
    # RIGHT COLUMN — schematic + QC + Pwf results
    # =========================================================================
    with right_col:

        # Build config
        config = WellMechConfig(
            well_id=well_id,
            casings=casings,
            tubing=tubing,
            perfs_top_ft=float(p_top),
            perfs_bottom_ft=float(p_bot),
            has_esp=has_esp,
            esp_intake_depth_ft=float(esp_intake) if esp_intake is not None else None,
        )

        # ── QC técnico ────────────────────────────────────────────────────
        qc_results = run_mech_qc(config)
        severity = mech_severity_level(qc_results)
        icon = _SEVERITY_ICON.get(severity, "")
        header = f"{icon} QC Estado Mecánico"

        with st.expander(header, expanded=(severity != "ok")):
            for r in qc_results:
                msg = f"**{r.title}**  \n{r.detail}"
                _SEVERITY_FN[r.severity](msg, icon=_SEVERITY_KW[r.severity])

        # ── Esquema mecánico ──────────────────────────────────────────────
        st.subheader("Esquema del pozo")
        try:
            fig = draw_well_schematic(config, qc_results)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as exc:
            st.error(f"Error al generar el esquema: {exc}")

        # Download PNG
        try:
            png_bytes = schematic_to_png_bytes(config)
            st.download_button(
                "⬇ Descargar esquema (PNG)",
                data=png_bytes,
                file_name=f"{well_id}_estado_mecanico.png",
                mime="image/png",
                use_container_width=True,
            )
        except Exception:
            pass

        st.divider()

        # ── Pwf estimation ────────────────────────────────────────────────
        st.subheader("Estimación Pwf")
        pwf_inp = PwfInputs(
            qo_stb_d=float(qo),
            qw_stb_d=float(qw),
            api=float(api),
            whp_psia=float(whp),
            tvd_perf_ft=float(tvd_perf),
            tubing_id_in=float(t_id),
            length_ft=float(t_set),
        )

        try:
            pwf_v2 = estimate_pwf_v2(pwf_inp, mu_o_cp=float(mu_o))
            pwf_v1 = estimate_pwf_v1(pwf_inp)

            m1, m2, m3 = st.columns(3)
            m1.metric("Pwf v2 — D-W (psia)", f"{pwf_v2:,.1f}")
            m2.metric("Pwf v1 — simplif. (psia)", f"{pwf_v1:,.1f}", delta=f"{pwf_v2 - pwf_v1:+.1f}")

            # Breakdown
            from src.well_mod.pwf import api_to_sg, _churchill_friction_factor, BBL_TO_FT3, DAY_TO_S
            import math
            sg_o_val = 141.5 / (float(api) + 131.5)
            ql = max(1e-9, float(qo) + float(qw))
            sg_mix = (float(qo) * sg_o_val + float(qw) * 1.0) / ql
            dp_hyd = 0.433 * sg_mix * float(tvd_perf)
            dp_fric = pwf_v2 - float(whp) - dp_hyd

            m3.metric("WC (%)", f"{100 * float(qw) / max(ql, 1e-9):.1f}")

            st.caption(
                f"Hidrostático: **{dp_hyd:,.1f} psia** | "
                f"Fricción D-W: **{dp_fric:,.1f} psia** | "
                f"WHP: **{float(whp):.1f} psia** | "
                f"SG mix: **{sg_mix:.3f}**"
            )

            if dp_fric > dp_hyd * 0.3:
                st.warning(
                    "La pérdida por fricción es > 30 % del gradiente hidrostático. "
                    "Para caudales altos con crudos pesados (alta viscosidad), "
                    "considere ajustar el diámetro del tubing.",
                    icon="⚠️",
                )

        except Exception as exc:
            st.error(f"Error en el cálculo de Pwf: {exc}")


if __name__ == "__main__":
    main()
