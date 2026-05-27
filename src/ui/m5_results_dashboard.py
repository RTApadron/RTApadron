"""Streamlit UI — Módulo 5: Resultados integrados ecoRTA.

Pestañas:
    📋 Resumen      — info del pozo, historia, QC badges
    🧪 PVT          — valores representativos M2
    📉 DCA          — EUR por modelo Arps + tabla comparativa
    🔬 RTA          — parámetros de yacimiento M4 + imagen overlay
    📊 Comparativo  — EUR DCA vs OOIP volumétrico vs volumen contactado
    💾 Exportar     — CSV / JSON / Excel / PDF por módulo
    📐 Validación   — tabla comparativa ecoRTA vs software comercial de referencia
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.domain.m5_models import ExternalSoftwareResult, WellResultsSummary
from src.services.m5_aggregator_service import build_well_results
from src.services.m5_comparison_service import (
    build_comparison_table,
    comparison_table_to_csv_bytes,
    load_external_result,
    save_external_result,
)

OUTPUT_DIR = PROJECT_ROOT / "output"

# ---------------------------------------------------------------------------
# Helpers de presentación
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "measured": "#27ae60",
    "calculated": "#2980b9",
    "estimated": "#e67e22",
    "demo": "#8e44ad",
    "preliminary": "#d35400",
    "missing": "#e74c3c",
}


def _badge(label: str, status: str) -> str:
    color = _STATUS_COLORS.get(status, "#7f8c8d")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.75rem;font-weight:600">{label}</span>'
    )


def _completeness_row(summary: WellResultsSummary) -> None:
    flags = summary.completeness_flags()
    cols = st.columns(len(flags))
    for col, (mod, ok) in zip(cols, flags.items()):
        icon = "✅" if ok else "⬜"
        col.metric(label=mod, value=icon)


def _fmt_num(value: float | None, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}{suffix}"


def _fmt_millions(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value / 1_000_000:,.3f} MM STB"


# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------

def _tab_resumen(summary: WellResultsSummary) -> None:
    wi = summary.well_info
    st.subheader("Información del pozo")

    c1, c2, c3 = st.columns(3)
    c1.metric("Well ID", wi.well_id)
    c2.metric("Nombre", wi.well_name or "—")
    c3.metric("Campo", wi.field_name or "—")

    c4, c5, c6 = st.columns(3)
    c4.metric("API (°)", _fmt_num(wi.api_gravity, 1))
    c5.metric("T pozo (°F)", _fmt_num(wi.temp_f, 1))
    c6.metric("Puntos historia", str(wi.history_points))

    st.subheader("Historia de producción")
    c7, c8, c9 = st.columns(3)
    c7.metric("Fecha inicio", str(wi.date_from) if wi.date_from else "—")
    c8.metric("Fecha fin", str(wi.date_to) if wi.date_to else "—")
    c9.metric("qo promedio", _fmt_num(wi.qo_avg_stb_d, 0, " STB/d"))

    if wi.pwf_source_counts:
        st.caption("Fuente de Pwf:")
        cols = st.columns(len(wi.pwf_source_counts))
        for col, (src, count) in zip(cols, wi.pwf_source_counts.items()):
            col.metric(src, count)

    st.subheader("Completitud de módulos")
    _completeness_row(summary)

    if summary.consolidated_warnings or wi.qc_warnings:
        with st.expander("⚠️ Advertencias consolidadas"):
            for w in summary.consolidated_warnings + wi.qc_warnings:
                st.warning(w)


def _tab_pvt(summary: WellResultsSummary) -> None:
    if summary.pvt is None:
        st.info("M2 PVT no disponible — ejecuta el pipeline M1-M2 primero.")
        return

    pvt = summary.pvt
    st.subheader("Propiedades PVT representativas (M2)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Correlación", pvt.oil_corr or "—")
    c2.metric("Calibrado con lab", "Sí" if pvt.calibrated else "No")
    c3.metric("Pb (psia)", _fmt_num(pvt.pb_psia, 0))

    c4, c5, c6 = st.columns(3)
    c4.metric("Bo promedio (RB/STB)", _fmt_num(pvt.avg_bo_rb_stb, 4))
    c5.metric("Rs promedio (scf/STB)", _fmt_num(pvt.avg_rs_scf_stb, 0))
    c6.metric("μo promedio (cp)", _fmt_num(pvt.avg_mu_o_cp, 3))

    # Badge de trazabilidad PVT
    _src_labels = {"lab": "medido (laboratorio)", "correlation": "estimado (correlación)", "default": "valores por defecto"}
    _src_label = _src_labels.get(pvt.pvt_source, pvt.pvt_source)
    st.markdown(
        _badge(_src_label, pvt.status),
        unsafe_allow_html=True,
    )
    st.caption("Valores representativos del intervalo de producción. Para análisis detallado ver UI M2.")


def _tab_dca(summary: WellResultsSummary) -> None:
    if summary.dca is None or not summary.dca.models:
        st.info("M3 DCA no disponible — ejecuta el pipeline con DCA activado.")
        return

    dca = summary.dca
    st.subheader("Análisis de Declinación Arps (M3)")

    import pandas as pd
    rows = []
    for m in dca.models:
        rows.append({
            "Modelo": m.model.capitalize(),
            "qi (STB/d)": round(m.qi_stb_d, 1),
            "Di nominal (/d)": f"{m.di_nominal_d:.6f}",
            "b": round(m.b, 3),
            "EUR (MM STB)": round(m.eur_stb / 1e6, 4),
            "R²": round(m.r2, 4),
            "RMSE (STB/d)": round(m.rmse_stb_d, 1),
            "Pronóstico (días)": m.forecast_days,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if dca.best_model:
        st.success(f"Mejor ajuste: **{dca.best_model.capitalize()}** (mayor R²)")

    st.markdown(
        _badge("calculado (ajuste Arps)", dca.status or "calculated"),
        unsafe_allow_html=True,
    )

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        labels = [m.model.capitalize() for m in dca.models]
        eurs = [m.eur_stb / 1e6 for m in dca.models]
        colors = ["#3498db", "#e74c3c", "#2ecc71"]
        for label, eur, color in zip(labels, eurs, colors):
            fig.add_bar(name=label, x=[label], y=[eur], marker_color=color)

        fig.update_layout(
            title="EUR por modelo Arps (MM STB)",
            yaxis_title="EUR (MM STB)",
            showlegend=False,
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.caption("Instala plotly para ver el gráfico de EUR.")


def _render_single_rta(rta: "RTASummary", overlay_png: "Path | None" = None) -> None:
    """Render metrics block for one RTASummary.

    *overlay_png*: path to the per-method PNG overlay (shown below metrics if present).
    """
    from pathlib import Path as _Path

    _rta_status = (rta.status or "demo")
    if _rta_status == "preliminary":
        st.markdown(
            _badge("△ PRELIMINAR — curvas tipo analíticas validadas", "preliminary"),
            unsafe_allow_html=True,
        )
        st.caption("Parámetros basados en curvas tipo analíticas. Pendiente validación vs software comercial.")
    else:
        st.markdown(
            _badge("⚠️ DEMO — curvas tipo no validadas", "demo"),
            unsafe_allow_html=True,
        )
        st.caption("Los parámetros siguientes son preliminares hasta que las curvas tipo sean digitalizadas y validadas.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Método", (rta.method or "—").replace("_", " ").title())
    c2.metric("kh (mD·ft)", _fmt_num(rta.kh_md_ft, 2))
    c3.metric("k (mD)", _fmt_num(rta.k_md, 4))

    # Row 2: static volumetric OOIP + static config drainage geometry
    c4, c5, c6 = st.columns(3)
    c4.metric(
        "OOIP volumétrico",
        _fmt_millions(rta.n_vol_stb),
        help="OOIP calculado de la geometría de drenaje ingresada en la configuración (no cambia con el joystick).",
    )
    c5.metric(
        "re config (ft)",
        _fmt_num(rta.re_ft, 0),
        help="Radio de drene de la configuración del yacimiento (input fijo).",
    )
    c6.metric(
        "Área config (acres)",
        _fmt_num(rta.area_acres, 2),
        help="Área de drene de la configuración del yacimiento (input fijo).",
    )

    # Row 3: dynamic match values from joystick (only if available)
    _has_dyn = rta.n_dyn_stb is not None or rta.re_dyn_ft is not None or rta.a_dyn_acres is not None
    if _has_dyn:
        c7, c8, c9 = st.columns(3)
        c7.metric(
            "N match (MM STB)",
            _fmt_millions(rta.n_dyn_stb),
            help="OOIP dinámico calculado desde la posición del joystick. Cuando N match ≈ OOIP volumétrico, el match es consistente con la geometría configurada.",
        )
        c8.metric(
            "re match (ft)",
            _fmt_num(rta.re_dyn_ft, 0),
            help="Radio de drene derivado del match joystick (dinámico).",
        )
        c9.metric(
            "Área match (acres)",
            _fmt_num(rta.a_dyn_acres, 2),
            help="Área de drene derivada del match joystick (dinámica).",
        )
    else:
        st.caption(
            "⏳ Valores de match dinámico no disponibles — mueve los dos ejes del joystick en M4 y vuelve a guardar.",
        )

    c10, c11 = st.columns(2)
    c10.metric("Multiplicador X", _fmt_num(rta.x_multiplier, 4))
    c11.metric("Multiplicador Y", _fmt_num(rta.y_multiplier, 4))

    b1, b2, _ = st.columns([1, 1, 2])
    with b1:
        st.markdown(
            _badge("kh · k — estimado (match)", rta.kh_status),
            unsafe_allow_html=True,
        )
    with b2:
        st.markdown(
            _badge("OOIP — estimado (volumétrico)", rta.n_vol_status),
            unsafe_allow_html=True,
        )

    if rta.qc_warnings:
        with st.expander("⚠️ QC técnico M4"):
            for w in rta.qc_warnings:
                st.warning(w)

    # Per-method overlay PNG
    if overlay_png is not None and isinstance(overlay_png, _Path) and overlay_png.exists():
        st.image(str(overlay_png), caption=f"Overlay curvas tipo M4 — {(rta.method or '').replace('_', ' ')}", use_container_width=True)


def _tab_rta(summary: WellResultsSummary) -> None:
    all_rta = getattr(summary, "rta_all_methods", {})

    if not all_rta and summary.rta is None:
        st.info("M4 RTA no disponible — realiza el matching en la UI M4 primero.")
        return

    st.subheader("Análisis de Transiente de Flujo — Match RTA (M4)")

    def _png_for(method_key: str) -> "Path | None":
        """Return the per-method overlay PNG path, falling back to legacy file."""
        from pathlib import Path as _Path
        _per_method = OUTPUT_DIR / f"{summary.well_id}_rta_{method_key}_overlay.png"
        if _per_method.exists():
            return _per_method
        # Legacy fallback: single file written before per-method support
        _legacy = OUTPUT_DIR / f"{summary.well_id}_rta_overlay.png"
        return _legacy if _legacy.exists() else None

    if all_rta:
        # Mostrar todos los métodos guardados como tabs
        _method_labels = {
            "fetkovich": "🔬 Fetkovich",
            "blasingame": "📊 Blasingame",
            "palacio_blasingame": "📊 Palacio-Blasingame",
            "agarwal_gardner": "📈 Agarwal-Gardner",
        }
        _methods_available = list(all_rta.keys())
        if len(_methods_available) == 1:
            _mk = _methods_available[0]
            _render_single_rta(all_rta[_mk], overlay_png=_png_for(_mk))
        else:
            _tab_names = [_method_labels.get(m, m.replace("_", " ").title()) for m in _methods_available]
            _method_tabs = st.tabs(_tab_names)
            for _mt, _mk in zip(_method_tabs, _methods_available):
                with _mt:
                    _render_single_rta(all_rta[_mk], overlay_png=_png_for(_mk))

        # Tabla comparativa de métodos si hay más de uno
        if len(_methods_available) > 1:
            st.divider()
            st.caption("Comparativa de métodos — convergencia de kh indica consistencia del match.")
            import pandas as _pd_m5
            _cmp = []
            for _mk, _rs in all_rta.items():
                _cmp.append({
                    "Método": _mk.replace("_", " ").title(),
                    "kh (mD·ft)": _fmt_num(_rs.kh_md_ft, 2),
                    "k (mD)": _fmt_num(_rs.k_md, 4),
                    "N vol. (MM STB)": _fmt_millions(_rs.n_vol_stb),
                    "Área match (acres)": _fmt_num(_rs.a_dyn_acres, 2),
                    "X": _fmt_num(_rs.x_multiplier, 4),
                    "Y": _fmt_num(_rs.y_multiplier, 4),
                    "Status": (_rs.status or "demo").upper(),
                })
            st.dataframe(_pd_m5.DataFrame(_cmp), use_container_width=True, hide_index=True)
    else:
        # Backward compat: single rta object
        _render_single_rta(
            summary.rta,
            overlay_png=_png_for(summary.rta.method or ""),
        )


def _tab_comparativo(summary: WellResultsSummary) -> None:
    st.subheader("Dashboard comparativo de volúmenes")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.warning("Instala plotly para ver el dashboard comparativo.")
        return

    # Recolectar valores
    eur_exp = summary.dca.eur_exponential_stb if summary.dca else None
    eur_hyp = summary.dca.eur_hyperbolic_stb if summary.dca else None
    eur_har = summary.dca.eur_harmonic_stb if summary.dca else None
    n_vol = summary.rta.n_vol_stb if summary.rta else None

    all_values = {
        "EUR Exponencial": eur_exp,
        "EUR Hiperbólico": eur_hyp,
        "EUR Armónico": eur_har,
        "OOIP Volumétrico\n(M4 config)": n_vol,
    }

    labels = [k for k, v in all_values.items() if v is not None]
    values_mm = [(v or 0) / 1e6 for v in all_values.values() if v is not None]

    if not labels:
        st.info("No hay datos suficientes de DCA o RTA para el comparativo.")
        return

    colors = ["#3498db", "#2ecc71", "#e67e22", "#8e44ad"]
    fig = go.Figure()
    for label, val, color in zip(labels, values_mm, colors):
        fig.add_bar(
            name=label,
            x=[label],
            y=[val],
            marker_color=color,
            text=[f"{val:.3f}"],
            textposition="outside",
        )

    fig.update_layout(
        title="Comparativo de volúmenes (MM STB)",
        yaxis_title="Volumen (MM STB)",
        showlegend=False,
        height=450,
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla resumen
    st.subheader("Tabla resumen integrada")
    import pandas as pd

    rows = []
    if summary.dca and summary.dca.models:
        best = max(summary.dca.models, key=lambda m: m.r2)
        rows.append({
            "Fuente": "DCA M3 (mejor ajuste)",
            "Método": best.model.capitalize(),
            "Volumen (MM STB)": round(best.eur_stb / 1e6, 4),
            "Status": "calculado",
            "R² / Confianza": round(best.r2, 4),
        })
        for m in summary.dca.models:
            if m.model != best.model:
                rows.append({
                    "Fuente": "DCA M3",
                    "Método": m.model.capitalize(),
                    "Volumen (MM STB)": round(m.eur_stb / 1e6, 4),
                    "Status": "calculado",
                    "R² / Confianza": round(m.r2, 4),
                })

    if summary.rta and summary.rta.n_vol_stb is not None:
        rows.append({
            "Fuente": "RTA M4",
            "Método": f"OOIP vol. ({(summary.rta.method or '').replace('_', ' ')})",
            "Volumen (MM STB)": round(summary.rta.n_vol_stb / 1e6, 4),
            "Status": (summary.rta.status or "demo").upper(),
            "R² / Confianza": "—",
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Relaciones de interés para tesis
    if eur_hyp and n_vol:
        ratio = eur_hyp / n_vol * 100
        st.info(
            f"**Factor de recobro aparente (EUR hiperbólico / OOIP vol.):** {ratio:.1f}%  \n"
            "⚠️ El OOIP es preliminar (curvas DEMO). Usar solo como referencia de orden de magnitud."
        )

    if summary.rta and summary.rta.status not in ("preliminary", None):
        st.markdown(
            _badge("Resultados RTA — DEMO. No usar para toma de decisiones.", "demo"),
            unsafe_allow_html=True,
        )


def _tab_exportar(summary: WellResultsSummary) -> None:
    st.subheader("Exportación consolidada")
    st.caption("Genera los artefactos de M5 y los hace disponibles para descarga.")

    from src.services.m5_export_service import (
        export_csv_bytes,
        export_json_bytes,
        export_excel_bytes,
        export_pdf_bytes,
    )

    well_id = summary.well_id

    # Cargar validación si existe (para incluirla en Excel y PDF)
    external = load_external_result(OUTPUT_DIR, well_id)
    comparison_rows = (
        build_comparison_table(summary, external)
        if external is not None
        else None
    )
    if external is not None:
        st.info(
            f"📎 Datos de validación cargados desde `output/{well_id}_external_reference.json` — "
            "se incluirá hoja 'Validacion' en Excel y página adicional en PDF."
        )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Preparar CSV consolidado"):
            data = export_csv_bytes(summary)
            st.download_button(
                "⬇ Descargar CSV",
                data=data,
                file_name=f"{well_id}_m5_summary.csv",
                mime="text/csv",
            )

        if st.button("Preparar JSON completo"):
            data = export_json_bytes(summary)
            st.download_button(
                "⬇ Descargar JSON",
                data=data,
                file_name=f"{well_id}_m5_summary.json",
                mime="application/json",
            )

    with col2:
        if st.button("Preparar Excel (por módulo)"):
            data = export_excel_bytes(summary, external, comparison_rows)
            st.download_button(
                "⬇ Descargar Excel",
                data=data,
                file_name=f"{well_id}_m5_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        if st.button("Preparar PDF (reporte)"):
            data = export_pdf_bytes(summary, external, comparison_rows)
            st.download_button(
                "⬇ Descargar PDF",
                data=data,
                file_name=f"{well_id}_m5_report.pdf",
                mime="application/pdf",
            )

    st.divider()
    st.caption(
        "Los archivos también se guardan en `output/` al hacer clic en cada botón. "
        "El PDF usa matplotlib — sin dependencias adicionales."
    )


# ---------------------------------------------------------------------------
# Tab 7: Validación vs software comercial
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
    "match":   "✅",
    "close":   "🟡",
    "diverge": "🔴",
    "missing": "⬜",
}
_STATUS_LABELS = {
    "match":   "Concordancia alta  (|Δ| < 5 %)",
    "close":   "Concordancia media (5 % ≤ |Δ| < 20 %)",
    "diverge": "Divergencia        (|Δ| ≥ 20 %)",
    "missing": "Dato faltante",
}


def _validation_header_badge(rows: list) -> None:
    """Muestra badge de validación global basado en el score de concordancia."""
    if not rows:
        return
    counts = {"match": 0, "close": 0, "diverge": 0, "missing": 0}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    total_comp = counts["match"] + counts["close"] + counts["diverge"]
    if total_comp == 0:
        return
    pct_ok = (counts["match"] + counts["close"]) / total_comp * 100
    if pct_ok >= 80:
        st.markdown(
            _badge(f"✅ VALIDADO — {pct_ok:.0f}% concordancia (match + close)", "measured"),
            unsafe_allow_html=True,
        )
    elif pct_ok >= 50:
        st.markdown(
            _badge(f"⚠️ VALIDACIÓN PARCIAL — {pct_ok:.0f}% concordancia", "preliminary"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _badge(f"🔴 DIVERGENCIA ALTA — solo {pct_ok:.0f}% concordancia", "demo"),
            unsafe_allow_html=True,
        )


def _tab_validacion(summary: WellResultsSummary) -> None:
    st.subheader("Validación vs software comercial")
    st.caption(
        "Ingresa los valores de referencia del software comercial para generar "
        "la tabla comparativa. Los valores de ecoRTA se toman de M3 DCA y M4 RTA."
    )

    well_id = summary.well_id

    # ── Cargar valores previos si existen ────────────────────────────────────
    saved = load_external_result(OUTPUT_DIR, well_id)
    if saved and "ext_loaded" not in st.session_state:
        st.session_state["ext_loaded"] = True
        st.session_state["ext_data"] = saved

    ext_init: ExternalSoftwareResult = st.session_state.get("ext_data") or ExternalSoftwareResult()

    if saved and "ext_loaded" in st.session_state:
        st.info(f"📂 Valores cargados desde `output/{well_id}_external_reference.json`")

    # ── Formulario de entrada ────────────────────────────────────────────────
    with st.expander("📝 Valores del software comercial", expanded=True):
        software_label = st.text_input(
            "Nombre del software de referencia",
            value=ext_init.software_label,
            help="Ej.: 'Software Comercial', 'Kappa Ecrin', etc.",
        )

        st.markdown("**DCA — Declinación**")
        c1, c2, c3, c4 = st.columns(4)
        ext_eur = c1.number_input(
            "EUR (MM STB)", min_value=0.0, value=float(ext_init.eur_stb / 1e6) if ext_init.eur_stb else 0.0,
            format="%.4f", step=0.001,
        )
        ext_qi = c2.number_input(
            "qi (STB/d)", min_value=0.0, value=float(ext_init.qi_stb_d or 0.0),
            format="%.1f", step=1.0,
        )
        ext_di = c3.number_input(
            "Di nominal (1/d)", min_value=0.0, value=float(ext_init.di_nominal_d or 0.0),
            format="%.6f", step=0.0001,
        )
        ext_b = c4.number_input(
            "b (Arps)", min_value=0.0, max_value=1.0, value=float(ext_init.b_factor or 0.0),
            format="%.3f", step=0.01,
        )

        st.markdown("**RTA — Parámetros de yacimiento**")
        c5, c6, c7 = st.columns(3)
        ext_kh = c5.number_input(
            "kh (mD·ft)", min_value=0.0, value=float(ext_init.kh_md_ft or 0.0),
            format="%.2f", step=0.1,
        )
        ext_k = c6.number_input(
            "k (mD)", min_value=0.0, value=float(ext_init.k_md or 0.0),
            format="%.4f", step=0.001,
        )
        ext_n = c7.number_input(
            "OOIP volumétrico (MM STB)", min_value=0.0, value=float(ext_init.n_vol_stb / 1e6) if ext_init.n_vol_stb else 0.0,
            format="%.4f", step=0.001,
        )

        ext_notes = st.text_area(
            "Notas / condiciones del modelo de referencia",
            value=ext_init.notes or "",
            height=60,
        )

    # ── Selector de modelo DCA para comparar ─────────────────────────────────
    dca_models_available = (
        [m.model for m in summary.dca.models] if summary.dca and summary.dca.models else []
    )
    dca_options = ["best"] + dca_models_available
    dca_sel = st.selectbox(
        "Modelo DCA ecoRTA a comparar",
        options=dca_options,
        format_func=lambda x: "Mejor R² (automático)" if x == "best" else x.capitalize(),
    )

    # ── Construir ExternalSoftwareResult ──────────────────────────────────────
    external = ExternalSoftwareResult(
        software_label=software_label,
        eur_stb=ext_eur * 1e6 if ext_eur > 0 else None,
        qi_stb_d=ext_qi if ext_qi > 0 else None,
        di_nominal_d=ext_di if ext_di > 0 else None,
        b_factor=ext_b if ext_b > 0 else None,
        kh_md_ft=ext_kh if ext_kh > 0 else None,
        k_md=ext_k if ext_k > 0 else None,
        n_vol_stb=ext_n * 1e6 if ext_n > 0 else None,
        notes=ext_notes or None,
    )

    # ── Guardar ───────────────────────────────────────────────────────────────
    col_save, col_dl = st.columns([1, 3])
    if col_save.button("💾 Guardar valores de referencia"):
        path = save_external_result(external, OUTPUT_DIR, well_id)
        st.session_state["ext_data"] = external
        st.success(f"Guardado en `{path.name}`")

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Tabla comparativa: ecoRTA vs {software_label}")

    rows = build_comparison_table(summary, external, dca_model=dca_sel)

    # Badge de validación global (visible en la parte superior del análisis)
    _validation_header_badge(rows)

    import pandas as pd

    table_data = []
    for r in rows:
        icon = _STATUS_ICONS.get(r.status, "")
        table_data.append({
            "Parámetro": r.parameter,
            "Unidades": r.units,
            "ecoRTA": f"{r.ecorta_value:.4g}" if r.ecorta_value is not None else "—",
            software_label: f"{r.external_value:.4g}" if r.external_value is not None else "—",
            "Δ relativo (%)": f"{r.rel_diff_pct:+.2f} %" if r.rel_diff_pct is not None else "—",
            "Concordancia": f"{icon} {r.status}",
            "Nota": r.note,
        })

    df_comp = pd.DataFrame(table_data)

    # Color de filas por status
    def _highlight(row: pd.Series) -> list[str]:
        conc = str(row.get("Concordancia", ""))
        if "match" in conc:
            return ["background-color: #eafaf1"] * len(row)
        if "close" in conc:
            return ["background-color: #fef9e7"] * len(row)
        if "diverge" in conc:
            return ["background-color: #fdedec"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_comp.style.apply(_highlight, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # Leyenda
    st.caption("  ".join(f"{icon} {label}" for icon, (_, label) in zip(
        _STATUS_ICONS.values(), _STATUS_LABELS.items()
    )))

    # ── Resumen de concordancia ───────────────────────────────────────────────
    counts = {"match": 0, "close": 0, "diverge": 0, "missing": 0}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1

    meaningful = [r for r in rows if r.status != "missing"]
    if meaningful:
        st.subheader("Resumen de concordancia")
        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("✅ Concordancia alta", counts["match"])
        cm2.metric("🟡 Concordancia media", counts["close"])
        cm3.metric("🔴 Divergencia", counts["diverge"])
        cm4.metric("⬜ Datos faltantes", counts["missing"])

        total_comp = counts["match"] + counts["close"] + counts["diverge"]
        if total_comp > 0:
            pct_ok = (counts["match"] + counts["close"]) / total_comp * 100
            if pct_ok >= 80:
                st.success(f"**{pct_ok:.0f} %** de los parámetros con concordancia alta o media.")
            elif pct_ok >= 50:
                st.warning(f"**{pct_ok:.0f} %** de los parámetros con concordancia alta o media.")
            else:
                st.error(f"**{pct_ok:.0f} %** de los parámetros con concordancia alta o media — revisar configuración.")

    # ── Descarga ──────────────────────────────────────────────────────────────
    if rows:
        csv_bytes = comparison_table_to_csv_bytes(rows, summary, external)
        st.download_button(
            "⬇ Descargar tabla comparativa (CSV)",
            data=csv_bytes,
            file_name=f"{well_id}_comparativo_vs_ref.csv",
            mime="text/csv",
        )

    # ── Advertencia status DEMO ───────────────────────────────────────────────
    if summary.rta and summary.rta.status not in ("preliminary", None):
        st.divider()
        st.markdown(
            _badge(
                "⚠️ Parámetros RTA son DEMO (curvas tipo no validadas) — "
                "comparación con fines exploratorios únicamente.",
                "demo",
            ),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Función embebible — para usar desde app.py sin st.set_page_config
# ---------------------------------------------------------------------------

def render_m5_embedded(well_id: str, output_dir: Path) -> None:
    """Renderiza el dashboard M5 completo embebido en otra app Streamlit.

    Usa well_id del sidebar del hub (app.py) sin mostrar selector propio.
    El estado se guarda en session_state con prefijo 'm5_' para no colisionar.
    """
    key_summary = f"m5_summary_{well_id}"

    # Botón de carga / recarga
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        reload = st.button("🔄 Cargar / actualizar M5", key=f"m5_load_{well_id}")

    if reload or key_summary not in st.session_state:
        with st.spinner("Agregando resultados M1-M4…"):
            try:
                summary = build_well_results(well_id=well_id, output_dir=output_dir)
                st.session_state[key_summary] = summary
                with col_status:
                    st.success("Resultados cargados correctamente.")
            except Exception as exc:
                with col_status:
                    st.error(f"Error al cargar resultados: {exc}")
                return

    summary: WellResultsSummary | None = st.session_state.get(key_summary)
    if summary is None:
        st.info("Haz clic en **Cargar / actualizar M5** para agregar los resultados de M1-M4.")
        return

    # Verificar que los datos corresponden al well_id activo
    if summary.well_id != well_id:
        st.warning("El Well ID cambió — haz clic en **Cargar / actualizar M5**.")
        return

    # Completitud rápida al inicio
    flags = summary.completeness_flags()
    n_ok = sum(flags.values())
    if n_ok < len(flags):
        missing = [k for k, v in flags.items() if not v]
        st.warning(f"Módulos sin datos en output/: {', '.join(missing)}. Ejecuta el pipeline primero.")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Resumen",
        "🧪 PVT",
        "📉 DCA",
        "🔬 RTA",
        "📊 Comparativo",
        "💾 Exportar",
        "📐 Validación",
    ])

    with tab1:
        _tab_resumen(summary)
    with tab2:
        _tab_pvt(summary)
    with tab3:
        _tab_dca(summary)
    with tab4:
        _tab_rta(summary)
    with tab5:
        _tab_comparativo(summary)
    with tab6:
        _tab_exportar(summary)
    with tab7:
        _tab_validacion(summary)


# ---------------------------------------------------------------------------
# Entry point — app standalone
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="ecoRTA — M5 Resultados",
        page_icon="🛢️",
        layout="wide",
    )
    st.title("🛢️ ecoRTA — M5: Resultados Integrados")
    st.caption("Módulo 5: agrega M1 historia · M2 PVT · M3 DCA · M4 RTA en un resumen ejecutivo.")

    # Selector de pozo
    well_id = st.text_input("Well ID", value="W001", help="Prefijo de los archivos en output/")

    if not well_id.strip():
        st.stop()

    if st.button("🔄 Cargar resultados", type="primary"):
        with st.spinner("Agregando resultados M1-M4…"):
            try:
                summary = build_well_results(well_id=well_id.strip(), output_dir=OUTPUT_DIR)
                st.session_state["m5_summary"] = summary
            except Exception as exc:
                st.error(f"Error al cargar resultados: {exc}")
                st.stop()

    summary: WellResultsSummary | None = st.session_state.get("m5_summary")

    if summary is None:
        st.info("Ingresa un Well ID y haz clic en **Cargar resultados** para comenzar.")
        return

    if summary.well_id != well_id.strip():
        st.warning("El Well ID cambió — haz clic en **Cargar resultados** para actualizar.")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Resumen",
        "🧪 PVT",
        "📉 DCA",
        "🔬 RTA",
        "📊 Comparativo",
        "💾 Exportar",
        "📐 Validación",
    ])

    with tab1:
        _tab_resumen(summary)
    with tab2:
        _tab_pvt(summary)
    with tab3:
        _tab_dca(summary)
    with tab4:
        _tab_rta(summary)
    with tab5:
        _tab_comparativo(summary)
    with tab6:
        _tab_exportar(summary)
    with tab7:
        _tab_validacion(summary)


if __name__ == "__main__":
    main()
