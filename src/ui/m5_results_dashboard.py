"""Streamlit UI — Módulo 5: Resultados integrados ecoRTA.

Pestañas:
    📋 Resumen      — info del pozo, historia, QC badges
    🧪 PVT          — valores representativos M2
    📉 DCA          — EUR por modelo Arps + tabla comparativa
    🔬 RTA          — parámetros de yacimiento M4 + imagen overlay
    📊 Comparativo  — EUR DCA vs OOIP volumétrico vs volumen contactado
    💾 Exportar     — CSV / JSON / Excel / PDF por módulo
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.domain.m5_models import WellResultsSummary
from src.services.m5_aggregator_service import build_well_results

OUTPUT_DIR = PROJECT_ROOT / "output"

# ---------------------------------------------------------------------------
# Helpers de presentación
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "measured": "#27ae60",
    "calculated": "#2980b9",
    "estimated": "#e67e22",
    "demo": "#8e44ad",
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

    st.markdown(
        _badge("estimado", pvt.status),
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


def _tab_rta(summary: WellResultsSummary) -> None:
    if summary.rta is None:
        st.info("M4 RTA no disponible — realiza el matching en la UI M4 primero.")
        return

    rta = summary.rta
    st.subheader("Análisis de Transiente de Flujo — Match RTA (M4)")

    st.markdown(
        _badge("⚠️ DEMO — curvas tipo no validadas", "demo"),
        unsafe_allow_html=True,
    )
    st.caption("Los parámetros siguientes son preliminares hasta que las curvas tipo sean digitalizadas y validadas.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Método", (rta.method or "—").replace("_", " ").title())
    c2.metric("kh (mD·ft)", _fmt_num(rta.kh_md_ft, 2))
    c3.metric("k (mD)", _fmt_num(rta.k_md, 4))

    c4, c5, c6 = st.columns(3)
    c4.metric("OOIP volumétrico", _fmt_millions(rta.n_vol_stb))
    c5.metric("re (ft)", _fmt_num(rta.re_ft, 0))
    c6.metric("Área drene (acres)", _fmt_num(rta.area_acres, 2))

    c7, c8 = st.columns(2)
    c7.metric("Multiplicador X", _fmt_num(rta.x_multiplier, 4))
    c8.metric("Multiplicador Y", _fmt_num(rta.y_multiplier, 4))

    # Imagen overlay si existe
    overlay_png = OUTPUT_DIR / f"{summary.well_id}_rta_overlay.png"
    if overlay_png.exists():
        st.image(str(overlay_png), caption="Overlay curvas tipo (M4)", use_container_width=True)

    if rta.qc_warnings:
        with st.expander("⚠️ QC técnico M4"):
            for w in rta.qc_warnings:
                st.warning(w)


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
            "Status": "DEMO",
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

    if summary.rta and summary.rta.status == "demo":
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
            data = export_excel_bytes(summary)
            st.download_button(
                "⬇ Descargar Excel",
                data=data,
                file_name=f"{well_id}_m5_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        if st.button("Preparar PDF (reporte)"):
            data = export_pdf_bytes(summary)
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
# Entry point
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Resumen",
        "🧪 PVT",
        "📉 DCA",
        "🔬 RTA",
        "📊 Comparativo",
        "💾 Exportar",
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


if __name__ == "__main__":
    main()
