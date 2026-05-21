"""M2 — Editor interactivo PVT.

Calcula y visualiza propiedades PVT dependientes de presión usando las
correlaciones Standing (1947) y Vasquez-Beggs (1980) para Rs/Bo,
y Beggs-Robinson (1975) para viscosidad.

Características:
  - Comparación side-by-side Standing vs Vasquez-Beggs
  - Gráficas Rs, Bo, μo y ρo vs presión con línea de Pb
  - Tabla completa de propiedades exportable a CSV
  - Métricas clave en Pb (presión de burbuja)
  - Carga opcional de datos de laboratorio (CSV) para comparación visual
  - QC automático: alerta si Pb está fuera del rango de presiones

Run from project root:
    python -m streamlit run src/ui/m2_pvt_editor.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st

from src.services.pvt_correlations import api_to_sg
from src.services.pvt_service import PVTTableInput, PVTPressurePoint, compute_pvt_table

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _points_to_df(points: list[PVTPressurePoint]) -> pd.DataFrame:
    """Convert list of PVTPressurePoint to a tidy DataFrame."""
    return pd.DataFrame([
        {
            "P (psia)": pt.p_psia,
            "Rs (scf/STB)": pt.rs_scf_stb,
            "Bo (bbl/STB)": pt.bo_rb_stb,
            "μo (cp)": pt.mu_o_cp,
            "ρo (lb/ft³)": pt.rho_o_lb_ft3,
            "Régimen": pt.regime,
            "Es Pb": pt.is_pb,
        }
        for pt in points
    ])


def _draw_pvt_plots(
    points_st: list[PVTPressurePoint],
    pb_st: float,
    points_vb: list[PVTPressurePoint] | None,
    pb_vb: float | None,
    lab_df: pd.DataFrame | None,
) -> plt.Figure:
    """Draw a 2×2 grid of PVT plots: Rs, Bo, μo, ρo."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="#FAFAFA")
    fig.suptitle("Propiedades PVT vs Presión", fontsize=13, fontweight="bold")

    p_st  = [pt.p_psia for pt in points_st]
    rs_st = [pt.rs_scf_stb for pt in points_st]
    bo_st = [pt.bo_rb_stb for pt in points_st]
    mu_st = [pt.mu_o_cp for pt in points_st]
    rho_st = [pt.rho_o_lb_ft3 for pt in points_st]

    plot_data = [
        (axes[0, 0], rs_st, "Rs (scf/STB)", "Rs — Solución GOR",
         "Rs_scfstb" if lab_df is not None else None),
        (axes[0, 1], bo_st, "Bo (bbl/STB)", "Bo — Factor Volumétrico",
         "Bo_rbblstb" if lab_df is not None else None),
        (axes[1, 0], mu_st, "μo (cp)", "μo — Viscosidad",
         "mu_cp" if lab_df is not None else None),
        (axes[1, 1], rho_st, "ρo (lb/ft³)", "ρo — Densidad en yacimiento",
         None),
    ]

    for ax, y_st, ylabel, title, lab_col in plot_data:
        ax.set_facecolor("#F8F9FA")
        ax.plot(p_st, y_st, color="#1D4ED8", linewidth=1.8, label="Standing")

        if points_vb is not None:
            p_vb = [pt.p_psia for pt in points_vb]
            if ylabel.startswith("Rs"):
                y_vb = [pt.rs_scf_stb for pt in points_vb]
            elif ylabel.startswith("Bo"):
                y_vb = [pt.bo_rb_stb for pt in points_vb]
            elif ylabel.startswith("μo"):
                y_vb = [pt.mu_o_cp for pt in points_vb]
            else:
                y_vb = [pt.rho_o_lb_ft3 for pt in points_vb]
            ax.plot(p_vb, y_vb, color="#DC2626", linewidth=1.4,
                    linestyle="--", label="Vasquez-Beggs")
            if pb_vb is not None:
                ax.axvline(pb_vb, color="#DC2626", linewidth=0.8,
                           linestyle=":", alpha=0.7)

        # Pb line (Standing)
        ax.axvline(pb_st, color="#1D4ED8", linewidth=1.0, linestyle=":",
                   alpha=0.8, label=f"Pb={pb_st:.0f} psia (St)")

        # Lab data overlay — drop rows where EITHER P_psia or the column is NaN
        if lab_df is not None and lab_col and lab_col in lab_df.columns and "P_psia" in lab_df.columns:
            _lab_pair = lab_df[["P_psia", lab_col]].dropna()
            if not _lab_pair.empty:
                ax.scatter(_lab_pair["P_psia"], _lab_pair[lab_col],
                           marker="o", color="#059669",
                           s=30, zorder=5, label="Laboratorio")

        ax.set_xlabel("Presión (psia)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.legend(fontsize=6.5, framealpha=0.85)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Core UI (shared between standalone and embedded modes)
# ---------------------------------------------------------------------------

def _pvt_core_ui() -> None:
    """PVT UI body — callable from both standalone main() and render_m2_embedded()."""

    left_col, right_col = st.columns([0.38, 0.62], gap="large")

    # =========================================================================
    # LEFT COLUMN — inputs
    # =========================================================================
    with left_col:

        # ── Fluid properties ──────────────────────────────────────────────
        st.subheader("🛢 Propiedades del fluido")
        fc1, fc2 = st.columns(2)
        api = fc1.number_input(
            "°API", min_value=5.0, max_value=60.0, value=14.5, step=0.5,
            help="Gravedad del aceite en °API. Llanos pesados: 9–16 °API.",
        )
        gamma_g = fc2.number_input(
            "γg (gas, aire=1)", min_value=0.50, max_value=1.20, value=0.75,
            step=0.01, format="%.2f",
            help="Gravedad específica del gas respecto al aire.",
        )
        fc3, fc4 = st.columns(2)
        t_f = fc3.number_input(
            "T (°F)", min_value=60.0, max_value=400.0, value=195.0, step=5.0,
            help="Temperatura de yacimiento en °F.",
        )
        rsb = fc4.number_input(
            "Rsb (scf/STB)", min_value=10.0, max_value=3000.0, value=120.0,
            step=10.0,
            help="GOR en solución a la presión de burbuja.",
        )

        st.divider()

        # ── Pressure range ────────────────────────────────────────────────
        st.subheader("📊 Rango de presión")
        pc1, pc2 = st.columns(2)
        p_min = pc1.number_input("P mín (psia)", min_value=14.7, max_value=500.0,
                                  value=50.0, step=10.0)
        p_max = pc2.number_input("P máx (psia)", min_value=200.0, max_value=15000.0,
                                  value=4000.0, step=100.0)
        n_pts = st.slider("Número de puntos en la grilla", 20, 200, 80, 10)

        st.divider()

        # ── Undersaturated properties ─────────────────────────────────────
        st.subheader("⬆️ Propiedades > Pb (sub-saturado)")
        co = st.number_input(
            "Compresibilidad del aceite co (1/psi)",
            min_value=1e-6, max_value=1e-3, value=1.2e-5,
            format="%.2e", step=1e-6,
            help="Compresibilidad isotérmica del aceite por encima de Pb. "
                 "Típico para crudos pesados: 5e-6 a 2e-5 psi⁻¹.",
        )

        st.divider()

        # ── Correlation comparison toggle ─────────────────────────────────
        st.subheader("🔬 Correlaciones")
        show_vb = st.checkbox(
            "Comparar con Vasquez-Beggs (1980)", value=True,
            help="Muestra la curva de Vasquez-Beggs superpuesta con Standing.",
        )

        st.divider()

        # ── Lab data ──────────────────────────────────────────────────────
        st.subheader("🧫 Datos de laboratorio (opcional)")

        lab_mode = st.radio(
            "Fuente de datos de laboratorio",
            ["Introducir manualmente", "Cargar CSV"],
            horizontal=True,
            label_visibility="collapsed",
        )

        lab_df: pd.DataFrame | None = None

        if lab_mode == "Cargar CSV":
            st.caption(
                "Columnas: `P_psia` (requerida), `Rs_scfstb`, `Bo_rbblstb`, `mu_cp` (opcionales)."
            )
            uploaded = st.file_uploader("Cargar CSV laboratorio", type=["csv"])
            if uploaded is not None:
                try:
                    lab_df = pd.read_csv(uploaded)
                    st.success(f"{len(lab_df)} puntos cargados.", icon="✅")
                except Exception as exc:
                    st.error(f"Error leyendo CSV: {exc}")

        else:
            # ── Manual entry via st.data_editor ──────────────────────────
            st.caption(
                "Completa solo las columnas que tienes. Deja vacías las demás. "
                "Usa ➕ (fila inferior) para agregar más puntos."
            )

            # Version counter — incrementing resets the editor widget
            if "lab_editor_version" not in st.session_state:
                st.session_state["lab_editor_version"] = 0

            col_btn, col_info = st.columns([1, 2])
            if col_btn.button(
                "↺ Regenerar plantilla",
                help="Recrea la tabla con 6 puntos distribuidos en el rango de presión actual",
                use_container_width=True,
            ):
                st.session_state["lab_editor_version"] += 1

            # Build template from current p_min / p_max
            _n = 6
            _step = (float(p_max) - float(p_min)) / (_n - 1)
            _template = pd.DataFrame({
                "P_psia":     [round(float(p_min) + i * _step) for i in range(_n)],
                "Rs_scfstb":  [None] * _n,
                "Bo_rbblstb": [None] * _n,
                "mu_cp":      [None] * _n,
            })

            _edited = st.data_editor(
                _template,
                num_rows="dynamic",
                use_container_width=True,
                key=f"lab_manual_{st.session_state['lab_editor_version']}",
                column_config={
                    "P_psia": st.column_config.NumberColumn(
                        "P (psia)", min_value=0.0, format="%.1f",
                    ),
                    "Rs_scfstb": st.column_config.NumberColumn(
                        "Rs (scf/STB)", min_value=0.0, format="%.1f",
                    ),
                    "Bo_rbblstb": st.column_config.NumberColumn(
                        "Bo (bbl/STB)", min_value=0.0, format="%.4f",
                    ),
                    "mu_cp": st.column_config.NumberColumn(
                        "μo (cp)", min_value=0.0, format="%.3f",
                    ),
                },
            )

            # Only keep rows where P_psia is filled
            _valid = _edited.dropna(subset=["P_psia"])
            if not _valid.empty:
                lab_df = _valid
                col_info.caption(f"✅ {len(_valid)} punto(s) activo(s)")

                # Download the manually entered data as CSV
                _csv_lab = _valid.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Guardar datos lab (CSV)",
                    data=_csv_lab,
                    file_name="lab_datos_laboratorio.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                col_info.caption("— Sin puntos activos")

    # =========================================================================
    # RIGHT COLUMN — results
    # =========================================================================
    with right_col:

        # Build Standing table
        inp_st = PVTTableInput(
            api=float(api),
            gamma_g=float(gamma_g),
            t_f=float(t_f),
            rsb_scf_stb=float(rsb),
            p_min_psia=float(p_min),
            p_max_psia=float(p_max),
            n_points=int(n_pts),
            co_psi=float(co),
            correlation="standing",
        )

        try:
            pb_st, points_st = compute_pvt_table(inp_st)
        except ValueError as exc:
            st.error(f"Error en PVT (Standing): {exc}")
            st.stop()

        # Build VB table (optional)
        pb_vb: float | None = None
        points_vb: list[PVTPressurePoint] | None = None
        if show_vb:
            inp_vb = PVTTableInput(
                api=float(api),
                gamma_g=float(gamma_g),
                t_f=float(t_f),
                rsb_scf_stb=float(rsb),
                p_min_psia=float(p_min),
                p_max_psia=float(p_max),
                n_points=int(n_pts),
                co_psi=float(co),
                correlation="vasquez_beggs",
            )
            try:
                pb_vb, points_vb = compute_pvt_table(inp_vb)
            except ValueError as exc:
                st.warning(f"Vasquez-Beggs no disponible: {exc}")

        # ── QC alerts ────────────────────────────────────────────────────
        if pb_st < float(p_min):
            st.error(
                f"⚠️ Pb Standing ({pb_st:.0f} psia) es **menor que P mín** "
                f"({p_min:.0f} psia). Ajuste el rango de presiones o el Rsb.",
                icon="🔴",
            )
        elif pb_st > float(p_max):
            st.warning(
                f"Pb Standing ({pb_st:.0f} psia) supera P máx ({p_max:.0f} psia). "
                "La curva está truncada — aumente P máx para ver toda la curva.",
                icon="⚠️",
            )

        # ── Key metrics at Pb ─────────────────────────────────────────────
        st.subheader("Propiedades en Pb (Standing)")
        pb_pt = next((p for p in points_st if p.is_pb), None)
        gamma_o = api_to_sg(float(api))

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Pb (psia)", f"{pb_st:,.0f}")
        if pb_pt:
            mc2.metric("Rs@Pb (scf/STB)", f"{pb_pt.rs_scf_stb:,.1f}")
            mc3.metric("Bo@Pb (bbl/STB)", f"{pb_pt.bo_rb_stb:.4f}")
            mc4.metric("μo@Pb (cp)", f"{pb_pt.mu_o_cp:.2f}")
            mc5.metric("ρo@Pb (lb/ft³)", f"{pb_pt.rho_o_lb_ft3:.2f}")

        if show_vb and pb_vb is not None:
            pb_pt_vb = next((p for p in (points_vb or []) if p.is_pb), None)
            st.caption(
                f"Vasquez-Beggs — Pb: **{pb_vb:,.0f} psia** | "
                + (
                    f"Rs@Pb: **{pb_pt_vb.rs_scf_stb:.1f} scf/STB** | "
                    f"Bo@Pb: **{pb_pt_vb.bo_rb_stb:.4f} bbl/STB** | "
                    f"μo@Pb: **{pb_pt_vb.mu_o_cp:.2f} cp**"
                    if pb_pt_vb else ""
                )
            )

        # ── Plots ─────────────────────────────────────────────────────────
        st.subheader("Curvas PVT")
        try:
            fig = _draw_pvt_plots(points_st, pb_st, points_vb, pb_vb, lab_df)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as exc:
            st.error(f"Error al generar gráficas: {exc}")

        # ── Full table ────────────────────────────────────────────────────
        st.divider()
        st.subheader("Tabla completa — Standing")
        df_st = _points_to_df(points_st)

        # Style: bold the Pb row
        def _style_pb(row: pd.Series) -> list[str]:
            if row["Es Pb"]:
                return ["font-weight: bold; background-color: #DBEAFE"] * len(row)
            return [""] * len(row)

        styled = (
            df_st.style
            .apply(_style_pb, axis=1)
            .format({
                "P (psia)": "{:.1f}",
                "Rs (scf/STB)": "{:.2f}",
                "Bo (bbl/STB)": "{:.5f}",
                "μo (cp)": "{:.4f}",
                "ρo (lb/ft³)": "{:.3f}",
            })
        )
        st.dataframe(styled, use_container_width=True, height=280)

        # CSV download
        csv_bytes = df_st.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Descargar tabla PVT (CSV)",
            data=csv_bytes,
            file_name=f"pvt_standing_api{api:.0f}_T{t_f:.0f}F.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if show_vb and points_vb:
            with st.expander("Tabla Vasquez-Beggs"):
                df_vb = _points_to_df(points_vb)
                st.dataframe(
                    df_vb.style.format({
                        "P (psia)": "{:.1f}",
                        "Rs (scf/STB)": "{:.2f}",
                        "Bo (bbl/STB)": "{:.5f}",
                        "μo (cp)": "{:.4f}",
                        "ρo (lb/ft³)": "{:.3f}",
                    }),
                    use_container_width=True,
                    height=220,
                )
                csv_vb = df_vb.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Descargar tabla Vasquez-Beggs (CSV)",
                    data=csv_vb,
                    file_name=f"pvt_vb_api{api:.0f}_T{t_f:.0f}F.csv",
                    mime="text/csv",
                )

        # ── μ_dead info ───────────────────────────────────────────────────
        from src.services.pvt_correlations import br_mu_dead as _mu_dead
        mu_dead_val = _mu_dead(float(api), float(t_f))
        st.divider()
        st.caption(
            f"μ_dead (Beggs-Robinson): **{mu_dead_val:.2f} cp** | "
            f"γo: **{gamma_o:.4f}** | "
            f"SG gas: **{gamma_g:.3f}** | "
            f"T: **{t_f:.0f} °F**"
        )


# ---------------------------------------------------------------------------
# Embedded entry point (called from app.py hub)
# ---------------------------------------------------------------------------

def render_m2_embedded(well_id: str) -> None:
    """Render M2 PVT editor embedded in the hub (no set_page_config)."""
    st.subheader("🧪 M2 — Propiedades PVT")
    st.caption(
        "Correlaciones Standing (1947) y Vasquez-Beggs (1980) para Rs/Bo "
        "y Beggs-Robinson (1975) para viscosidad. "
        "Temperatura en **°F**, presión en **psia**."
    )
    _pvt_core_ui()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="ecoRTA M2 — PVT",
        page_icon="🧪",
        layout="wide",
    )
    st.title("🧪 ecoRTA M2 — Propiedades PVT")
    st.caption(
        "Correlaciones Standing (1947) y Vasquez-Beggs (1980) para Rs/Bo "
        "y Beggs-Robinson (1975) para viscosidad. "
        "Temperatura en **°F**, presión en **psia**."
    )
    _pvt_core_ui()


if __name__ == "__main__":
    main()
