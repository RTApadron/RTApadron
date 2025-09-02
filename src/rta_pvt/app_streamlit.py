# src/rta_pvt/app_streamlit.py
import io
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from pvt_tools import (
    PVTInputs,
    compute_pvt_table,
    gamma_o_from_api,
    a_standing,
    pb_standing,
    rs_standing,    # usaremos esto cuando el usuario ingrese Pb
)

st.set_page_config(page_title="Sprint 1 - PVT", layout="wide")

st.title("🛠️ Sprint 1 - PVT interactivo (Standing + Beggs-Robinson)")
st.caption("Ajusta los inputs y genera curvas Rs, Bo y μo con anotación de Pb, más exportación a CSV/XLSX.")

# --- Sidebar: Inputs ---
st.sidebar.header("Inputs del modelo")

col_api, col_gas = st.sidebar.columns(2)
API = col_api.number_input("API (°API)", value=30.0, min_value=5.0, max_value=60.0, step=0.5)
gamma_g = col_gas.number_input("γ_g (aire=1)", value=0.80, min_value=0.55, max_value=1.20, step=0.01)

T_F = st.sidebar.number_input("Temperatura (°F)", value=200.0, min_value=60.0, max_value=350.0, step=5.0)

st.sidebar.markdown("---")
modo = st.sidebar.radio("Modo de burbujeo", options=["Ingresar Rsb", "Ingresar Pb"], index=0)

if modo == "Ingresar Rsb":
    Rsb = st.sidebar.number_input("Rsb (scf/STB)", value=600.0, min_value=0.0, max_value=3000.0, step=25.0)
    Pb = pb_standing(API, T_F, gamma_g, Rsb)
else:
    Pb = st.sidebar.number_input("Pb (psia)", value=3000.0, min_value=100.0, max_value=10000.0, step=50.0)
    # Invertimos Standing: a Pb, Rs = Rsb
    Rsb = rs_standing(Pb, API, T_F, gamma_g)

st.sidebar.markdown("---")
pmin = st.sidebar.number_input("p mínima (psia)", value=500, min_value=10, step=50)
pmax = st.sidebar.number_input("p máxima (psia)", value=4500, min_value=pmin+50, step=50)
step = st.sidebar.number_input("Δp (psia)", value=250, min_value=10, step=10)

presiones = list(range(int(pmin), int(pmax) + int(step), int(step)))

# Resumen rápido
gamma_o = gamma_o_from_api(API)
a_val = a_standing(API, T_F)

with st.expander("📋 Resumen de entradas / parámetros", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("API (°API)", f"{API:.2f}")
    c2.metric("γ_g (aire=1)", f"{gamma_g:.3f}")
    c3.metric("T (°F)", f"{T_F:.1f}")
    c1.metric("γ_o (agua=1)", f"{gamma_o:.4f}")
    c2.metric("a (Standing)", f"{a_val:.5f}")
    c3.metric("Pb (psia)", f"{Pb:.2f}")
    st.caption(f"Modo: **{modo}** | Rsb usado: **{Rsb:.2f} scf/STB** | p: {pmin}–{pmax} (step={step})")

# --- Computo tabla (incluye punto en Pb) ---
inputs = PVTInputs(API=API, gamma_g=gamma_g, T_F=T_F, Rsb=Rsb, pressures=presiones)
df = pd.DataFrame(compute_pvt_table(inputs))

# --- Mostrar tabla ---
st.subheader("Tabla de propiedades PVT")
st.dataframe(df, use_container_width=True)

# --- Gráficos (con anotación de Pb) ---
def plot_with_pb(x, y, xlabel, ylabel, title, pb_value):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o")
    ax.axvline(pb_value, linestyle="--", color="red", label=f"Pb = {pb_value:.1f} psia")

    ymin, ymax = float(np.min(y)), float(np.max(y))
    y_text = ymin + 0.05 * (ymax - ymin)
    ax.annotate(
        f"Pb = {pb_value:.1f} psia",
        xy=(pb_value, y_text),
        xytext=(pb_value + 0.06 * (max(x) - min(x)), y_text),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9,
        color="red",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

st.subheader("Gráficas")
cA, cB, cC = st.columns(3)
with cA:
    plot_with_pb(df["pressure_psi"], df["Rs_scf_per_STB"], "Presión (psia)", "Rs (scf/STB)", "Rs vs Presión", Pb)
with cB:
    plot_with_pb(df["pressure_psi"], df["Bo_bbl_per_STB"], "Presión (psia)", "Bo (bbl/STB)", "Bo vs Presión", Pb)
with cC:
    log_y = st.checkbox("Usar escala logarítmica en μo", value=False)
    fig_mu, ax_mu = plt.subplots()
    ax_mu.plot(df["pressure_psi"], df["mu_o_cP"], marker="o")
    ax_mu.axvline(Pb, linestyle="--", color="red", label=f"Pb = {Pb:.1f} psia")
    ymin, ymax = float(np.min(df["mu_o_cP"])), float(np.max(df["mu_o_cP"]))
    y_text = ymin + 0.05 * (ymax - ymin)
    ax_mu.annotate(
        f"Pb = {Pb:.1f} psia",
        xy=(Pb, y_text),
        xytext=(Pb + 0.06 * (max(df['pressure_psi']) - min(df['pressure_psi'])), y_text),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9,
        color="red",
    )
    ax_mu.set_xlabel("Presión (psia)")
    ax_mu.set_ylabel("Viscosidad (cP)")
    ax_mu.set_title("μo vs Presión")
    ax_mu.legend()
    if log_y:
        ax_mu.set_yscale("log")
    st.pyplot(fig_mu, clear_figure=True)

# --- Descargas ---
st.subheader("Descargar resultados")

# CSV
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar CSV",
    data=csv_bytes,
    file_name="pvt_streamlit_results.csv",
    mime="text/csv"
)

# XLSX (en memoria)
try:
    import xlsxwriter  # noqa: F401
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="PVT", index=False)
    st.download_button(
        "⬇️ Descargar Excel (XLSX)",
        data=output.getvalue(),
        file_name="pvt_streamlit_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
except Exception as e:
    st.info("Para exportar a Excel, instala `xlsxwriter` (pip install xlsxwriter).")
