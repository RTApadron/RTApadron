
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Helpers and correlations
# -----------------------------
def api_to_sgo(API):
    # Stock-tank oil specific gravity (relative to water at 60 F)
    return 141.5 / (API + 131.5)

def standing_rs(P_psia, API, sg_gas, T_F):
    # Standing (1947) Rs for P <= Pb (scf/STB)
    # Rs = sg_gas * [ ((P + 1.4)/18.2) * 10^(0.0125*API - 0.00091*T) ]^1.2048
    A = ((P_psia + 1.4) / 18.2) * (10 ** (0.0125 * API - 0.00091 * T_F))
    return sg_gas * (A ** 1.2048)

def standing_pb(Rs_scfstb, API, sg_gas, T_F):
    # Standing bubble point from Rs
    # Pb = 18.2 * (Rs/sg_gas)^0.83 * 10^(0.00091*T - 0.0125*API) - 1.4
    return 18.2 * ((Rs_scfstb / sg_gas) ** 0.83) * (10 ** (0.00091 * T_F - 0.0125 * API)) - 1.4

def standing_bo_sat(Rs_scfstb, API, sg_gas, T_F):
    # Standing saturated Bo (bbl/STB)
    # Bo = 0.9759 + 0.00012 * ( Rs * (sg_gas/sg_o)**0.5 + 1.25*T )^1.2
    sg_o = api_to_sgo(API)
    term = Rs_scfstb * ((sg_gas / sg_o) ** 0.5) + 1.25 * T_F
    return 0.9759 + 0.00012 * (term ** 1.2)

def vb_rs(P_psia, API, sg_gas, T_F):
    # Vasquez & Beggs (1980) Rs (approximate implementation without separator corrections)
    # Region constants by API
    if API <= 30:
        C1, C2, C3 = 0.0362, 1.0937, 25.7240
    else:
        C1, C2, C3 = 0.0178, 1.1870, 23.9310
    # Rs = (C1 * sg_gas * P^C2) * 10^(C3 * API / (T_F + 460))
    return (C1 * sg_gas * (P_psia ** C2)) * (10 ** (C3 * API / (T_F + 460.0)))

def glaso_rs(P_psia, API, sg_gas, T_F):
    # Glaso (1980) Rs (scf/STB) using common published form
    # Convert units as needed per published correlation
    # y_g = sg_gas
    api = API
    y_g = sg_gas
    t = T_F
    p = P_psia
    # Glaso introduces intermediate variables:
    # F = 10^(0.00091*T - 0.0125*API)
    F = 10 ** (0.00091 * t - 0.0125 * api)
    # Rs = 10^(X), where
    # X = 2.8869 - 14.1811*( (p / (y_g * F)) ** 0.83 ) + 3.3093 * ( (p / (y_g * F)) ** (0.83**2) )
    # NOTE: Literature has variants; this is a commonly cited approximation to produce reasonable curves.
    # To avoid negative or nonsensical values, guard inputs:
    denom = max(y_g * F, 1e-9)
    Z = (p / denom) ** 0.83
    X = 2.8869 + 1.3604 * np.log10(Z) + 0.2588 * (np.log10(Z) ** 2)  # alternative smooth form seen in practice
    return 10 ** X

def beggs_robinson_mu_dead(API, T_F):
    # Dead oil viscosity (cP): mu_od = 10^x - 1, x = T^{-1.163} * exp(13.108 - 6.591/SG_o)
    SG_o = api_to_sgo(API)
    x = (T_F ** (-1.163)) * np.exp(13.108 - 6.591 / SG_o)
    return (10 ** x) - 1.0

def beggs_robinson_mu_sat(Rs_scfstb, mu_dead):
    # Saturated oil viscosity (<= Pb): mu_os = A * mu_od^B
    A = 10.715 * (Rs_scfstb + 100.0) ** (-0.515)
    B = 5.44 * (Rs_scfstb + 150.0) ** (-0.338)
    return A * (mu_dead ** B)

def beggs_robinson_mu_undersat(mu_sat, P_psia, Pb_psia):
    # Undersaturated (> Pb): mu = mu_sat * (P/Pb)^m, m = 2.6*P^1.187*exp(-(11.513 + 8.98e-5*P))
    # Use Pb as reference pressure in the exponent term as per common implementation guidance.
    m = 2.6 * (P_psia ** 1.187) * np.exp(-(11.513 + 8.98e-5 * P_psia))
    return mu_sat * (P_psia / Pb_psia) ** m

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Módulo 2 — PVT (tesis)", layout="wide")
st.title("Módulo 2 — PVT (Standing, Vasquez–Beggs, Glaso)")
st.caption("Comparación de correlaciones y ajuste básico a datos de laboratorio")

with st.sidebar:
    st.header("Parámetros del fluido")
    API    = st.number_input("API del crudo", 5.0, 60.0, 32.0, 0.1, key="api")
    sg_gas = st.number_input("Gravedad específica del gas (air=1.0)", 0.55, 1.20, 0.75, 0.01, key="sg_gas")
    T_F    = st.number_input("Temperatura (°F)", 60.0, 350.0, 180.0, 1.0, key="temp_f")

    st.header("Rango de presión")
    pmin = st.number_input("P mínima (psia)", 14.7, 10000.0, 100.0, 10.0, key="pmin")
    pmax = st.number_input("P máxima (psia)", 100.0, 15000.0, 5000.0, 50.0, key="pmax")
    npts = st.slider("Puntos", 20, 200, 80, 5, key="npts")

    st.divider()
    st.header("Modo Pb / Rsb")
    mode = st.radio("¿Qué quieres fijar?",
                    ["Derivar Pb desde Rsb", "Derivar Rsb desde Pb"],
                    index=0, key="mode_pb_rsb")

    if mode == "Derivar Pb desde Rsb":
        Rsb_input = st.number_input("Rsb @ Pb (scf/STB)", 0.0, 5000.0, 600.0, 10.0, key="rsb_input")
        Pb_input = None
    else:
        Pb_input = st.number_input("Pb (psia)", 100.0, 15000.0, 2500.0, 50.0, key="pb_input")
        Rsb_input = None

    co = st.number_input("Compresibilidad aceite cₒ (>Pb) [1/psi]",
                         0.0, 5e-4, 1.2e-5, 1e-6, format="%.1e", key="co_input")
    add_usat = st.checkbox("Calcular propiedades > Pb (usar cₒ y μo>Pb)", value=True, key="add_usat")

    st.divider()
    st.header("Correlaciones a mostrar")
    show_standing = st.checkbox("Standing (Rs, Bo)", value=True, key="show_standing")
    show_vb       = st.checkbox("Vasquez–Beggs (Rs, Bo sat)", value=True, key="show_vb")
    show_glaso    = st.checkbox("Glaso (Rs)", value=False, key="show_glaso")
    show_mu       = st.checkbox("Viscosidad (Beggs–Robinson)", value=True, key="show_mu")
    normalize_at_pb = st.checkbox("Normalizar Rs en Pb a Rsb (recomendado)", value=True, key="norm_pb")


# Malla de presión
P = np.linspace(pmin, pmax, int(npts))

# Cálculo de Rs por correlación
Rs_all = {}

if show_standing:
    Rs_st = standing_rs(P, API, sg_gas, T_F)
    Rs_all[("Rs", "Standing")] = Rs_st

if show_vb:
    Rs_vb = vb_rs(P, API, sg_gas, T_F)
    Rs_all[("Rs", "Vasquez–Beggs")] = Rs_vb

if show_glaso:
    Rs_gl = glaso_rs(P, API, sg_gas, T_F)
    Rs_all[("Rs", "Glaso")] = Rs_gl
# Elegir una Base para Pb (prioridad: Standing, luego Vasquez–Beggs, luego Glaso)
base_key = None
for k in [("Rs","Standing"), ("Rs","Vasquez–Beggs"), ("Rs","Glaso")]:
    if k in Rs_all:
        base_key = k
        break
if base_key is None:
    st.error("Selecciona al menos una correlación de Rs.")
    st.stop()

Rs_base = Rs_all[base_key]  # array sobre P

if mode == "Derivar Pb desde Rsb":
    Rsb = float(Rsb_input)
    Pb  = float(standing_pb(Rsb, API, sg_gas, T_F))  # Standing: Pb(Rsb)
else:
    Pb  = float(Pb_input)
    # Rsb desde la correlación base seleccionada
    if base_key[1] == "Standing":
        Rsb = float(standing_rs(Pb, API, sg_gas, T_F))
    elif base_key[1] == "Vasquez–Beggs":
        Rsb = float(vb_rs(Pb, API, sg_gas, T_F))
    else:
        Rsb = float(glaso_rs(Pb, API, sg_gas, T_F))

# Rs piecewise: <=Pb correlación; >Pb congelado en Rsb
Rs_pw = {}
for (prop, name), arr in Rs_all.items():
    if prop != "Rs":
        continue
    # valor que predice ESTA correlación justo en Pb
    arr_pb = float(np.interp(Pb, P, arr))
    # escala para que Rs(Pb) = Rsb
    if normalize_at_pb and np.isfinite(arr_pb) and arr_pb > 0:
        scale = Rsb / arr_pb
        arr_sat = arr * scale        # solo afecta P <= Pb
    else:
        arr_sat = arr
    arr_piece = np.where(P <= Pb, arr_sat, Rsb)
    Rs_pw[(prop, name)] = arr_piece


# Bubble point estimado (Standing a partir de Rs(P))
if show_standing:
    # Estimar Pb como la presión donde Rs_st tiende a máximo razonable (monótona creciente)
    # Aquí tomamos Pb como pmax si solo crece, para fines de graficación
    Rs_st_vals = Rs_st
    # Definir Pb_st como la presión donde la derivada cae por debajo de un umbral (simple)
    dRs = np.gradient(Rs_st_vals, P)
    try:
        idx_pb = np.where(dRs < 1e-3)[0][0]
        Pb_est = P[idx_pb]
    except IndexError:
        Pb_est = P[-1]
else:
    Pb_est = None

# Bo: saturado hasta Pb; >Pb con compresibilidad cₒ
Bo_series = {}
for (prop, name), arr in Rs_pw.items():
    if prop != "Rs":
        continue
    Bo_sat_curve = standing_bo_sat(np.where(P <= Pb, arr, Rsb), API, sg_gas, T_F)
    if add_usat:
        Bo_b = standing_bo_sat(Rsb, API, sg_gas, T_F)
        Bo_usat = Bo_b * np.exp(co * (Pb - P))  # >Pb
        Bo_total = np.where(P <= Pb, Bo_sat_curve, Bo_usat)
        Bo_series[("Bo", f"{name} (sat/undersat)")] = Bo_total
    else:
        Bo_series[("Bo", f"{name} (sat)")] = Bo_sat_curve

# μo: <=Pb saturado (Beggs–Robinson); >Pb extensión desde μ(Pb)
mu_series = None
if show_mu:
    mu_dead = beggs_robinson_mu_dead(API, T_F)
    mu_dict = {}
    for (prop, name), arr in Rs_pw.items():
        if prop != "Rs":
            continue
        mu_sat_curve = beggs_robinson_mu_sat(np.where(P <= Pb, arr, Rsb), mu_dead)
        if add_usat:
            mu_b = beggs_robinson_mu_sat(Rsb, mu_dead)
            mu_usat = beggs_robinson_mu_undersat(mu_b, P, Pb)
            mu_total = np.where(P <= Pb, mu_sat_curve, mu_usat)
        else:
            mu_total = mu_sat_curve
        mu_dict[name] = mu_total
    mu_series = mu_dict.get(base_key[1])  # muestra la base elegida


# -----------------------------
# Datos de laboratorio y ajuste simple
# -----------------------------
st.subheader("Datos de laboratorio (opcional)")
st.caption("Sube un CSV con columnas: P_psia, Rs_scfstb, Bo_rbblstb, mu_cp (puedes omitir alguna).")
uploaded = st.file_uploader("Cargar CSV de PVT (diferential liberation)", type=["csv"])

lab_df = None
if uploaded is not None:
    try:
        lab_df = pd.read_csv(uploaded)
        st.dataframe(lab_df.head(20))
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")

def grid_tune(pred, truth, bounds=(0.8, 1.2), steps=41):
    # Ajuste multiplicativo simple para minimizar RMSE
    if truth is None or pred is None:
        return 1.0, np.nan
    mask = np.isfinite(truth) & np.isfinite(pred)
    if mask.sum() < 2:
        return 1.0, np.nan
    s_vals = np.linspace(bounds[0], bounds[1], steps)
    best_s, best_rmse = 1.0, 1e99
    for s in s_vals:
        err = rmse(truth[mask], s * pred[mask])
        if err < best_rmse:
            best_rmse, best_s = err, s
    return best_s, best_rmse

# -----------------------------
# Gráficas
# -----------------------------
st.subheader("Curvas de correlaciones")
col1, col2, col3 = st.columns(3)

# Rs
with col1:
    fig1, ax1 = plt.subplots()
    for (prop, name), arr in Rs_pw.items():
        if prop == "Rs":
            ax1.plot(P, arr, label=name)
    if lab_df is not None and "Rs_scfstb" in lab_df.columns and "P_psia" in lab_df.columns:
        ax1.scatter(lab_df["P_psia"], lab_df["Rs_scfstb"], marker="o", label="Lab Rs")
    ax1.set_xlabel("Presión (psia)")
    ax1.set_ylabel("Rs (scf/STB)")
    ax1.set_title("Solución GOR (Rs)")
    ax1.legend()
    st.pyplot(fig1)

# Bo
with col2:
    fig2, ax2 = plt.subplots()
    for (prop, name), arr in Bo_series.items():
        if prop == "Bo":
            ax2.plot(P, arr, label=name)
    if lab_df is not None and "Bo_rbblstb" in lab_df.columns and "P_psia" in lab_df.columns:
        ax2.scatter(lab_df["P_psia"], lab_df["Bo_rbblstb"], marker="s", label="Lab Bo")
    ax2.set_xlabel("Presión (psia)")
    ax2.set_ylabel("Bo (bbl/STB)")
    ax2.set_title("Factor Volumétrico del Aceite (Bo) — saturado")
    ax2.legend()
    st.pyplot(fig2)

# mu
with col3:
    fig3, ax3 = plt.subplots()
    if mu_series is not None:
        ax3.plot(P, mu_series, label="Beggs–Robinson")
    if lab_df is not None and "mu_cp" in lab_df.columns and "P_psia" in lab_df.columns:
        ax3.scatter(lab_df["P_psia"], lab_df["mu_cp"], marker="^", label="Lab μo")
    ax3.set_xlabel("Presión (psia)")
    ax3.set_ylabel("Viscosidad (cP)")
    ax3.set_title("Viscosidad del Aceite (μo)")
    ax3.legend()
    st.pyplot(fig3)
def draw_pb(ax, Pb_value):
    ylim = ax.get_ylim()
    ax.axvline(Pb_value, linestyle="--", alpha=0.7)
    ax.text(Pb_value, ylim[1]*0.95, f"Pb ≈ {Pb_value:.0f} psia", rotation=90, va="top")

# ... después de plotear cada figura:
draw_pb(ax1, Pb)  # Rs
draw_pb(ax2, Pb)  # Bo
draw_pb(ax3, Pb)  # μo

# -----------------------------
# Ajuste básico (factor de escala)
# -----------------------------
st.subheader("Ajuste a datos de laboratorio (factor multiplicativo)")
colA, colB, colC = st.columns(3)

metrics_rows = []

if lab_df is not None and "P_psia" in lab_df.columns:
    P_lab = lab_df["P_psia"].values
    # Interpolar predicciones en P_lab
    def interp_safe(x, y, xq):
        return np.interp(xq, x, y, left=np.nan, right=np.nan)

    # Rs
    with colA:
        if "Rs_scfstb" in lab_df.columns and len(Rs_all) > 0:
            st.write("**Rs**")
            for (prop, name), arr in Rs_all.items():
                if prop != "Rs":
                    continue
                pred = interp_safe(P, arr, P_lab)
                truth = lab_df["Rs_scfstb"].values
                s, err = grid_tune(pred, truth)
                mape_val = mape(truth, s * pred)
                st.text(f"{name}: escala óptima ≈ {s:.3f}  |  RMSE ≈ {err:.2f}  |  MAPE ≈ {mape_val:.1f}%")
                metrics_rows.append(["Rs", name, s, err, mape_val])

    # Bo
    with colB:
        if "Bo_rbblstb" in lab_df.columns and len(Bo_series) > 0:
            st.write("**Bo**")
            for (prop, name), arr in Bo_series.items():
                if prop != "Bo":
                    continue
                pred = interp_safe(P, arr, P_lab)
                truth = lab_df["Bo_rbblstb"].values
                s, err = grid_tune(pred, truth)
                mape_val = mape(truth, s * pred)
                st.text(f"{name}: escala óptima ≈ {s:.3f}  |  RMSE ≈ {err:.4f}  |  MAPE ≈ {mape_val:.2f}%")
                metrics_rows.append(["Bo", name, s, err, mape_val])

    # mu
    with colC:
        if "mu_cp" in lab_df.columns and mu_series is not None:
            st.write("**μo (Beggs–Robinson)**")
            pred = interp_safe(P, mu_series, P_lab)
            truth = lab_df["mu_cp"].values
            s, err = grid_tune(pred, truth, bounds=(0.6, 1.4))
            mape_val = mape(truth, s * pred)
            st.text(f"BR-75: escala óptima ≈ {s:.3f}  |  RMSE ≈ {err:.3f}  |  MAPE ≈ {mape_val:.1f}%")
            metrics_rows.append(["mu", "Beggs–Robinson", s, err, mape_val])

# Exportar reporte
if len(metrics_rows) > 0:
    rep = pd.DataFrame(metrics_rows, columns=["Propiedad", "Correlación", "Escala_opt", "RMSE", "MAPE_%"])
    st.download_button(
        "Descargar reporte de ajuste (CSV)",
        rep.to_csv(index=False).encode("utf-8"),
        file_name="pvt_fit_report.csv",
        mime="text/csv",
    )

st.divider()
st.markdown("**Notas:** Esta herramienta compara correlaciones clásicas para crudo negro y permite un ajuste multiplicativo simple a datos de laboratorio. Para usos críticos, valida con tus PVT medidos y ten en cuenta los rangos de validez publicados de cada correlación.")
