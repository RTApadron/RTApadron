
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, os, io, datetime

# ----------------------------------
# Utilidades y correlaciones base
# ----------------------------------
def api_to_sgo(API):
    """Convierte °API a gravedad específica del aceite a 60°F (aire=1)."""
    return 141.5 / (API + 131.5)

def standing_rs(P_psia, API, sg_gas, T_F):
    A = ((P_psia + 1.4) / 18.2) * (10 ** (0.0125 * API - 0.00091 * T_F))
    return sg_gas * (A ** 1.2048)

def standing_pb(Rs_scfstb, API, sg_gas, T_F):
    return 18.2 * ((Rs_scfstb / sg_gas) ** 0.83) * (10 ** (0.00091 * T_F - 0.0125 * API)) - 1.4

def standing_bo_sat(Rs_scfstb, API, sg_gas, T_F):
    sg_o = api_to_sgo(API)
    term = Rs_scfstb * ((sg_gas / sg_o) ** 0.5) + 1.25 * T_F
    return 0.9759 + 0.00012 * (term ** 1.2)

def vb_rs(P_psia, API, sg_gas, T_F):
    if API <= 30:
        C1, C2, C3 = 0.0362, 1.0937, 25.7240
    else:
        C1, C2, C3 = 0.0178, 1.1870, 23.9310
    return (C1 * sg_gas * (P_psia ** C2)) * (10 ** (C3 * API / (T_F + 460.0)))

def glaso_rs(P_psia, API, sg_gas, T_F):
    api = API
    y_g = sg_gas
    t = T_F
    p = P_psia
    F = 10 ** (0.00091 * t - 0.0125 * api)
    denom = np.maximum(y_g * F, 1e-12)
    Z = (p / denom) ** 0.83
    X = 2.8869 + 1.3604 * np.log10(Z) + 0.2588 * (np.log10(Z) ** 2)
    return 10 ** X

# --- Viscosidad: modelos de "aceite muerto" ---
def mu_dead_beggs_robinson(API, T_F):
    """
    Beggs & Robinson (1975): μ_dead = 10^x - 1
    x = T_F^(-1.163) * exp(13.108 - 6.591/SG_o)
    """
    SG_o = api_to_sgo(API)
    x = (T_F ** (-1.163)) * np.exp(13.108 - 6.591 / SG_o)
    return (10 ** x) - 1.0

def mu_dead_beal(API, T_F):
    """
    Beal (1946/Standing form). T en °F, API en °API.
    μ_dead = [ 0.32 + (1.8e7)/API^4.53 ] * [ 360 / (T + 200) ]^a
    con a = 10^( 0.43 + 8.33/API )
    """
    API = np.maximum(API, 1e-6)
    a = 10 ** (0.43 + 8.33 / API)
    return (0.32 + (1.8e7) / (API ** 4.53)) * ((360.0 / (T_F + 200.0)) ** a)

def mu_dead_glaso(API, T_F):
    """
    Glasø (1980) para dead oil. T en °R (T_F+460).
    μ_dead = C * 10^{ D * log10(γ_o) }
    C = 3.141e10 * T_R^(-3.444)
    D = 10.313 * log10(T_R) - 36.447
    γ_o = gravedad específica del aceite (API -> SG_o)
    """
    T_R = T_F + 460.0
    SG_o = api_to_sgo(API)
    C = 3.141e10 * (T_R ** (-3.444))
    D = 10.313 * np.log10(T_R) - 36.447
    return C * (10 ** (D * np.log10(SG_o)))

def mu_dead_andrade(T_F, A, B):
    """
    Andrade (genérica): μ = A * exp(B / T_K)
    T_K en Kelvin.
    """
    T_K = (T_F - 32.0) * 5.0/9.0 + 273.15
    return A * np.exp(B / T_K)

def mu_sat_beggs_robinson(Rs_scfstb, mu_dead):
    A = 10.715 * (Rs_scfstb + 100.0) ** (-0.515)
    B = 5.44   * (Rs_scfstb + 150.0) ** (-0.338)
    return A * (mu_dead ** B)

def mu_undersat_from_pb(mu_b, P_psia, Pb_psia):
    m = 2.6 * (P_psia ** 1.187) * np.exp(-(11.513 + 8.98e-5 * P_psia))
    return mu_b * (P_psia / Pb_psia) ** m

# --- Métricas y utilidades ---
def rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0: return np.nan
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    if mask.sum() == 0: return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def grid_tune(pred, truth, bounds=(0.6, 1.6), steps=41):
    mask = np.isfinite(truth) & np.isfinite(pred)
    if mask.sum() < 2:
        return 1.0, np.nan, np.nan
    s_vals = np.linspace(bounds[0], bounds[1], steps)
    best = (1.0, 1e99, 1e99)
    for s in s_vals:
        r = rmse(truth[mask], s * pred[mask])
        m = mape(truth[mask], s * pred[mask])
        if r < best[1]:
            best = (s, r, m)
    return best

def draw_pb(ax, Pb_value, label="Pb"):
    ylim = ax.get_ylim()
    ax.axvline(Pb_value, linestyle="--", alpha=0.7)
    ax.text(Pb_value, ylim[1]*0.95, f"{label} ≈ {Pb_value:.0f} psia", rotation=90, va="top")

# ----------------------------------
# App
# ----------------------------------
st.set_page_config(page_title="Módulo 2 — PVT (v5)", layout="wide")
st.title("ecoRTA Módulo 2 — PVT")

with st.sidebar:
    st.header("Parámetros del fluido")
    API    = st.number_input("API del crudo", 5.0, 60.0, 32.0, 0.1, key="api")
    sg_gas = st.number_input("Gravedad específica del gas (air=1.0)", 0.55, 1.20, 0.75, 0.01, key="sg_gas")
    T_F    = st.number_input("Temperatura (°F)", 60.0, 350.0, 180.0, 1.0, key="temp_f")

    st.header("Rango de presión")
    pmin = st.number_input("P mínima (psia)", 14.7, 10000.0, 50.0, 10.0, key="pmin")
    pmax = st.number_input("P máxima (psia)", 100.0, 20000.0, 5000.0, 50.0, key="pmax")
    npts = st.slider("Puntos", 20, 400, 100, 5, key="npts")

    st.divider()
    st.header("Definir Pb/Rsb")
    mode = st.radio("¿Qué quieres definir?",
                    ["Rsb", "Pb"],
                    index=0, key="mode_pb_rsb")
    if mode == "Rsb":
        Rsb_input = st.number_input("Rsb en Pb (scf/STB)", 0.0, 5000.0, 600.0, 10.0, key="rsb_input")
        Pb_input = None
    else:
        Pb_input = st.number_input("Pb (psia)", 100.0, 15000.0, 2500.0, 50.0, key="pb_input")
        Rsb_input = None

    normalize_at_pb = st.checkbox("Normalizar Rs en Pb a Rsb", value=True, key="norm_pb")
    co = st.number_input("Compresibilidad aceite cₒ (>Pb) [1/psi]",
                         0.0, 5e-4, 1.2e-5, 1e-6, format="%.1e", key="co_input")
    add_usat = st.checkbox("Calcular propiedades > Pb", value=True, key="add_usat")

    st.divider()
    st.header("Correlaciones a mostrar")
    show_standing = st.checkbox("Standing (Rs, Bo)", value=True, key="show_standing")
    show_vb       = st.checkbox("Vasquez–Beggs (Rs, Bo sat)", value=True, key="show_vb")
    show_glaso    = st.checkbox("Glaso (Rs)", value=True, key="show_glaso")

    st.divider()
    st.header("Modelo de viscosidad μo")
    visc_model = st.selectbox("Selecciona el modelo de μ_dead",
                              ["Beggs–Robinson","Beal","Glasø (dead oil)","Andrade"],
                              index=0, key="visc_model")
    # Parámetros específicos
    with st.expander("Parámetros del modelo de viscosidad"):
        if visc_model == "Andrade":
            A_and = st.number_input("A (cP)", 0.001, 1e6, 1.5, step=0.1, key="A_and")
            B_and = st.number_input("B (K)",   10.0,  2e4, 1800.0, step=10.0, key="B_and")
        else:
            A_and = None; B_and = None

    st.divider()
    st.header("Autofit / Configuración")
    st.caption("Base para Rs/μo cuando defines Rsb desde Pb:")
    base_for_rsb_from_pb = st.selectbox("Base para Rsb(Pb) + μo",
                                        ["Standing", "Vasquez–Beggs", "Glaso"], index=0)
    do_autofit = st.button("🔧 Ejecutar Autofit con datos")
    accept_fit = st.button("✅ Aceptar ajuste (guardar JSON)")
    reset_fit  = st.button("♻️ Limpiar ajuste aceptado")

# ----------------------------------
# Malla de presión y Rs
# ----------------------------------
P = np.linspace(pmin, pmax, int(npts))

Rs_all = {}
if show_standing:
    Rs_all[("Rs", "Standing")] = standing_rs(P, API, sg_gas, T_F)
if show_vb:
    Rs_all[("Rs", "Vasquez–Beggs")] = vb_rs(P, API, sg_gas, T_F)
if show_glaso:
    Rs_all[("Rs", "Glaso")] = glaso_rs(P, API, sg_gas, T_F)

# base_key = primera disponible
base_key = None
for k in [("Rs","Standing"), ("Rs","Vasquez–Beggs"), ("Rs","Glaso")]:
    if k in Rs_all:
        base_key = k
        break
if base_key is None:
    st.error("Selecciona al menos una correlación de Rs.")
    st.stop()

# Derivar Pb/Rsb
if mode == "Rsb":
    Rsb = float(Rsb_input)
    Pb  = float(standing_pb(Rsb, API, sg_gas, T_F))  # Standing para Pb(Rsb)
else:
    Pb = float(Pb_input)
    if base_for_rsb_from_pb == "Standing":
        Rsb = float(standing_rs(Pb, API, sg_gas, T_F)); base_key = ("Rs","Standing")
    elif base_for_rsb_from_pb == "Vasquez–Beggs":
        Rsb = float(vb_rs(Pb, API, sg_gas, T_F));       base_key = ("Rs","Vasquez–Beggs")
    else:
        Rsb = float(glaso_rs(Pb, API, sg_gas, T_F));    base_key = ("Rs","Glaso")

if not (pmin < Pb < pmax):
    st.warning(f"Pb={Pb:.0f} psia está fuera del rango; ajusta pmin/pmax.")

# Rs piecewise con normalización en Pb
Rs_pw = {}
for (prop, name), arr in Rs_all.items():
    if prop != "Rs": continue
    arr_pb = float(np.interp(Pb, P, arr))
    if normalize_at_pb and np.isfinite(arr_pb) and arr_pb > 0:
        arr_sat = arr * (Rsb / arr_pb)
    else:
        arr_sat = arr
    Rs_pw[(prop, name)] = np.where(P <= Pb, arr_sat, Rsb)

# Bo (Standing sat + compresión >Pb)
Bo_series = {}
for (prop, name), arr in Rs_pw.items():
    if prop != "Rs": continue
    Bo_sat_curve = standing_bo_sat(np.where(P <= Pb, arr, Rsb), API, sg_gas, T_F)
    if add_usat:
        Bo_b = standing_bo_sat(Rsb, API, sg_gas, T_F)
        Bo_usat = Bo_b * np.exp(co * (Pb - P))
        Bo_total = np.where(P <= Pb, Bo_sat_curve, Bo_usat)
        Bo_series[("Bo", f"{name} (sat/undersat)")] = Bo_total
    else:
        Bo_series[("Bo", f"{name} (sat)")] = Bo_sat_curve

# μo: seleccionar modelo de μ_dead y encadenar BR para ≤Pb
def compute_mu_series(visc_model, API, T_F, Rs_curve, Rsb, P, Pb, A_and=None, B_and=None):
    # μ_dead
    if visc_model == "Beggs–Robinson":
        mu_dead = mu_dead_beggs_robinson(API, T_F)
    elif visc_model == "Beal":
        mu_dead = mu_dead_beal(API, T_F)
    elif visc_model == "Glasø (dead oil)":
        mu_dead = mu_dead_glaso(API, T_F)
    elif visc_model == "Andrade":
        A_val = 1.5 if A_and is None else A_and
        B_val = 1800.0 if B_and is None else B_and
        mu_dead = mu_dead_andrade(T_F, A_val, B_val)
    else:
        mu_dead = mu_dead_beggs_robinson(API, T_F)

    mu_sat_curve = mu_sat_beggs_robinson(np.where(P <= Pb, Rs_curve, Rsb), mu_dead)

    # >Pb anclado en μ(Pb)
    mu_b = mu_sat_beggs_robinson(Rsb, mu_dead)
    mu_usat = mu_undersat_from_pb(mu_b, P, Pb)
    return np.where(P <= Pb, mu_sat_curve, mu_usat), mu_dead

mu_series, mu_dead_value = compute_mu_series(visc_model, API, T_F,
                                             Rs_pw[("Rs", base_key[1])],
                                             Rsb, P, Pb, A_and, B_and)

# ----------------------------------
# Datos de laboratorio (CSV + editor)
# ----------------------------------
st.subheader("Datos de laboratorio")
c1, c2 = st.columns([1,1])
with c1:
    uploaded = st.file_uploader("Cargar CSV con columnas: P_psia, Rs_scfstb, Bo_rbblstb, mu_cp", type=["csv"])
with c2:
    use_editor = st.checkbox("Editar datos manualmente en tabla", value=True)

lab_df = None
template = pd.DataFrame({
    "P_psia":[500,1000,1500,2000,2500,3000],
    "Rs_scfstb":[np.nan]*6,
    "Bo_rbblstb":[np.nan]*6,
    "mu_cp":[np.nan]*6
})

if uploaded is not None:
    try:
        lab_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")

if use_editor:
    st.caption("Puedes editar la tabla; usa ‘+’ para agregar filas. Deja en blanco si no tienes datos.")
    lab_df = st.data_editor(lab_df if lab_df is not None else template, num_rows="dynamic", use_container_width=True, key="lab_editor")
elif lab_df is not None:
    st.dataframe(lab_df.head(30), use_container_width=True)

# ----------------------------------
# Gráficas principales
# ----------------------------------
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
    ax1.set_xlabel("Presión (psia)"); ax1.set_ylabel("Rs (scf/STB)"); ax1.set_title("Solución GOR (Rs)")
    ax1.legend(); draw_pb(ax1, Pb); st.pyplot(fig1)

# Bo
with col2:
    fig2, ax2 = plt.subplots()
    for (prop, name), arr in Bo_series.items():
        if prop == "Bo":
            ax2.plot(P, arr, label=name)
    if lab_df is not None and "Bo_rbblstb" in lab_df.columns and "P_psia" in lab_df.columns:
        ax2.scatter(lab_df["P_psia"], lab_df["Bo_rbblstb"], marker="s", label="Lab Bo")
    ax2.set_xlabel("Presión (psia)"); ax2.set_ylabel("Bo (bbl/STB)"); ax2.set_title("Factor Volumétrico del Aceite (Bo)")
    ax2.legend(); draw_pb(ax2, Pb); st.pyplot(fig2)

# μo
with col3:
    fig3, ax3 = plt.subplots()
    if mu_series is not None:
        ax3.plot(P, mu_series, label=f"μo ({visc_model})")
    if lab_df is not None and "mu_cp" in lab_df.columns and "P_psia" in lab_df.columns:
        ax3.scatter(lab_df["P_psia"], lab_df["mu_cp"], marker="^", label="Lab μo")
    ax3.set_xlabel("Presión (psia)"); ax3.set_ylabel("Viscosidad (cP)"); ax3.set_title("Viscosidad del Aceite (μo)")
    ax3.legend(); draw_pb(ax3, Pb); st.pyplot(fig3)

# ----------------------------------
# Autofit (escala multiplicativa)
# ----------------------------------
fit_result = {"s_rs":1.0, "s_bo":1.0, "s_mu":1.0, "base": base_key[1], "Pb": float(Pb), "Rsb": float(Rsb),
              "co": float(co), "normalize_at_pb": bool(normalize_at_pb), "visc_model": visc_model}

if lab_df is not None and "P_psia" in lab_df.columns:
    P_lab = lab_df["P_psia"].values

    def interp(x, y, xq):
        return np.interp(xq, x, y, left=np.nan, right=np.nan)

    base_name = base_key[1]
    rs_pred = interp(P, Rs_pw[("Rs", base_name)], P_lab) if ("Rs", base_name) in Rs_pw else None
    rs_truth = lab_df["Rs_scfstb"].values if "Rs_scfstb" in lab_df.columns else None
    if rs_pred is not None and rs_truth is not None:
        s_rs, rs_rmse, rs_mape = grid_tune(rs_pred, rs_truth, bounds=(0.6, 1.4), steps=41)
        fit_result.update({"s_rs": float(s_rs), "rs_rmse": float(rs_rmse), "rs_mape": float(rs_mape)})

    bo_key = None
    for (prop, name), arr in Bo_series.items():
        if prop=="Bo" and base_name in name:
            bo_key = (prop, name); break
    if bo_key and "Bo_rbblstb" in lab_df.columns:
        bo_pred = interp(P, Bo_series[bo_key], P_lab)
        s_bo, bo_rmse, bo_mape = grid_tune(bo_pred, lab_df["Bo_rbblstb"].values, bounds=(0.8, 1.2), steps=41)
        fit_result.update({"s_bo": float(s_bo), "bo_rmse": float(bo_rmse), "bo_mape": float(bo_mape)})

    if mu_series is not None and "mu_cp" in lab_df.columns:
        mu_pred = interp(P, mu_series, P_lab)
        s_mu, mu_rmse, mu_mape = grid_tune(mu_pred, lab_df["mu_cp"].values, bounds=(0.6, 1.6), steps=41)
        fit_result.update({"s_mu": float(s_mu), "mu_rmse": float(mu_rmse), "mu_mape": float(mu_mape)})

    if do_autofit:
        st.markdown("### Vista *Autofit* (curvas escaladas)")
        colA, colB, colC = st.columns(3)

        with colA:
            figA, axA = plt.subplots()
            axA.plot(P, Rs_pw[("Rs", base_name)] * fit_result["s_rs"], label=f"Rs ajustada ({base_name})")
            if "Rs_scfstb" in lab_df.columns:
                axA.scatter(lab_df["P_psia"], lab_df["Rs_scfstb"], marker="o", label="Lab Rs")
            axA.set_xlabel("Presión (psia)"); axA.set_ylabel("Rs (scf/STB)"); axA.set_title("Rs — Autofit")
            axA.legend(); draw_pb(axA, Pb); st.pyplot(figA)

        with colB:
            figB, axB = plt.subplots()
            axB.plot(P, Bo_series[bo_key] * fit_result["s_bo"], label=f"Bo ajustada ({base_name})")
            if "Bo_rbblstb" in lab_df.columns:
                axB.scatter(lab_df["P_psia"], lab_df["Bo_rbblstb"], marker="s", label="Lab Bo")
            axB.set_xlabel("Presión (psia)"); axB.set_ylabel("Bo (bbl/STB)"); axB.set_title("Bo — Autofit")
            axB.legend(); draw_pb(axB, Pb); st.pyplot(figB)

        with colC:
            figC, axC = plt.subplots()
            axC.plot(P, mu_series * fit_result["s_mu"], label=f"μo ajustada ({visc_model})")
            if "mu_cp" in lab_df.columns:
                axC.scatter(lab_df["P_psia"], lab_df["mu_cp"], marker="^", label="Lab μo")
            axC.set_xlabel("Presión (psia)"); axC.set_ylabel("Viscosidad (cP)"); axC.set_title("μo — Autofit")
            axC.legend(); draw_pb(axC, Pb); st.pyplot(figC)

        st.write("**Métricas del autofit**")
        cols = st.columns(3)
        with cols[0]:
            if "rs_rmse" in fit_result:
                st.text(f"Rs: escala ≈ {fit_result['s_rs']:.3f} | RMSE ≈ {fit_result['rs_rmse']:.3f} | MAPE ≈ {fit_result['rs_mape']:.2f}%")
        with cols[1]:
            if "bo_rmse" in fit_result:
                st.text(f"Bo: escala ≈ {fit_result['s_bo']:.3f} | RMSE ≈ {fit_result['bo_rmse']:.4f} | MAPE ≈ {fit_result['bo_mape']:.2f}%")
        with cols[2]:
            if "mu_rmse" in fit_result:
                st.text(f"μo: escala ≈ {fit_result['s_mu']:.3f} | RMSE ≈ {fit_result['mu_rmse']:.3f} | MAPE ≈ {fit_result['mu_mape']:.1f}%")

# ----------------------------------
# Aceptar ajuste y guardar JSON
# ----------------------------------
CFG_FILENAME = "pvt_fit_config.json"
if "accepted_fit" not in st.session_state:
    st.session_state["accepted_fit"] = None

if accept_fit:
    cfg = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "base_correlation": base_key[1],
        "API": float(API),
        "sg_gas": float(sg_gas),
        "T_F": float(T_F),
        "Pb": float(Pb),
        "Rsb": float(Rsb),
        "co": float(co),
        "normalize_at_pb": bool(normalize_at_pb),
        "viscosity_model": visc_model,
        "visc_params": {"A": A_and, "B": B_and} if visc_model == "Andrade" else None,
        "scales": {
            "Rs": float(fit_result.get("s_rs", 1.0)),
            "Bo": float(fit_result.get("s_bo", 1.0)),
            "mu": float(fit_result.get("s_mu", 1.0))
        }
    }
    st.session_state["accepted_fit"] = cfg
    try:
        with open(CFG_FILENAME, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        st.success(f"Ajuste aceptado y guardado en '{CFG_FILENAME}'.")
    except Exception as e:
        st.error(f"No se pudo guardar {CFG_FILENAME}: {e}")

    st.download_button(
        "Descargar configuración PVT (JSON)",
        data=json.dumps(cfg, indent=2).encode("utf-8"),
        file_name=CFG_FILENAME,
        mime="application/json"
    )

if reset_fit:
    st.session_state["accepted_fit"] = None
    try:
        if os.path.exists(CFG_FILENAME):
            os.remove(CFG_FILENAME)
        st.info("Ajuste aceptado eliminado.")
    except Exception as e:
        st.error(f"No se pudo eliminar {CFG_FILENAME}: {e}")

if st.session_state["accepted_fit"]:
    st.markdown("### Ajuste aceptado (vigente)")
    st.json(st.session_state["accepted_fit"])

st.caption("Tip: también puedes pegar/editar la tabla de laboratorio directamente arriba. El Autofit funciona con cualquiera de las dos entradas (CSV o tabla).")
