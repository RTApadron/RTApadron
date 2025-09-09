# src/well_mod/app_streamlit_well.py
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from well_mod.io_csv import load_wells, load_survey, load_mech, load_lift, load_history
from well_mod.survey import min_curv
from well_mod.pwf import PwfInputs, estimate_pwf_v1
from PIL import Image

st.set_page_config(page_title="eco RTA Módulo 1 — Información de Pozo", layout="wide")

#-------LOGO-----
logo_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "logo.jpg")
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=180)  # ajusta el ancho a gusto

#------Titulo
st.title("ecoRTA Módulo 1 — Información de Pozo")
st.caption("Carga/valida CSVs, calcula TVD y estima Pwf (v1: hidrostático + fricción simple).")

# -------------------------------
# Helpers
# -------------------------------
def _read_uploaded_csv(file):
    return pd.read_csv(io.BytesIO(file.read()))

def _load_or_upload(name, loader_func, default_path):
    """Si el usuario sube archivo lo usa; si no, intenta leer de data/<name>.csv a través del loader pydantic."""
    col_up, col_stat = st.columns([3, 2])
    up = col_up.file_uploader(f"{name}.csv", type=["csv"], key=f"up_{name}")
    if up is not None:
        try:
            df = _read_uploaded_csv(up)
            ok, errs = loader_func.__self__._parse_df(df, loader_func.__annotations__['return'] if False else None)  # no-op
        except Exception:
            # Volvemos al camino normal de parseo usando df -> csv temporal si el loader requiere path
            tmp = f"_tmp_{name}.csv"
            df.to_csv(tmp, index=False)
            ok, errs = loader_func(tmp)
            os.remove(tmp)
        # Mostrar estado
        if errs:
            col_stat.error(f"{len(errs)} error(es) en {name}")
            with st.expander(f"Ver errores de {name}"):
                for e in errs:
                    st.text(e)
        else:
            col_stat.success(f"{len(ok)} registros válidos en {name}")
        return ok, errs, df
    else:
        # fallback a data/<name>.csv
        path = os.path.join("data", f"{name}.csv")
        if not os.path.exists(path):
            st.info(f"Sube **{name}.csv** o crea `data/{name}.csv`.")
            return [], [f"missing {path}"], None
        ok, errs = loader_func(path)
        df = pd.read_csv(path)
        if errs:
            col_stat.error(f"{len(errs)} error(es) en {name}")
            with st.expander(f"Ver errores de {name}"):
                for e in errs:
                    st.text(e)
        else:
            col_stat.success(f"{len(ok)} registros válidos en {name}")
        return ok, errs, df

# -------------------------------
# Carga/validación de archivos
# -------------------------------
st.header("1) Carga y validación de datos")

wells, e_wells, df_wells = _load_or_upload("wells", load_wells, "data/wells.csv")
svy,   e_svy,   df_svy   = _load_or_upload("survey", load_survey, "data/survey.csv")
mech,  e_mech,  df_mech  = _load_or_upload("mechanical_state", load_mech, "data/mechanical_state.csv")
lift,  e_lift,  df_lift  = _load_or_upload("lift", load_lift, "data/lift.csv")
hist,  e_hist,  df_hist  = _load_or_upload("history", load_history, "data/history.csv")

any_errs = sum([len(e) for e in (e_wells, e_svy, e_mech, e_lift, e_hist)])
if any_errs:
    st.warning("Hay errores de validación. Revisa las secciones anteriores.")
if not wells or not svy or not mech or not lift or not hist:
    st.stop()

# -------------------------------
# Selección de pozo
# -------------------------------
st.header("2) Selección de pozo y TVD (mínima curvatura)")
well_ids = sorted({w.well_id for w in wells})
wid = st.selectbox("Well ID", well_ids, index=0)

svy_w = [s for s in svy if s.well_id == wid]
mech_w = [m for m in mech if m.well_id == wid]
lift_w = [l for l in lift if l.well_id == wid]
hist_w = [h for h in hist if h.well_id == wid]

if not (svy_w and mech_w and lift_w and hist_w):
    st.error("Faltan datos para este pozo (survey, mechanical_state, lift o history).")
    st.stop()

# TVD por mínima curvatura
tvd_table = min_curv(svy_w)  # [(MD, TVD)]
md, tvd = zip(*tvd_table)
colA, colB = st.columns(2)
with colA:
    fig, ax = plt.subplots()
    ax.plot(md, tvd, marker="o")
    ax.set_xlabel("MD (ft)")
    ax.set_ylabel("TVD (ft)")
    ax.set_title(f"TVD — {wid}")
    st.pyplot(fig, clear_figure=True)

with colB:
    st.write("Survey (primeros 10 puntos):")
    st.dataframe(pd.DataFrame({"MD_ft": md, "TVD_ft": tvd}).head(10), use_container_width=True)

# Profundidad de perfo/tubería
ms = mech_w[0]
tvd_perf_ft = float(ms.perfs_bottom_ft)  # puedes cambiar a top o promedio
tubing_id_in = float(ms.tubing_id_in)
tubing_set_depth_ft = float(ms.tubing_set_depth_ft)
wh = lift_w[0]
whp_psia = float(wh.whp_psia or 100.0)

# -------------------------------
# Parámetros de estimación
# -------------------------------
st.header("3) Parámetros de estimación Pwf (v1)")
c1, c2, c3 = st.columns(3)
Cf = c1.slider("Coeficiente de fricción (Cf)", 0.005, 0.10, 0.02, step=0.005)
ID_in = c2.number_input("Tubing ID (in)", value=tubing_id_in, min_value=0.5, max_value=5.0, step=0.0625, format="%.4f")
use_len = c3.radio("Longitud para fricción", ["tubing_set_depth_ft", "tvd_perforaciones"], index=0)
L_fr = tubing_set_depth_ft if use_len == "tubing_set_depth_ft" else tvd_perf_ft

st.caption(f"WhP = **{whp_psia:.1f} psia** | TVD(perfs) = **{tvd_perf_ft:.0f} ft** | L_fricción = **{L_fr:.0f} ft**")

# -------------------------------
# Cálculo de Pwf por fecha
# -------------------------------
st.header("4) Estimación Pwf por fecha")
# convertir historia a DF con ese pozo
dfh = pd.DataFrame([{
    "date": h.date, "qo_stb_d": h.qo_stb_d, "qw_stb_d": h.qw_stb_d,
    "qg_mscf_d": h.qg_mscf_d, "pwf_psia": h.pwf_psia, "api": h.api, "temp_F": h.temp_F
} for h in hist_w]).sort_values("date")

def api_from_well():
    w = [w for w in wells if w.well_id == wid][0]
    return float(w.api or 30.0)

rows = []
for _, r in dfh.iterrows():
    if pd.notna(r.get("pwf_psia")) and float(r["pwf_psia"]) > 0:
        est = float(r["pwf_psia"])
        flag = "medida"
    else:
        pin = PwfInputs(
            qo_stb_d=float(r["qo_stb_d"]),
            qw_stb_d=float(r["qw_stb_d"]),
            api=float(r["api"]) if pd.notna(r["api"]) else api_from_well(),
            whp_psia=whp_psia,
            tvd_perf_ft=tvd_perf_ft,
            tubing_id_in=ID_in,
            length_ft=L_fr,
            Cf=Cf,
        )
        est = estimate_pwf_v1(pin)
        flag = "estimada"
    rows.append({
        "date": r["date"], "pwf_est_psia": est, "origen": flag,
        "qo_stb_d": r["qo_stb_d"], "qw_stb_d": r["qw_stb_d"], "qg_mscf_d": r["qg_mscf_d"]
    })

out = pd.DataFrame(rows)

# Tabla
st.dataframe(out, use_container_width=True)

# Gráfico
fig2, ax2 = plt.subplots()
ax2.plot(out["date"], out["pwf_est_psia"], marker="o")
ax2.set_title(f"Pwf (estimada/medida) — {wid}")
ax2.set_xlabel("Fecha")
ax2.set_ylabel("Pwf (psia)")
st.pyplot(fig2, clear_figure=True)

# Descarga CSV
st.subheader("Descargar resultados")
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar pwf_estimated.csv", data=csv_bytes, file_name="pwf_estimated.csv", mime="text/csv")
