import pandas as pd
import streamlit as st

st.set_page_config(page_title="RTA Tool – Sprint 0", layout="centered")
st.title("RTA Tool – Sprint 0 ✅")

st.markdown("""
Pequeña verificación de pipeline (código + entorno + UI).
- Python 3.11
- Streamlit
- pandas
""")

df = pd.DataFrame(
    {"Presión (psi)": [1000, 2000, 3000], "Bo (rb/stb)": [1.25, 1.20, 1.15]}
)

st.subheader("Tabla demo")
st.dataframe(df, use_container_width=True)

st.success("Streamlit está corriendo correctamente 🎉")
