Módulo 2 — PVT (Standing, Vazquez–Beggs, Glaso)
================================================

Este archivo `app_pvt.py` es una aplicación Streamlit para comparar correlaciones PVT de crudo negro (Rs, Bo, μo) y ajustar con datos de laboratorio (CSV).
- Correlaciones implementadas:
  - Rs: Standing (1947), Vasquez–Beggs (1980), Glaso (1980)*
  - Bo sat.: Standing (1947) usando el Rs seleccionado
  - μo: Beggs & Robinson (1975) (dead/sat y extensión >Pb simplificada)

*La forma de Glaso implementada produce curvas razonables, pero recuerda validar con tus datos PVT locales.

Cómo correr
-----------
1. Instala dependencias si hace falta: `pip install streamlit numpy pandas matplotlib`
2. Ejecuta: `streamlit run app_pvt.py`
3. En la interfaz, ingresa API, gravedad del gas, temperatura y rango de presión.
4. (Opcional) Sube tu CSV de laboratorio. Ejemplo incluido: `pvt_lab_example.csv`.

Formato CSV esperado
--------------------
Columnas (puedes omitir alguna): 
- P_psia, Rs_scfstb, Bo_rbblstb, mu_cp

Salida
------
- Gráficas comparativas por propiedad.
- Reporte de errores (RMSE, MAPE) y factor de escala por correlación (descarga CSV).

Nota
----
Usa esta herramienta como apoyo. Para decisiones de ingeniería, prioriza datos de laboratorio y revisa rangos de validez de cada correlación.