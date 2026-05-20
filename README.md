# ecoRTA - Herramienta Python para Análisis RTA de Pozos Petroleros

ecoRTA es una herramienta en desarrollo para apoyar el análisis de pozos petroleros mediante técnicas de **Rate Transient Analysis (RTA)**, **Decline Curve Analysis (DCA)**, integración de historia de producción, estimación de presión de fondo fluyendo y caracterización PVT.

El proyecto forma parte de un trabajo de grado orientado al desarrollo de una herramienta digital para aplicar RTA en pozos exploratorios durante pruebas extensas de producción en los Llanos Orientales.

---

## Estado actual del proyecto

El proyecto se está construyendo de forma incremental por módulos:

| Módulo | Descripción | Estado |
|---|---|---|
| M1 | Información de pozo, historia de producción y Pwf | Implementado parcialmente |
| M2 | Propiedades PVT y enriquecimiento de historia | Implementado parcialmente |
| M3 | DCA / curvas de declinación | Implementado con pruebas |
| M4 | RTA mediante curvas tipo | En desarrollo |
| M5 | Resultados integrados del pozo | Pendiente |

Actualmente el repositorio tiene:

- Pipeline M1-M2 para generar historia enriquecida.
- Pipeline M3 para generar artefactos básicos de DCA.
- Infraestructura M4 para cargar curvas tipo base desde CSV o tablas internas.
- Overlay visual de puntos del pozo sobre curvas tipo.
- Joystick logarítmico estilo arcade para matching manual.
- Pruebas automatizadas con `pytest`.

Último estado verificado:

```bash
27 passed

Estructura principal del repositorio: 

src/
  pipeline/
    run_full_workflow.py

  rta_type_curves/
    __init__.py
    models.py
    loader.py
    registry.py
    overlay.py
    sample_data.py

  services/
    rta_overlay_points_service.py

  ui/
    app.py
    m4_type_curve_overlay.py

  rta_pvt/
    make_pvt_cli.py

  well_mod/
    run_estimator.py
    models.py

  utils/
    units.py

data/
  type_curves/
    fetkovich_base.csv
    palacio_blasingame_base.csv
    agarwal_gardner_base.csv

tests/
  test_dca_service.py
  test_full_workflow_pipeline.py
  test_m1_m2_integration.py
  test_run_dca_pipeline.py
  test_type_curve_loader.py
  test_type_curve_overlay.py
  test_rta_overlay_points_service.py

Limitaciones actuales

El proyecto todavía no hace:

digitalización validada de curvas tipo reales;
cálculo formal de variables adimensionales RTA;
cálculo de tiempo de balance de materiales;
derivadas RTA;
matching automático;
estimación de:
presión de yacimiento;
kh;
skin;
volumen contactado;
OOIP;
EUR vía RTA;
comparación con software comercial.


Próximos pasos técnicos
1. Reemplazar curvas demo por curvas reales

Digitalizar o cargar tablas validadas para:

Fetkovich
Palacio-Blasingame
Agarwal-Gardner

Cada curva debe quedar marcada como:

digitized_pending_qc

y luego, cuando sea revisada:

validated
2. Crear servicio de transformación RTA

Archivo propuesto:

src/services/rta_transform_service.py

Salida esperada:

well_id
date
method
x
y
x_label
y_label
qo_stb_d
pwf_used_psia
delta_p_psia
normalized_rate
material_balance_time
3. Integrar puntos RTA reales en Streamlit

La UI debe dejar de depender de columnas arbitrarias X/Y y pasar a usar transformaciones RTA por método.

4. Implementar matching automático

Inicialmente como ajuste numérico simple sobre multiplicadores:

x_multiplier
y_multiplier

Luego incorporar restricciones por régimen de flujo y calidad de ajuste.

5. Calcular parámetros de yacimiento

Una vez exista matching validado:

kh
skin
volumen contactado
OOIP
presión de yacimiento
Filosofía de desarrollo

El proyecto sigue una estrategia conservadora:

no romper módulos existentes;
no borrar scripts funcionales;
mantener compatibilidad hacia atrás;
separar UI de lógica de cálculo;
evitar fórmulas no validadas;
marcar datos demo como demo;
agregar pruebas antes de integrar features complejas.