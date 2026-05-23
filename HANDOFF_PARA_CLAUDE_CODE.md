# Handoff ecoRTA → Claude Code

Pega este mensaje en Claude Code para retomar el trabajo exactamente donde quedamos.

---

## El proyecto

**ecoRTA** es una herramienta Python para Rate Transient Analysis (RTA) de pozos
petroleros, desarrollada como proyecto de grado de Maestría en Ingeniería de
Yacimientos. El caso de aplicación son pozos exploratorios en los Llanos Orientales
de Colombia durante pruebas extensas de producción.

Repositorio: `https://github.com/RTApadron/RTApadron`
Stack: Python 3.11, Pydantic, pandas, matplotlib, Streamlit, pytest.

---

## Estado actual confirmado

```
27 passed   ← todos los tests pasan antes de los cambios de hoy
```

| Módulo | Estado |
|--------|--------|
| M1 – Info de pozo / Pwf | Implementado parcialmente |
| M2 – PVT | Implementado parcialmente |
| M3 – DCA | Funcional con pruebas |
| M4 – RTA curvas tipo | En desarrollo activo |
| M5 – Resultados integrados | Pendiente |

---

## Estructura relevante del repositorio

```
src/
  pipeline/
    run_full_workflow.py          ← orquesta M1-M2 y M3
  rta_type_curves/
    __init__.py
    models.py                     ← TypeCurve, RTATypeCurveMethod, CurveDataStatus
    loader.py
    registry.py
    overlay.py                    ← ManualMatchConfig, build_overlay, plot_overlay
    sample_data.py
  services/
    rta_overlay_points_service.py ← carga CSV, detecta columnas, construye RTAOverlayPoint
    rta_transform_service.py      ← NUEVO: variables RTA físicas (ver abajo)
  ui/
    app.py
    m4_type_curve_overlay.py      ← UI Streamlit M4, joystick arcade
  rta_pvt/
    make_pvt_cli.py
  well_mod/
    run_estimator.py
    models.py
  utils/
    units.py
data/
  type_curves/
    fetkovich_base.csv            ← status: demo, no usar para interpretación técnica
    palacio_blasingame_base.csv   ← status: demo
    agarwal_gardner_base.csv      ← status: demo
tests/
  test_dca_service.py
  test_full_workflow_pipeline.py
  test_m1_m2_integration.py
  test_run_dca_pipeline.py
  test_type_curve_loader.py
  test_type_curve_overlay.py
  test_rta_overlay_points_service.py
  test_rta_transform_service.py   ← NUEVO (ver abajo)
```

---

## Cambios realizados HOY (aún no commiteados)

### 1. Fix bugs joystick — `src/ui/m4_type_curve_overlay.py`

Tres correcciones aplicadas:

**A) Import agregado:**
```python
import matplotlib.ticker as mticker
```

**B) Formateador de ticks plano en `_plot_overlay_streamlit`**
Reemplaza el formateador default de Matplotlib que genera LaTeX (`$10^{-12}$`)
y causa `ParseException` al llegar a valores extremos:
```python
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda val, _: f"{val:g}" if val != 0 else "0"
))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda val, _: f"{val:g}" if val != 0 else "0"
))
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
try:
    fig.tight_layout()
except Exception:
    pass
```

**C) Clamp defensivo en `_init_match_state`**
Solo corrige si está fuera de rango, no en cada rerun:
```python
for key in ("x_multiplier", "y_multiplier"):
    current = st.session_state[key]
    clamped = _clamp_multiplier(current)
    if clamped != current:
        st.session_state[key] = clamped
```

Estado: **verificado funcionando** — joystick llega a límites sin estallar.

---

### 2. Nuevo servicio — `src/services/rta_transform_service.py`

Primer servicio con variables RTA físicamente significativas.
Toma un DataFrame enriquecido (salida M1-M2) y produce `RTATransformPoint` con:

| Campo | Descripción |
|-------|-------------|
| `delta_p_psia` | `pi - pwf_used` |
| `normalized_rate` | `qo / Δp` (STB/d/psi) |
| `material_balance_time` | `Np / q` (días) |
| `x`, `y` | Ejes del método |
| `x_label`, `y_label` | Etiquetas físicas |

API pública:
```python
compute_rta_transforms(dataframe, pi_psia, methods=None) → list[RTATransformPoint]
compute_rta_transforms_from_csv(path, pi_psia, methods=None) → list[RTATransformPoint]
rta_points_to_dataframe(points) → pd.DataFrame
```

Los tres métodos (Fetkovich, Palacio-Blasingame, Agarwal-Gardner) usan
`MBT vs normalized_rate` como ejes primarios. Las derivadas e integrales
(para Blasingame completo) están documentadas como trabajo futuro en el código.

---

### 3. Nuevo archivo de pruebas — `tests/test_rta_transform_service.py`

15 pruebas que cubren:
- Shape del output y métodos correctos
- Invariantes físicas: `Δp = pi - pwf`, `nr = q/Δp`
- Monotonía del MBT
- Primera fila excluida porque `Np=0` → `MBT=0` → no log-safe
- Tasas cero/negativas descartadas
- Caso borde: una sola fila
- Conversión a `RTAOverlayPoint`
- Export a DataFrame
- Wrapper CSV

---

## Commits pendientes (acordados con el usuario)

El usuario quiere hacer commit de lo de hoy y luego continuar. El orden acordado es:

### M4 — faltan ~6 commits

**Commit siguiente (prioridad):**
> **Configuración de yacimiento/fluidos para RTA**
> - Panel de inputs editables: `pi`, `ct`, `phi`, `h`, `rw`, `re`, `area`, viscosidad/Bo promedio
> - Guardar escenario RTA por pozo (JSON en `output/`)
> - El `rta_transform_service` ya existe y espera `pi_psia` como input — la UI aún no lo expone

**Commits siguientes M4:**
3. Parámetros físicos desde match (kh, k, volumen contactado, OOIP preliminar)
4. Comparación de métodos RTA (tabla Fetkovich / Blasingame / Agarwal-Gardner)
5. Exportación M4 (`output/<well_id>_rta_results.csv`, `_rta_match_summary.json`, PNG)
6. QC técnico M4 (advertencias drawdown inestable, pocos puntos, no unicidad de match)
7. Pulido UI M4 (joystick estética arcade 90s real con `st.components.v1.html`, guardar/cargar escenarios)

### M5 — faltan ~4-5 commits
1. Modelo común de resultados (unificar M1-M2-M3-M4)
2. Dashboard comparativo (EUR DCA vs volumen contactado RTA vs OOIP)
3. Reporte exportable (CSV/JSON consolidado, luego Excel por módulo)
4. QC final y trazabilidad
5. Reporte final para tesis (tabla para comparar vs software comercial)

---

## Filosofía de desarrollo (no romper)

- No romper módulos existentes
- No borrar scripts funcionales
- Mantener compatibilidad hacia atrás
- Separar UI de lógica de cálculo
- Evitar fórmulas no validadas — marcar todo como `demo` hasta validación
- Agregar pruebas antes de integrar features complejas
- Las curvas tipo actuales están marcadas `status = demo` y NO deben usarse para interpretación técnica hasta ser digitalizadas y validadas

---

## Pendiente técnico importante

Las curvas tipo demo (`fetkovich_base.csv`, etc.) deben reemplazarse por
curvas reales digitalizadas. El usuario tiene los papers de referencia
(Fetkovich 1980, Palacio-Blasingame 1993, Agarwal-Gardner 1999) y los
compartirá cuando lleguemos al commit de variables adimensionales reales.

---

## Cómo correr el proyecto

```bash
# Tests
pytest

# UI M4
python -m streamlit run src/ui/m4_type_curve_overlay.py

# Pipeline completo
python src/pipeline/run_full_workflow.py
```
