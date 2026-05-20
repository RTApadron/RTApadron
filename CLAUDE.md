# CLAUDE.md — ecoRTA project context

Este archivo es leído automáticamente por Claude Code al inicio de cada sesión.
Actualizar el estado de módulos cada vez que se complete un commit.

---

## Proyecto

**ecoRTA** — herramienta Python para Rate Transient Analysis (RTA) de pozos
petroleros. Proyecto de grado de Robert Eduardo Padrón García,
Maestría en Ingeniería de Yacimientos, Fundación Universidad de América, Bogotá.

**Título oficial:** "Desarrollo de una herramienta digital para la aplicación de RTA
(Análisis de Transiente de Flujo) para pozos exploratorios en pruebas extensas de
producción en los Llanos Orientales."

**Contexto de aplicación:**
- Cliente interno: Ecopetrol SA / Hocol (Gerencia de Exploración, equipo de Evaluación de Formaciones)
- Pozos objetivo: exploratorios en CPO-9 (crudos pesados 9-16°API, unidades T2 y K1 del Cretácico)
  y Llanos-123/Llanos-87 (crudos intermedios 14-34°API, alta permeabilidad)
- Tipo de prueba: pruebas extensas de producción (ESP como sistema de levantamiento)
- Validación final: comparar resultados contra Harmony (IHS/Fekete), licencia disponible en Hocol

---

## Stack y convenciones

- Python 3.11
- Pydantic v2 para modelos de datos (`model_config`, `field_validator`, `model_validator`)
- pandas para manipulación tabular
- matplotlib para gráficas de archivo; Streamlit para UI interactiva
- pytest para todas las pruebas — correr con `pytest` desde la raíz
- Imports absolutos desde `src/` (el project root está en `sys.path`)
- Archivos de prueba en `tests/`, prefijo `test_`
- Outputs generados en `output/`

---

## Estado de módulos

| Módulo | Descripción | Estado |
|--------|-------------|--------|
| M1 | Info de pozo, historia de producción, estimación Pwf | Implementado parcialmente |
| M2 | PVT: Bo, Rs, μo, correlaciones Standing/Beggs-Robinson | Implementado parcialmente |
| M3 | DCA: curvas de declinación Arps | Funcional con pruebas |
| M4 | RTA mediante curvas tipo | En desarrollo activo |
| M5 | Resultados integrados del pozo | Pendiente |

Tests: `pytest` debe pasar en verde antes de cualquier commit. Último estado: 27 passed.

---

## Servicios y archivos clave

```
src/services/rta_overlay_points_service.py  — puntos para overlay visual (columnas arbitrarias)
src/services/rta_transform_service.py       — variables RTA físicas: MBT, normalized_rate, Δp
src/rta_type_curves/overlay.py              — ManualMatchConfig, build_overlay
src/rta_type_curves/models.py               — RTATypeCurveMethod, TypeCurve, CurveDataStatus
src/ui/m4_type_curve_overlay.py             — UI Streamlit M4 con joystick arcade log-log
src/pipeline/run_full_workflow.py           — orquesta M1-M2 y M3
```

---

## Variables RTA implementadas (rta_transform_service.py)

| Variable | Fórmula | Unidades |
|----------|---------|----------|
| `delta_p_psia` | `pi - pwf_used` | psia |
| `normalized_rate` | `qo / Δp` | STB/d/psi |
| `material_balance_time` | `Np / q` (ecuación 12 Palacio-Blasingame) | días |

Ejes actuales (todos los métodos, versión actual):
- X = material balance time (días)
- Y = normalized rate qo/Δp (STB/d/psi)

Pendiente (sprint futuro): derivadas e integrales para Blasingame completo
(qDdi, qDdid según ecuaciones 14-15 del anteproyecto).

---

## Curvas tipo — estado de datos

```
data/type_curves/fetkovich_base.csv          status: demo — NO usar para interpretación técnica
data/type_curves/palacio_blasingame_base.csv status: demo
data/type_curves/agarwal_gardner_base.csv    status: demo
```

Las curvas reales deben digitalizarse desde los papers de referencia (ver sección
Bibliografía abajo). Hasta entonces, todos los resultados de matching son **preliminares**.

---

## Marco teórico — variables adimensionales por método

### Fetkovich (SPE 4629, 1980)

Variables adimensionales originales:

    qD  = 141.3 * q(t) * μ * Bo / (kh * (pi - pwf))          [Ec. 6]
    tD  = 0.00634 * k * t / (φ * μ * ct * rw²)               [Ec. 7]

Reformuladas en variables de declinación:

    qDd = q(t) / qi = qD * [ln(re/rw) - 1/2]                 [Ec. 21]
    tDd = tD / { ½[(re/rw)²-1] * [ln(re/rw) - 1/2] }        [Ec. 20]

Parámetros de yacimiento desde match point:

    kh  = 141.3 * μ * Bo * [ln(re/rw) - ½] * (qDd_MP / (Δp * qi_MP))
    Npi = (qi/Npi)_MP⁻¹  →  OOIP inicial

Familia de curvas: re/rw = 4 a 100,000 (transient stems) + curvas Arps b=0..1 (BDF stems)

### Palacio-Blasingame (SPE 25909, 1993)

Tiempo de balance de materia (MBT):

    t̄  = Np / qo                                              [Ec. 12 / Ec. 4]

Constante de flujo pseudoestable:

    bpss = 141.2 * Bo * μo / (ko * h) * ½ * ln(4/(eᵞ * CA * rw²))   [Ec. 3/11]

Tasa adimensional de declinación:

    qDd  = qo / (pi - pwf) * bpss                             [Ec. 6/13]
    qDdi = (1/t̄) * ∫₀^t̄ qDd dt̄                             [Ec. 14]  — integral
    qDdid = -t̄ * d(qDd)/dt̄                                   [Ec. 15]  — derivada integral

Todas las curvas BDF colapsan en la rama armónica b=1 cuando se usa t̄.
Parámetros desde match point (Ec. A-14 a A-16):

    N  = 1/(cti) * (t̄_MP / tDd_MP) * 1/(qDd_MP / (qo/(pi-pwf))_MP)
    k  = 141.2 * Bo * μo / h * [ln(re/rw) - ½] * ((qo/(pi-pwf))_MP / qDd_MP)

### Agarwal-Gardner (SPE 49222, 1998)

Tiempo adimensional basado en área:

    tDA = 0.00633 * k * t / (φ * μ * ct * A)                  [Ec. A-5]

Tasa adimensional inversa:

    1/pwD = 141.2 * q * B * μ / (kh * Δm(p))                  [Ec. A-6]

La curva 1/pwD vs tDA muestra claramente la transición transiente→BDF.
Durante BDF: 1/dlnPwD' tiene pendiente -1 (harmonic decline).
Durante flujo transiente: 1/dlnPwD' = constante = 2.0.

Parámetros desde match point (Ec. A-6, A-8, A-9):

    GIP = 1/cti * (t̄_MP / tDA_MP) * (1/(qDd_MP / (q/Δp)_MP))
    k   = 141.2 * Bo * μ / h * [ln(re/rw) - ½] * ((q/Δp)_MP / qDd_MP)

---

## Próximo commit prioritario

**Configuración de yacimiento/fluidos para RTA**

El `rta_transform_service` recibe `pi_psia` pero la UI M4 aún no lo expone.
Implementar panel de inputs con:
- `pi_psia` (presión inicial de yacimiento, psia)
- `phi` (porosidad, fracción)
- `h` (espesor neto, ft)
- `ct` (compresibilidad total, psi⁻¹)
- `rw` (radio de pozo, ft)
- `re` o `area` (radio/área de drene, ft o acres)
- `Bo` y `mu_o` promedio (RB/STB y cp)
- `CA` (factor de forma de Dietz — default 31.62 para circular)
- Guardar escenario como `output/<well_id>_rta_scenario.json`

---

## Commits pendientes M4

3. Parámetros físicos desde match (kh, k, volumen contactado, OOIP preliminar)
   — requiere el panel de configuración del commit anterior
4. Comparación de métodos RTA (tabla Fetkovich / Blasingame / Agarwal-Gardner)
5. Exportación M4 (`output/<well_id>_rta_results.csv`, `_rta_match_summary.json`, PNG)
6. QC técnico M4 (advertencias drawdown inestable, pocos puntos, no unicidad)
7. Pulido UI M4 (joystick estética arcade 90s con `st.components.v1.html`)

## Commits pendientes M5

1. Modelo común de resultados (unificar M1-M2-M3-M4)
2. Dashboard comparativo (EUR DCA vs volumen contactado RTA vs OOIP)
3. Reporte exportable (CSV/JSON consolidado, luego Excel por módulo)
4. QC final y trazabilidad (medido/estimado/calculado, exclusiones, config usada)
5. Reporte final para tesis (tabla para comparar vs Harmony de IHS)

---

## Filosofía de desarrollo — respetar siempre

1. No romper módulos existentes ni borrar scripts funcionales
2. Separar UI de lógica de cálculo
3. No implementar fórmulas no validadas — marcar como `demo` o `digitized_pending_qc`
4. Agregar pruebas antes de integrar features complejas
5. Mantener compatibilidad hacia atrás
6. `pytest` en verde antes de cada commit
7. Las curvas tipo actuales son DEMO — no usar para interpretación técnica

---

## Bibliografía de referencia

| Ref | Autores | Año | SPE/DOI | Relevancia |
|-----|---------|-----|---------|------------|
| Fetkovich | M.J. Fetkovich | 1980 | SPE-4629 | Curvas tipo base, qDd/tDd, kh desde match |
| Palacio-Blasingame | J.C. Palacio, T.A. Blasingame | 1993 | SPE-25909 | MBT, curvas líquido/gas, qDdi/qDdid |
| Agarwal-Gardner | R.G. Agarwal, D.C. Gardner et al. | 1998 | SPE-49222 | 1/pwD vs tDA, fracturas, derivadas |
| Arps | J.J. Arps | 1945 | Trans. AIME 160 | Declinación empírica exponencial/hiperbólica/armónica |

**Papers disponibles en el proyecto** (compartidos por el usuario):
- `fetkovich1980.pdf` — SPE 4629
- `SPE_25909_Palacio-Blasingame_Gas_Well_Dec_TC_Anl1993.pdf` — SPE 25909
- `agarwal1998.pdf` — SPE 49222

---

## Comandos frecuentes

```bash
pytest                                                    # correr todas las pruebas
python -m streamlit run src/ui/m4_type_curve_overlay.py  # UI M4
python src/pipeline/run_full_workflow.py                  # pipeline M1-M2-M3
```
