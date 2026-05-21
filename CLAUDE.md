# CLAUDE.md — ecoRTA project context

**Este archivo es leído automáticamente por Claude Code al inicio de cada sesión.**
Mantenerlo actualizado elimina la necesidad de re-contextualizar en sesiones nuevas.
Actualizar tras cada commit relevante.

---

## Proyecto

**ecoRTA** — herramienta Python para Rate Transient Analysis (RTA) de pozos
petroleros. Proyecto de grado de Robert Eduardo Padrón García,
Maestría en Ingeniería de Yacimientos, Fundación Universidad de América, Bogotá.

**Título oficial:** "Desarrollo de una herramienta digital para la aplicación de RTA
(Análisis de Transiente de Flujo) para pozos exploratorios en pruebas extensas de
producción en los Llanos Orientales."

**Contexto de aplicación:**
- Cliente: Ecopetrol SA / Hocol — Gerencia de Exploración, equipo Evaluación de Formaciones
- Pozos objetivo: CPO-9 (crudos pesados 9–16 °API, Cretácico T2/K1) y
  Llanos-123/Llanos-87 (14–34 °API, alta permeabilidad)
- Tipo de prueba: pruebas extensas de producción con ESP
- Validación final: comparar vs Harmony (IHS/Fekete), licencia disponible en Hocol

---

## Stack y convenciones

- Python 3.11
- Pydantic v2 (`field_validator`, `model_validator`) — **no usar @validator de v1**
- pandas para manipulación tabular; matplotlib (backend Agg) para gráficas
- Streamlit para todas las UI interactivas
- pytest — correr siempre con `pytest` desde la raíz del proyecto antes de commit
- Imports absolutos desde `src/` (PROJECT_ROOT en `sys.path` vía `Path(__file__).parents[N]`)
- Archivos de prueba en `tests/`, prefijo `test_`; outputs en `output/`
- Branch activo: `feature/m4-type-curve-overlay`

---

## Estado de módulos (actualizado 2026-05-20)

| Módulo | Descripción | Estado | Tests |
|--------|-------------|--------|-------|
| M1 | Historia de producción, Pwf v2 Darcy-Weisbach, esquema mecánico + 9 QC checks | ✅ Funcional | `test_well_mech_qc_service.py` (51) |
| M2 | PVT: Rs/Bo/μo/ρo — Standing (1947), Vasquez-Beggs (1980), Beggs-Robinson (1975) | ✅ Funcional | `test_pvt_correlations.py` (46) |
| M3 | DCA: curvas de declinación Arps | ✅ Funcional | varios |
| M4 | RTA curvas tipo (Fetkovich, Blasingame, Agarwal-Gardner), match manual, QC | 🔧 Activo | varios |
| M5 | Resultados integrados, dashboard comparativo, reporte tesis | ⏳ Pendiente | — |

**Tests totales: 296 passed, 1 warning (Pydantic v1 @validator en `src/well_mod/models.py`)**

---

## Mapa de archivos clave

```
src/
├── services/
│   ├── pvt_correlations.py        — funciones puras PVT: Standing, VB, BR, densidad
│   ├── pvt_service.py             — PVTTableInput, PVTPressurePoint, compute_pvt_table()
│   ├── well_mech_qc_service.py    — WellMechConfig, CasingString, TubingString, run_mech_qc()
│   ├── rta_qc_service.py          — QCResult, run_rta_qc(), qc_severity_level()
│   ├── rta_transform_service.py   — MBT, normalized_rate, delta_p
│   ├── rta_overlay_points_service.py
│   ├── rta_match_params_service.py
│   ├── rta_export_service.py
│   ├── rta_scenario_service.py
│   ├── rta_synthetic_case.py
│   ├── dca_service.py
│   └── integration_service.py
├── ui/
│   ├── m1_well_editor.py          — Editor estado mecánico + esquema + Pwf
│   ├── m2_pvt_editor.py           — PVT interactivo: curvas, comparación corrs., lab manual/CSV
│   └── m4_type_curve_overlay.py   — Overlay curvas tipo, match arcade log-log, QC
├── well_mod/
│   ├── pwf.py                     — estimate_pwf_v1 (legacy), estimate_pwf_v2 (D-W/Churchill)
│   ├── schematic.py               — draw_well_schematic(), schematic_to_png_bytes()
│   └── models.py                  — MechState, Lift (tiene @validator de Pydantic v1 — no tocar)
├── rta_pvt/
│   ├── pvt_tools.py               — LEGACY: bug T_F-460 en a_standing(). No tocar — usar pvt_correlations.py
│   └── app_pvt_v5.py              — referencia histórica Streamlit PVT (no integrado)
├── adapters/
│   ├── m2_pvt_adapter.py          — build_pvt_table() estático, NO tocar (lo usan tests integración)
│   └── m1_loader_adapter.py
├── domain/
│   └── models.py                  — WellStatic, HistoryPoint, PVTConfig, PVTPoint, EnrichedHistoryPoint
├── rta_type_curves/
│   ├── overlay.py                 — ManualMatchConfig, build_overlay()
│   └── models.py                  — RTATypeCurveMethod, TypeCurve, CurveDataStatus
└── pipeline/
    └── run_full_workflow.py       — orquesta M1-M2-M3

tests/
├── test_pvt_correlations.py       — 46 tests: Standing, VB, BR, pvt_service
├── test_well_mech_qc_service.py   — 51 tests: 9 QC checks mecánicos
├── test_rta_qc_service.py         — 40 tests: 6 QC checks RTA
├── test_m1_m2_integration.py      — 8 tests integración historia+PVT (no romper)
└── ...

data/type_curves/
├── fetkovich_base.csv             — ⚠️ DEMO — no usar para interpretación técnica
├── palacio_blasingame_base.csv    — ⚠️ DEMO
└── agarwal_gardner_base.csv       — ⚠️ DEMO
```

---

## Bugs conocidos / deuda técnica

| Archivo | Problema | Estado |
|---------|----------|--------|
| `src/rta_pvt/pvt_tools.py` | `a_standing()` usa `(T_F - 460)` en vez de `T_F` — Pb incorrecto (~3× bajo) | ⚠️ Conocido, no corregido (legacy) |
| `src/well_mod/models.py` | `@validator` de Pydantic v1 — genera deprecation warning | ⚠️ Conocido, no urgente |
| Curvas tipo | Datos digitalizados son DEMO, no calibrados contra papers | ⏳ Pendiente digitalizar |

**Regla:** usar siempre `src/services/pvt_correlations.py` para PVT nuevo, nunca `pvt_tools.py`.

---

## Variables RTA implementadas

| Variable | Fórmula | Unidades |
|----------|---------|----------|
| `delta_p_psia` | `pi - pwf_used` | psia |
| `normalized_rate` | `qo / Δp` | STB/d/psi |
| `material_balance_time` (MBT) | `Np / qo` (ec. 12 Palacio-Blasingame) | días |
| `qDdi` | integral de MBT de qDd | adimensional |
| `qDdid` | derivada-integral de qDd | adimensional |

---

## Marco teórico — variables adimensionales por método

### Fetkovich (SPE-4629, 1980)

    qD  = 141.3 * q(t) * μ * Bo / (kh * (pi - pwf))          [Ec. 6]
    tD  = 0.00634 * k * t / (φ * μ * ct * rw²)               [Ec. 7]
    qDd = qD * [ln(re/rw) - ½]                                [Ec. 21]
    tDd = tD / { ½[(re/rw)²-1] * [ln(re/rw) - ½] }          [Ec. 20]

Parámetros desde match:
    kh = 141.3 * μ * Bo * [ln(re/rw) - ½] * (qDd_MP / (Δp * qi_MP))

### Palacio-Blasingame (SPE-25909, 1993)

    t̄   = Np / qo                                             [MBT, Ec. 12]
    bpss = 141.2 * Bo * μo / (ko * h) * ½ * ln(4/(eᵞ*CA*rw²))
    qDd  = qo / (pi - pwf) * bpss
    qDdi = (1/t̄) * ∫₀^t̄ qDd dt̄                             [integral, Ec. 14]
    qDdid = -t̄ * d(qDd)/dt̄                                   [deriv-integral, Ec. 15]

### Agarwal-Gardner (SPE-49222, 1998)

    tDA = 0.00633 * k * t / (φ * μ * ct * A)
    1/pwD = 141.2 * q * B * μ / (kh * Δm(p))

---

## Curvas tipo PVT — correlaciones implementadas

### Standing (1947) — `pvt_correlations.py`
    a   = 0.0125·API − 0.00091·T_F          (T en °F directamente)
    Rs  = γg · [(P/18.2 + 1.4) · 10^a]^1.2048   [scf/STB]
    Pb  = 18.2 · [(Rsb/γg)^0.83 · 10^(−a) − 1.4] [psia]
    Bo  = 0.972 + 0.000147 · F^1.175        F = Rs·(γg/γo)^0.5 + 1.25·T

### Vasquez-Beggs (1980)
    API ≤ 30: C1=0.0362, C2=1.0937, C3=25.724
    API > 30: C1=0.0178, C2=1.1870, C3=23.931
    Rs  = C1·γg·P^C2·exp(C3·API/(T+460))

### Beggs-Robinson (1975) — viscosidad
    μ_dead = 10^x − 1;   x = T^(−1.163) · exp(13.108 − 6.591/γo)
    μ_sat  = A · μ_dead^B;  A = 10.715·(Rs+100)^(−0.515), B = 5.44·(Rs+150)^(−0.338)
    μ_us   = μ_b · (P/Pb)^m;  m = 2.6·P^1.187·exp(−11.513 − 8.98e-5·P)

---

## Pending work por módulo

### M4 — pendiente
- [ ] Panel de configuración de yacimiento: `pi_psia`, `phi`, `h`, `ct`, `rw`, `re/area`, `Bo`, `μo`, `CA`
      → permite calcular kh, k, OOIP/contactado desde match point
- [ ] Parámetros físicos desde match (kh, k, OOIP) usando el panel anterior
- [ ] Exportación M4: `output/<well_id>_rta_results.csv`, `_match_summary.json`, PNG
- [ ] Pulido UI M4

### M5 — pendiente (todo)
- [ ] Modelo común de resultados (unificar M1–M4)
- [ ] Dashboard comparativo (EUR DCA vs volumen contactado RTA vs OOIP)
- [ ] Reporte exportable (CSV/JSON consolidado)
- [ ] QC final y trazabilidad
- [ ] Tabla para tesis (comparación vs Harmony)

---

## Filosofía de desarrollo — respetar siempre

1. No romper módulos existentes ni eliminar scripts funcionales
2. Separar UI (Streamlit) de lógica de cálculo (services/)
3. No implementar fórmulas no validadas — marcar como `demo`
4. Agregar pruebas antes de integrar features complejas
5. Mantener compatibilidad hacia atrás (esp. `m2_pvt_adapter.py` y `test_m1_m2_integration.py`)
6. `pytest` en verde antes de cada commit — sin excepciones
7. Las curvas tipo actuales son DEMO — no usar para interpretación técnica real

---

## Bibliografía de referencia

| Ref | Autores | Año | SPE | Relevancia |
|-----|---------|-----|-----|------------|
| Fetkovich | M.J. Fetkovich | 1980 | SPE-4629 | qDd/tDd, kh desde match |
| Palacio-Blasingame | J.C. Palacio, T.A. Blasingame | 1993 | SPE-25909 | MBT, qDdi/qDdid |
| Agarwal-Gardner | R.G. Agarwal, D.C. Gardner et al. | 1998 | SPE-49222 | 1/pwD vs tDA, derivadas |
| Arps | J.J. Arps | 1945 | Trans. AIME 160 | Declinación empírica |
| Standing | M.B. Standing | 1947 | API Drill. Prod. Pract. | Rs, Pb, Bo |
| Vasquez-Beggs | M.E. Vasquez, H.D. Beggs | 1980 | JPT Jun-1980 | Rs, Bo (API≤30 y >30) |
| Beggs-Robinson | H.D. Beggs, J.R. Robinson | 1975 | JPT Sep-1975 | μo dead/sat/undersat |
| Churchill | S.W. Churchill | 1977 | Chem. Eng. 84(24) | Factor fricción D-W todos regímenes |

**PDFs en el proyecto:** `fetkovich1980.pdf`, `SPE_25909_Palacio-Blasingame_Gas_Well_Dec_TC_Anl1993.pdf`, `agarwal1998.pdf`

---

## Comandos frecuentes

```bash
pytest                                                      # todas las pruebas
pytest tests/test_pvt_correlations.py -v                   # solo PVT
pytest tests/test_well_mech_qc_service.py -v               # solo mecánico M1
python -m streamlit run src/ui/m1_well_editor.py            # UI M1 estado mecánico
python -m streamlit run src/ui/m2_pvt_editor.py            # UI M2 PVT
python -m streamlit run src/ui/m4_type_curve_overlay.py    # UI M4 RTA
python src/pipeline/run_full_workflow.py                    # pipeline M1-M2-M3
```
