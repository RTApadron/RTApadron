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

## Estado de módulos (actualizado 2026-05-23 — post sesión 6)

| Módulo | Descripción | Estado | Tests |
|--------|-------------|--------|-------|
| M1 | Historia + Pwf v2 D-W + esquema mecánico + 9 QC checks + editor embebido en hub | ✅ Funcional | `test_well_mech_qc_service.py` (51) |
| M2 | PVT: Rs/Bo/μo/ρo — Standing, VB, BR; botón "✅ Confirmar datos" semáforo verde | ✅ Funcional | `test_pvt_correlations.py` (46) |
| M3 | DCA multi-método Arps; semilog + best-fit; semáforo verde; **botón "💾 Guardar DCA para M5"** | ✅ Funcional | varios |
| M4 | RTA 60 curvas; Plotly zoom; 🎯 Auto stem; derivada log-log; SNES; **Bo/μo desde correlación a Pi** | ✅ Funcional | varios |
| M5 | Resultados integrados, dashboard 7 pestañas, exportación; **EUR DCA + badge PRELIMINAR** | ✅ Funcional + integrado | varios |
| Inicio | Tarjetas M1→M5 con logos PNG 140px; semáforo; botones nav; GPL-3 | ✅ Funcional | — |

**Tests totales: 391 passed, 1 warning (Pydantic v1 @validator en `src/well_mod/models.py`)**

> Sesión 6 añadió BLASINGAME al enum — test `test_log_derivative_present_for_all_methods`
> excluye explícitamente BLASINGAME (sin dispatch, reutiliza PB en M4).

---

## Historial de commits relevantes (más recientes primero)

### Sesión 6 — 2026-05-23 (Blasingame M4 + unificación P-B + SNES fixes)

**`32e4634` — assets: actualiza snes_controller.png con diseño nuevo (knob sin línea, 270°)**
- PNG nuevo del SNES controller (sin línea vertical en el knob)

**`fc56bbb` — fix(SNES): _imgH() con naturalWidth/naturalHeight + hotspots recalibrados**
- `src/ui/components/snes_controller/index.html`:
  - Nueva función `_imgH()` calcula altura real desde `img.naturalWidth/naturalHeight` — evita inflación del iframe por elementos absolutamente posicionados (sens-label)
  - Antes: `H = root.offsetHeight || W` usaba W como fallback cuadrado → `H = W = 340px` para imagen 2:1 de 170px real → todos los `top:%` CSS se computaban mal
  - `_setHeight()`, `_positionNeedle()`, `_positionSensLabel()` usan `_imgH()` consistentemente
  - Hotspots recalibrados para imagen 2:1: UP top:22%, DOWN top:59%, LEFT/RIGHT top:41%, RESET top:16%, SAVE top:41%, AUTO top:65%

**`65fccef` — fix(M4): checkboxes Blasingame persisten durante st.rerun() del SNES**
- Checkboxes qDd/qDdi/qDdid se renderizan ANTES de `st.columns()` — evita que el rerun del joystick destruya el registro del widget
- Keys pre-inicializadas en el bloque `_init_match_state()` para cada método

**`b5e560e` — feat(M4): unifica tab P-B con Blasingame, 3 checkboxes qDd/qDdi/qDdid**
- `_TAB_LABELS = ["🔬 Fetkovich", "📊 Palacio-Blasingame", "📈 Agarwal-Gardner"]` — Blasingame unificado en P-B
- `_TAB_METHODS = [FETKOVICH, BLASINGAME, AGARWAL_GARDNER]` — P-B tab carga blasingame_base.csv
- Scatter sigue usando PALACIO_BLASINGAME transform points (misma física)
- 3 checkboxes unificados (curvas tipo + nube simultáneos) en lugar de 6 separados
- `display_curves` filtra por `y_label` según checkboxes → ambos `_fig` y `_png` lo respetan

**`13b831d` — fix(generate_type_curves): filtro y>0 DESPUÉS de round para eliminar y=0.0**
- Antes: `if y > eps: rows.append(...)` + `y=round(y,10)` → round podía producir exactamente 0.0 pasando el filtro
- Corregido: `y = round(...); if x > 0 and y > 0:` — blasingame_base.csv: 4237 pts (era 5203 con y=0.0)

**`937240b` — feat(M4+M5): Blasingame tab, 3-series checkboxes, SNES AUTO, DataStatus fix**
- `src/domain/m5_models.py`: añade `"preliminary"` a `DataStatus` literal (corrige crash Pydantic M5)
- `src/rta_type_curves/models.py`: añade `BLASINGAME = "blasingame"` al enum `RTATypeCurveMethod`
- `src/services/rta_transform_service.py`: dispatch seguro con `.get()` + `continue` para BLASINGAME
- `scripts/generate_type_curves.py`: función `generate_blasingame()` + escritura `blasingame_base.csv`
- `data/type_curves/blasingame_base.csv` (NUEVO): 24 curvas, status=demo
- `src/ui/m4_type_curve_overlay.py`: SNES AUTO dispara `_find_best_bdf_stem` + pending key
- Rango aguja SNES: 270° (era 240°), offset -135°; labels ["MIN","1","2","3","4","5","MAX"]
- Botón AUTO (yellow) cableado en HTML del SNES controller
- `tests/test_rta_transform_service.py`: excluye BLASINGAME del test de log_derivative

**`cf1f89c` — feat: motor Blasingame numérico + script QC slides** (commit inicial sesión 6)
- `src/rta_type_curves/blasingame.py` (NUEVO): solver implícito radial en coordenadas ln(rD)
- `scripts/generate_qc_slides.py` (NUEVO): genera `output/ecoRTA_QC_tecnico_M4.pptx`
- `HANDOFF_PARA_CLAUDE_CODE.md` (NUEVO): documento de traspaso legacy

### Sesión 5 — 2026-05-22 (bugfixes + features con datos reales W001)

**`fe25798` — M3/M5: save DCA model summaries + M4 status from curve registry**
- M3: nuevo botón "💾 Guardar DCA para M5" → escribe `_dca_model_summary.csv` (1 fila/modelo: qi, Di_nominal, b, EUR_stb, R², RMSE, forecast_days, n_points). R²/RMSE calculados en el momento del save con los sliders actuales.
- M5 aggregator: lee `_dca_model_summary.csv` primero; si `_dca_fit_results.csv` no tiene columnas de modelo, advierte al usuario que use el nuevo botón.
- M4/M5 badge: `build_match_summary` acepta `curve_status`; cuando la curva tiene `status=validated`, el match se graba como `"preliminary"` (no `"demo"`). M5 muestra "△ PRELIMINAR" en lugar de "⚠️ DEMO".

**`1d22825` — M4: recompute Bo/μo from PVT correlations when Pi changes**
- Detecta en cada render si Pi widget cambió vs `rta_pvt_last_pi`; si cambió, llama `compute_pvt_table` al nuevo Pi y actualiza `rta_Bo_rb_stb` / `rta_mu_o_cp` en session_state antes de que los widgets rendericen.

**`c90aa5c` — M4/M5: compute Bo/μo at Pi from PVT correlations + fix legacy match JSON format**
- `_init_reservoir_config_state`: determina Pi antes de leer PVT; si `pvt_config_ui.json` no tiene `bo_rb_stb`/`mu_o_cp` (archivo viejo), llama `compute_pvt_table` con los inputs de correlación (api/gamma_g/temp_f/rsb) y elige el punto más cercano a Pi. Para W001 Pi=3800 psia → Bo≈1.08, μo≈8.88.
- `_build_rta_summary`: añade `_f_mult()` que intenta `match["x_multiplier"]` (formato nuevo) y luego `match_params["effective_x_multiplier"]` (formato legacy). Corrige test que fallaba.

**Commits sesión 5 — parte 1 (bug fixes con datos reales):**
- Fix duplicate Streamlit key en M3 tab "Gráficas PNG" — inline PNG rendering sin `render_dca_graphs_tab`
- Fix `session_state.ref_curve_fetkovich cannot be modified after widget instantiated` — patrón `auto_pending_{mval}`
- Fix `RTATransformPoint object has no attribute log_derivative` — `getattr` defensivo + limpiar `__pycache__`
- Fix `rta_well_id` mismatch M4→M5 — fuerza `rta_well_id = hub_well_id` en `_render_reservoir_config`
- Fix Bo/μo siempre mínimos — override incondicional desde PVT al final de `_init_reservoir_config_state`
- Fix CA defaulting a 0.1 — `_safe(config.CA, 1.0, 200.0, 31.62)` rechaza valores ≤ 1.0
- Fix parámetros reseteando a mínimos — filtro `_safe()` + `save_rta_scenario` al guardar match
- Fix M5 mostrando "—" para todos los parámetros RTA — `_build_rta_summary` leía claves del nivel incorrecto en JSON anidado

### Sesión 3 fixes 2026-05-22 (commit — ver abajo)

**`501ff36` — fix(sesion3): SNES controller, curvas tipo sin picos, logos 140px, M2 Confirmar**
- **M4 SNES controller:** componente Streamlit bidireccional (`declare_component`); imagen PNG
  real del SNES superpuesta con hotspots transparentes (D-pad, perilla sensibilidad ×2, RESET, SAVE);
  needle rotatoria −120°→+120°; protocolo `{action, seq}` con dedup por seq
- **M4 joystick reset bug:** `"Medio"` → `_SENSITIVITY_DEFAULT` en `_cb_reset()`
- **M4 curvas tipo Fetkovich:** pico en stem transiente eliminado con `min(qD_log, qD_early)`;
  antes qDd saltaba de 1.19 → 7.16 en el cruce de las dos aproximaciones
- **M4 Y-axis:** clamped a [1e-4, 200] en `_plot_all_curves_streamlit` — evita que el
  exponencial BDF b=0 (y→1.9e-9) aplaste el rango visible
- **CSVs regenerados:** 60 curvas, 7111 pts — Fetkovich stems monotónicamente decrecientes ✅
- **M2:** botón "✅ Confirmar datos" en `_pvt_core_ui(well_id)` — escribe
  `{well_id}_pvt_config_ui.json`; semáforo pasa a 🟢 sin datos de laboratorio
- **Inicio:** logos PNG M1–M5 ampliados de 80 → 140px
- **Nuevos archivos:** `assets/snes_controller.png`, `src/ui/components/snes_controller/index.html`

### Sesión 2 UX 2026-05-22 (commit cd48fff)

**`cd48fff` — feat(sesion2): joystick 7 pasos + SAVE, logos Inicio, M2 Confirmar, BDF colores**
- **M4 joystick:** `st.select_slider` con 7 pasos "1·MIN"→"7·MAX"; ×1.012 a ×3.162/click;
  botón `💾 SAVE` verde + `⟳ RESET` rojo en fila debajo del D-pad; SAVE guarda en tabla comparativa
- **M4 BDF dropdown:** leyenda de colores tab10 (cuadrado de color por curva BDF);
  la curva seleccionada se resalta; color map sincronizado con el chart
- **M2:** botón `✅ Confirmar datos` (secondary) → escribe `pvt_config_ui.json`
  sin validación estricta → semáforo M2 pasa a 🟢 sin datos de laboratorio
- **Inicio:** logos PNG M1–M5 (`assets/logo_m{1-5}.png`) sobre cada tarjeta del flujo;
  carga condicional con `_logo_path.exists()`

### Sesión bugfix 2026-05-22 (commit e440def)

**`e440def` — fix(bugs): corrige 7 bugs post-sprint — M3 semáforo, Y-axis, charts M4**
- **Root cause M3 semáforo siempre rojo:** `run_m3_dca_step` escribía `_dca_fit.csv`
  pero `compute_module_status` y `WorkflowArtifacts` buscaban `_dca_fit_results.csv`;
  también `_dca_rate_plot.png` → renombrado a `_dca_rate_fit.png`
- **M4 charts duplicados:** eliminado `st.pyplot()` dentro de `_plot_all_curves_streamlit`
  (se renderizaba por `st.pyplot` + `st.image` del caller)
- **M3 Y-axis 10⁴⁶:** `update_yaxes(type="log", range=[-1, ceil(log10(qmax))+1])`
- **M3 botón Ejecutar:** cambia a `type="secondary"` + "Re-ejecutar" cuando DCA ya existe
- **M3 label/qi:** "Escala semilog" (sin emoji largo); qi default = `iloc[-1]` (último Qo)
- **M4 P-B/A-G saltos:** `qDdid` calculado con `np.gradient` en `log(tDd)` en lugar de
  lineal — estable en grids log-espaciados. CSVs regenerados (60 curvas, 7111 pts)
- **M4 checkbox qDdi/qDdid:** off por defecto; opt-in "Mostrar series integrales"

### Sprint 2026-05-22 (commit 13cb28b)

**`13cb28b` — feat(sprint-22may): curvas tipo analíticas + M4 UX tabs + M3 semilog + best-fit**
- `scripts/generate_type_curves.py` (NUEVO): genera 60 curvas analíticas (Fetkovich SPE-4629,
  Palacio-Blasingame SPE-25909, Agarwal-Gardner SPE-49222) con ecuaciones exactas; `status=validated`
- `data/type_curves/` (NUEVO): 3 CSVs (~1000 / ~5000 / ~1000 pts). Loader los usa automáticamente.
- M4 — tabs por método en lugar de selectbox; elimina file uploader CSV (solo auto-load);
  layout `columns([3, 1.2])`: chart+QC izquierda, joystick+resultados derecha;
  parámetros expandidos por defecto; QC visible sin expander; Pi warning;
  joystick con claves per-método (sin conflictos entre tabs)
- M3 — checkbox `📐 Escala log Y (semilog)` activado por defecto; trazas punteadas
  best-fit (Exp/Hip/Arm) sobre ventana de ajuste histórico; métricas qi/Di/b/R² bajo sliders

### Sprint UX tarde — 2026-05-21 (commits 8dc3b8d → 2f957ed)

**`8dc3b8d` — fix(m4): sensibilidad joystick Grueso/Medio/Fino**
- `select_slider` numérico → `radio` con etiquetas descriptivas
- Grueso ×3.2/click · Medio ×1.26/click · Fino ×1.05/click
- `match_sensitivity_decades` float sigue siendo fuente de verdad

**`83d1295` — fix(m3): semáforo, multi-método DCA, sliders Di/b, colores**
- `st.rerun()` post-éxito: sidebar actualiza semáforo M3 correctamente
- Multi-método: checkboxes Exp🔴 / Hip🔵 / Arm🟠 — curvas simultáneas
- Sliders Di (%/año efectivo) y b por método; conversión correcta a nominal/día
- Panel EUR por método (MSTB / MM STB) debajo de la gráfica
- Excluye `forecast_start_rate` / `abandonment_rate` de curvas overlay
- Checklist visible solo en M1; M2 semáforo 🟡 con defaults / 🟢 si usuario guardó
- PVT defaults escritos a `_pvt_config_default.json` (separado de `_pvt_config_ui.json`)

**`981f310` — fix(m1): uploader Historia, survey import, editor embebido**
- CSV uploader movido de sidebar → M1 pestaña Historia (2-col: carga + status)
- Importador survey CSV/XLSX: calcula `inclination_deg = arccos(ΔTVD/ΔMD)`
- `render_m1_editor_embedded(well_id)` en `m1_well_editor.py`: casing/tubing/ESP/
  perforaciones + esquema matplotlib + QC mecánico; guarda a `_well_geometry.json`
- `main()` lee upload desde `st.session_state["m1_hist_uploader"]`
- Botón "🔄 Actualizar Historia" deshabilitado sin geometría configurada

**`2f957ed` — fix(m2): gráficas PVT + feat(inicio): pantalla bienvenida**
- Fix bug: `_lab_pair = df[["P_psia", col]].dropna()` evita error x/y distintos tamaños
- Pantalla Inicio: hero well_id, botones acción, tarjetas M1→M5 con flechas, semáforo,
  disclaimer GPL-3; contacto: `robert.padron@ecopetrol.com.co`
- Sidebar: 🏠 Inicio → M1-M5-Descargas → ❓ Ayuda; app abre en Inicio por defecto

### M4 integration bridges — 2026-05-21 (commit 64021b3)

**`64021b3` — feat(m4): integration bridges**
- `_init_reservoir_config_state(config, well_id, pvt_json_path)`: nueva firma
  - Siembra `rta_well_id` desde el `well_id` del hub (antes quedaba como "W-001")
  - Lee `bo_rb_stb` y `mu_o_cp` de `data/ui_uploads/{well_id}_pvt_config_ui.json`
    si existe, y los usa para pre-popular `rta_Bo_rb_stb` / `rta_mu_o_cp`
- `_run_m4_overlay`: construye path PVT y pasa `well_id` + `pvt_json_path` al init
- Auto-carga `output/{well_id}_history_enriched.csv` si existe; `file_uploader`
  queda como override opcional; mensaje claro si no hay ninguno de los dos
- Corregido `st.info()` que decía "No calcula todavía kh, skin..." — reemplazado
  por descripción precisa de capacidades actuales
- Corregido caption en `_render_reservoir_config`: menciona pre-carga automática PVT

### UX Sprint completo — 2026-05-20/21 (commits 38caa6e, 52da594)

**`38caa6e` — pipeline: --dca-only + CLI args faltantes**
- `--dca-only`: salta M1-M2, lee historia enriquecida existente → permite ejecutar sólo M3 DCA
- `--exclude-first-n`, `--forecast-start-rate-mode`, `--forecast-start-rate` añadidos al parser
  (antes app.py los pasaba pero el parser no los definía → subprocess siempre fallaba)
- `--history-csv` y `--pvt-config-json` ya no son `required=True`; se validan manualmente

**`52da594` — hub UX: checklist 4 pasos, mapper, logo, títulos, tabs**
- Checklist de configuración en main area: 4 expanders con semáforo (✅/⚪/⚠️)
  - Paso 1: Historia (estado, botón re-mapear)
  - Paso 2: Estado mecánico/survey — **context-aware**: si ya estás en M1 muestra
    texto apuntando a pestaña "⚙️ Geometría / Survey"; desde otro módulo muestra botón
  - Paso 3: PVT con warning/valores defaults inline + botón "→ M2 — PVT"
  - Paso 4: ▶ Ejecutar M1-M2 (solo si hay historia cargada)
- Botón ▶ Ejecutar M3 DCA movido al interior del módulo M3 (`render_artifacts`)
- `render_artifacts(well_id, inputs=None)` acepta dict de inputs para que M3 tenga acceso a params DCA
- Mapper CSV: auto-detección de separador (`csv.Sniffer`); sección de formato de fecha
  con auto-detección y normalización a ISO YYYY-MM-DD al guardar
- Fix checkbox limpiar: clave versionada `confirm_clear_checkbox_{n}` evita StreamlitAPIException
- Logo `assets/logo.jpg` en sidebar (`st.sidebar.image`)
- Título: `"ecoRTA"` sin sufijo modular; subtítulo descriptivo
- Pestañas M1: `📊 Historia`, `⚙️ Geometría / Survey`, `✏️ Edición Pwf`
- CSS tabs: font-weight 600, borde verde #2d6a4f activo, hover suave
- Banner ⚠️ sobre pestañas M1 cuando geometría no configurada

### Commit UX anterior (53fd5c5) — 2026-05-20
- Column mapper completo con auto-detección de alias, preview, confirm/skip
- Botón "🗑 Limpiar análisis" en sidebar
- `SESSION_HISTORY_MAPPER_ACTIVE`, `SESSION_HISTORY_MAPPED_PATH`, `SESSION_HISTORY_RAW_PATH`

### Commit UX (5c86dcf) — 2026-05-20
- Hub `app.py`: sidebar rediseñada como navegación M1→M5 + Descargas con semáforo
- `render_m2_embedded(well_id)` en `m2_pvt_editor.py`
- `render_m4_joystick_embedded(well_id, output_dir)` en `m4_type_curve_overlay.py`

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
│   ├── app.py                     — HUB PRINCIPAL (ver arquitectura abajo)
│   ├── m1_well_editor.py          — Editor estado mecánico + esquema + Pwf (standalone);
│   │                                exporta render_m1_editor_embedded(well_id)
│   ├── m2_pvt_editor.py           — PVT interactivo; exporta render_m2_embedded(well_id)
│   ├── m4_type_curve_overlay.py   — Overlay curvas tipo; exporta render_m4_joystick_embedded(well_id, output_dir)
│   └── m5_results_dashboard.py    — Dashboard M5; exporta render_m5_embedded(well_id, output_dir)
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
│   ├── models.py                  — WellStatic, HistoryPoint, PVTConfig, PVTPoint, EnrichedHistoryPoint
│   └── m5_models.py               — WellResultsSummary, WellInfoSummary, PVTSummary, DCASummary, RTASummary,
│                                    ExternalSoftwareResult, ComparisonRow
├── rta_type_curves/
│   ├── overlay.py                 — ManualMatchConfig, build_overlay()
│   ├── models.py                  — RTATypeCurveMethod, TypeCurve, CurveDataStatus
│   └── blasingame.py              — solver implícito radial; genera qDd/qDdi/qDdid (pendiente integración)
└── pipeline/
    └── run_full_workflow.py       — orquesta M1-M2-M3; flags: --skip-dca, --dca-only

assets/
└── logo.jpg                       — Logo ecoRTA (sidebar del hub)

tests/
├── test_pvt_correlations.py       — 46 tests: Standing, VB, BR, pvt_service
├── test_well_mech_qc_service.py   — 51 tests: 9 QC checks mecánicos
├── test_rta_qc_service.py         — 40 tests: 6 QC checks RTA
├── test_m1_m2_integration.py      — 8 tests integración historia+PVT (no romper)
└── ...

scripts/
├── generate_type_curves.py        — genera CSVs analíticos (Fetkovich/P-B/A-G); re-run tras cambios
└── generate_qc_slides.py          — genera output/ecoRTA_QC_tecnico_M4.pptx (12 slides, paleta arcade dark)

data/
├── type_curves/                   — 84 curvas (60 validated + 24 demo); regenerar con generate_type_curves.py
│   ├── fetkovich_base.csv         — 12 curvas, ~1036 pts (status=validated)
│   ├── palacio_blasingame_base.csv — 36 curvas, ~5040 pts (qDd/qDdi/qDdid; status=validated)
│   ├── agarwal_gardner_base.csv   — 12 curvas, ~1035 pts (status=validated)
│   └── blasingame_base.csv        — 24 curvas, ~4237 pts (8 reD × qDd/qDdi/qDdid; status=demo)
└── ui_uploads/                    — archivos guardados por la UI (no commitear)
```

---

## Arquitectura del hub `app.py`

### Flujo principal `main()`
```
configure_page() → initialize_session_defaults() → apply_light_css() → ensure_dirs()
    ↓
Título "ecoRTA" + caption (ocultos si active == "Inicio")
    ↓
render_sidebar_nav()
  ← 🏠 Inicio | M1-M5 | Descargas | ❓ Ayuda
  ← ID del pozo (text_input), PVT upload, DCA window, forecast, limpiar análisis
  ← devuelve inputs dict (SIN history_upload — movido a M1 pestaña Historia)
    ↓
st.session_state.get("m1_hist_uploader") → _hist_upload_widget
save_uploaded_file(_hist_upload_widget)  ← guarda CSV en data/ui_uploads/
    ↓
Column mapper (si hay upload nuevo sin mapear)
  → render_history_column_mapper(): auto-sep, auto-fecha, normaliza a ISO YYYY-MM-DD
  → [early return con render_artifacts()]
    ↓
Resolver history_csv_path (mapped > edited > raw)
Resolver pvt_config_json_path:
  - ui_pvt_config_ui.json (guardado explícito M2) → 🟢
  - uploaded_pvt_config.json (upload sidebar) → 🟢
  - _pvt_config_default.json (auto-generado) → no modifica semáforo M2
    ↓
_show_checklist = (active == "M1")
  if not _show_checklist:
    - compact status bar (excepto Inicio/Ayuda)
    - render_artifacts() → return
  else (M1):
    Checklist 4 pasos (expanders):
      Paso 1 — Historia (status + re-mapear)
      Paso 2 — Estado mecánico/survey (context-aware)
      Paso 3 — PVT (warning defaults + botón → M2)
      Paso 4 — 🔄 Actualizar Historia (disabled sin geometría; on success: st.rerun())
    ↓
render_artifacts(well_id, inputs=inputs)
  active == "Inicio"    → hero + tarjetas M1→M5 + semáforo + GPL-3
  active == "Ayuda"     → placeholder + contacto
  active == "M1"        → tabs [📊 Historia (uploader + render_m1_summary) |
                                 ⚙️ Geometría/Survey (render_m1_editor_embedded +
                                                       render_m1_geometry_and_survey_panel) |
                                 ✏️ Edición Pwf]
  active == "M2"        → render_m2_embedded(well_id)
  active == "M3"        → ▶ Ejecutar M3 DCA + ajuste interactivo multi-método
  active == "M4"        → render_m4_joystick_embedded(well_id, output_dir)
  active == "M5"        → render_m5_embedded(well_id, output_dir)
  active == "Descargas" → render_downloads_tab(artifacts)
```

### Builders de comando
```python
build_m1m2_command(*, well_id, history_csv, pvt_config_json, fit_from_date, fit_to_date)
    → [..., "--skip-dca"]

build_m3_command(*, well_id, fit_from_date, fit_to_date, exclude_first_n,
                 forecast_days, abandonment_rate, forecast_start_rate_mode, forecast_start_rate)
    → [..., "--dca-only"]

build_full_workflow_command(...)   ← mantenido para compatibilidad, ya no se usa en la UI
```

### Session state keys relevantes
```python
SESSION_ACTIVE_MODULE          = "active_module"          # Inicio/M1/M2/M3/M4/M5/Descargas/Ayuda
SESSION_HISTORY_MAPPER_ACTIVE  = "history_mapper_active"  # bool
SESSION_HISTORY_MAPPED_PATH    = "history_mapped_csv_path"
SESSION_HISTORY_RAW_PATH       = "history_raw_csv_path"
SESSION_EDITED_HISTORY_PATH    = "edited_history_csv_path"
SESSION_PVT_CONFIG_PATH        = "pvt_config_ui_path"
# Widget keys con estado persistente:
"m1_hist_uploader"             # file_uploader en M1 pestaña Historia
"m1_survey_import_upload"      # file_uploader importador survey en Geometría/Survey
"inicio_confirm_clear"         # bool — confirmación limpiar desde Inicio
"clear_checkbox_version"       # int — versión para evitar StreamlitAPIException
```

---

## Bugs conocidos / deuda técnica

| Archivo | Problema | Estado |
|---------|----------|--------|
| `src/rta_pvt/pvt_tools.py` | `a_standing()` usa `(T_F - 460)` en vez de `T_F` — Pb incorrecto (~3× bajo) | ⚠️ Conocido, no corregido (legacy) |
| `src/well_mod/models.py` | `@validator` de Pydantic v1 — genera deprecation warning | ⚠️ Conocido, no urgente |
| Curvas tipo | Curvas analíticas validadas (status=validated). Match marcado "PRELIMINAR". Pendiente validación cuantitativa vs Software Comercial. | ⏳ Sesión 6 |
| `app.py` M1 Geometría | Dos paneles de guardado en la misma pestaña (`render_m1_editor_embedded` + `render_m1_geometry_and_survey_panel`). Ambos escriben `_well_geometry.json` — el último en guardar gana. Revisar si unificar o eliminar uno. | ⚠️ Revisar mañana |

**Regla:** usar siempre `src/services/pvt_correlations.py` para PVT nuevo, nunca `pvt_tools.py`.

---

## Pending work — Backlog (próximo sprint a planificar)

### ✅ Sprint 2026-05-22 sesión 3 — COMPLETADO
- [x] M4 SNES controller (componente bidireccional real con imagen PNG)
- [x] M4 joystick reset bug (_SENSITIVITY_DEFAULT)
- [x] M4 curvas tipo Fetkovich: pico transiente eliminado
- [x] M4 Y-axis clamped [1e-4, 200]
- [x] M2 botón "Confirmar datos" en ruta correcta (_pvt_core_ui)
- [x] Inicio logos 140px

### ✅ Sprint 2026-05-22 sesiones 1-2 — COMPLETADO (commits cd48fff, e440def, 13cb28b)
- [x] T1: `scripts/generate_type_curves.py` + 60 curvas analíticas validated
- [x] T2: M4 rediseño UX (tabs, layout [3,1.2], sin uploader, QC visible, Pi warning)
- [x] T3: M3 semilog + best-fit punteado + métricas R²
- [x] M4 joystick 7 pasos + SAVE verde + leyenda BDF colores
- [x] M2 "Confirmar datos" semáforo
- [x] Pantalla Inicio completa

### ✅ Sprint sesión 4 — COMPLETADO (2026-05-22, commits ab170e3 → 4312892)

- [x] M4 zoom interactivo: `st.image(png)` → Plotly (`_plot_all_curves_plotly`); PNG queda para descargas
- [x] M4 auto-selección mejor stem: `_find_best_bdf_stem()` + botón 🎯 Auto (distancia log-log media)
- [x] M4 derivada log-log: campo `log_derivative` en `RTATransformPoint`, checkbox opt-in en 3 tabs
- [x] M1 paneles geometría unificados: sub-pestañas 🛢 Esquema / 📐 Survey + merge no-destructivo
- [x] M3 pre-cargar best-fit en sliders: Di/b/qi seeded al cambiar ventana de ajuste
- [x] Módulo Ayuda: 5 tabs (guía M1→M5, unidades, correlaciones, bibliografía, licencia)
- [x] 4 tests nuevos (log_derivative) → 391 passed

**Pendientes baja urgencia (dejar para sesión 5):**
- M4 SNES hotspots fine-tuning (baja urgencia)
- M4 P-B qDdid V-shapes (usuario dijo "no importa")
- Semáforo hover info: tooltip con detalle en sidebar

### ✅ Sprint sesión 5 — COMPLETADO (2026-05-22)

- [x] Validar workflow end-to-end con datos reales W001 — funcionando correctamente
- [x] Fix cascada de bugs encontrados durante prueba con W001 (ver sesión 5 arriba)
- [x] M4: Bo/μo calculados desde correlación PVT a Pi (no más mínimos)
- [x] M4: recomputar Bo/μo cuando Pi cambia
- [x] M3: botón "💾 Guardar DCA para M5" → `_dca_model_summary.csv`
- [x] M5: EUR DCA aparece en comparativo (leía formato de serie de tiempo, no resumen)
- [x] M5: badge DEMO → PRELIMINAR cuando curvas tienen status=validated
- [x] Push branch a origin — sincronizado

### ✅ Sprint sesión 6 — COMPLETADO (2026-05-23)

Commits: `cf1f89c`, `937240b`, `daf28a3`, `13b831d`, `b5e560e`, `65fccef`, `fc56bbb`, `32e4634`

**P3 Blasingame — COMPLETADO (con variación respecto al plan):**
- [x] BUG CRÍTICO M5: "preliminary" añadido a DataStatus (corrige crash Pydantic)
- [x] BLASINGAME añadido al enum RTATypeCurveMethod
- [x] Dispatch seguro en rta_transform_service (.get() + continue)
- [x] generate_type_curves.py: genera blasingame_base.csv (24 curvas, 4237 pts, y=0.0 artefactos eliminados)
- [x] Tab Blasingame unificado en "📊 Palacio-Blasingame" (no pestaña separada — simplifica UX)
- [x] 3 checkboxes unificados qDd / qDdi / qDdid (controlan curvas tipo Y nube simultáneamente)
- [x] Checkbox persistence fix: checkboxes se renderizan antes de st.columns() → sobreviven st.rerun()
- [x] SNES AUTO action: dispara _find_best_bdf_stem desde botón físico
- [x] 391 tests passed; test_log_derivative excluye BLASINGAME explícitamente

**P4 SNES hotspots — PARCIALMENTE COMPLETADO:**
- [x] SNES HTML: _imgH() con naturalWidth/naturalHeight — corrige inflación del iframe
- [x] SNES HTML: rango aguja 270° (era 240°), labels ["MIN","1","2","3","4","5","MAX"]
- [x] SNES HTML: botón AUTO (yellow) integrado y cableado
- [x] assets/snes_controller.png nuevo PNG guardado y commiteado (32e4634)
- [~] Alineación hotspots mejorada pero no al 100% — pendiente ajuste fino sesión 7

**No completado de sesión 6 (del plan P1/P2/P5):**
- [ ] P1: score global validación, export Excel/PDF, badge validación
- [ ] P2: badges per-parámetro trazabilidad M5
- [ ] P5: semáforo hover info

### 🟡 Próximo sprint — sesión 7 (prioridades)

**P1 — Validación cuantitativa vs Software Comercial** (clave para tesis):
- Score global en `_tab_validacion`: N concordantes, N divergentes, % match+close → `st.metric()`
- Export hoja "Validacion vs SW Comercial" en Excel existente (`m5_export_service.py`)
- Badge "✅ VALIDADO" / "⚠️ VALIDACIÓN PARCIAL" en header del tab (condicionado a % concordancia)
- Sección PDF "Validación vs Software Comercial" con tabla de comparación

**P2 — M5 trazabilidad badges per-parámetro:**
- Badges medido/estimado/calculado en tabs PVT/DCA/RTA de M5
- `PVTSummary` campos: `pvt_source`, `kh_status`, `n_vol_status`
- `_build_pvt_summary()`: calibrated_flag → status "measured" vs "estimated"

**P4b — SNES hotspots ajuste fino:**
- Ajustar coordenadas CSS de los 9 botones hasta coincidir al 100% con el PNG nuevo

**P5 — Semáforo hover info:**
- Tooltip `help="..."` en indicadores semáforo sidebar con detalle de qué archivo activa cada módulo

### 🟢 Prioridad baja / futuro

- **Importar análisis** (botón disabled en Inicio): formato .zip con JSONs + CSVs.
- **Pydantic v1 warning** en `src/well_mod/models.py` → migrar a `@field_validator`.

### ✅ Completado sesión 2026-05-21 (mañana + tarde)
- [x] M4 bridges: auto-carga historia, PVT pre-populate, well_id seeding
- [x] M4 joystick: radio Grueso/Medio/Fino reemplaza select_slider numérico
- [x] M3: multi-método checkboxes, sliders Di/b, EUR panel, semáforo fix, colores
- [x] M1: CSV uploader movido a pestaña Historia, survey importer, editor embebido
- [x] M2: fix bug gráficas PVT (dropna pareado), semáforo 🟡 con defaults / 🟢 con config
- [x] Pantalla Inicio: hero, tarjetas M1→M5, semáforo, GPL-3, contacto
- [x] Sidebar: 🏠 Inicio + ❓ Ayuda, app abre en Inicio por defecto

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

## Filosofía de desarrollo — respetar siempre

1. No romper módulos existentes ni eliminar scripts funcionales
2. Separar UI (Streamlit) de lógica de cálculo (services/)
3. No implementar fórmulas no validadas — marcar como `demo`
4. Agregar pruebas antes de integrar features complejas
5. Mantener compatibilidad hacia atrás (esp. `m2_pvt_adapter.py` y `test_m1_m2_integration.py`)
6. `pytest` en verde antes de cada commit — sin excepciones
7. Las curvas tipo son analíticas validadas (status=validated). Match se graba como "preliminary". No citar como definitivo hasta validar vs Software Comercial.
8. No nombrar software comercial por nombre — usar "Software Comercial" como etiqueta

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
python -m streamlit run src/ui/app.py --server.port 8506   # HUB PRINCIPAL (puerto libre)
python -m streamlit run src/ui/m1_well_editor.py            # UI M1 standalone (esquema mecánico)
python -m streamlit run src/ui/m2_pvt_editor.py            # UI M2 PVT standalone
python -m streamlit run src/ui/m4_type_curve_overlay.py    # UI M4 RTA standalone
python -m streamlit run src/ui/m5_results_dashboard.py     # UI M5 standalone
python src/pipeline/run_full_workflow.py \
    --well-id W001 \
    --history-csv data/ui_uploads/W001_history_mapped.csv \
    --pvt-config-json data/ui_uploads/W001_pvt_config_ui.json   # pipeline M1-M2-M3
python src/pipeline/run_full_workflow.py \
    --well-id W001 --dca-only                               # solo M3 DCA
python src/pipeline/run_full_workflow.py \
    --well-id W001 --history-csv ... --pvt-config-json ... \
    --skip-dca                                              # solo M1-M2
git push origin feature/m4-type-curve-overlay              # push al origin
```
