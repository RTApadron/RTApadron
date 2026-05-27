# ecoRTA

**Herramienta digital para Análisis de Transiente de Flujo (RTA) en pozos exploratorios**

Desarrollada como proyecto de grado de la Maestría en Ingeniería de Yacimientos,  
Fundación Universidad de América — Bogotá, Colombia.

**Autor:** Robert Eduardo Padrón García  
**Aplicación:** Pruebas extensas de producción con ESP en los Llanos Orientales  
**Cliente objetivo:** Ecopetrol S.A. / Hocol — Gerencia de Exploración

---

## ¿Qué es ecoRTA?

ecoRTA integra en una sola interfaz Streamlit los cinco pasos del análisis técnico de un pozo exploratorio:

| Módulo | Función | Estado |
|--------|---------|--------|
| **M1 — Historia** | Carga de producción, cálculo de Pwf (Darcy-Weisbach/Churchill), esquema mecánico, 9 QC checks | ✅ Funcional |
| **M2 — PVT** | Correlaciones Standing, Vasquez-Beggs, Beggs-Robinson; Rs, Bo, μo vs presión | ✅ Funcional |
| **M3 — DCA** | Declinación Arps multi-método (Exp/Hip/Arm), semilog, best-fit, EUR, R² | ✅ Funcional |
| **M4 — RTA** | Curvas tipo analíticas Fetkovich / Palacio-Blasingame / Agarwal-Gardner / Blasingame; joystick SNES; derivada log-log; kh · k · OOIP | ✅ Funcional |
| **M5 — Resultados** | Dashboard integrado, comparativo de volúmenes, parámetros de yacimiento por método, exportación CSV/JSON/Excel/PDF, validación vs software comercial | ✅ Funcional |

**413 pruebas automatizadas — todas pasando.**

---

## Características principales

### Análisis de Transiente de Flujo (M4)

- **79 curvas tipo analíticas** derivadas de las ecuaciones originales publicadas:
  - **Fetkovich** (SPE-4629, 1980) — 12 stems transientes + 5 curvas BDF Arps
  - **Palacio-Blasingame** (SPE-25909, 1993) — qDd / qDdi / qDdid, 7 re/rw × 3 series
  - **Agarwal-Gardner** (SPE-49222, 1998) — 7 curvas completas por re/rw (transiente E₁ exacto + BDF exponencial b=0, junction tDA ≈ 0.164)
  - **Blasingame** (solver numérico FD implícito) — 8 reD × 3 series
- **Joystick SNES** interactivo — matching manual con 7 niveles de sensibilidad
- **N match dinámico** — OOIP calculado en tiempo real desde la posición del joystick
- **Derivada log-log** — diagnóstico de régimen de flujo
- **Botón Auto** — selección automática del mejor stem BDF por distancia log-log media
- Visualización **Plotly interactivo** (zoom, hover, export PNG)

### Decline Curve Analysis (M3)

- Ajuste simultáneo Exponencial / Hiperbólico / Armónico con sliders interactivos
- Métricas qi, Di, b, R², RMSE bajo cada curva
- Best-fit precargado en sliders al cambiar ventana de ajuste
- Exportación de resultados al comparativo de M5

### Dashboard integrado (M5)

- Comparativo de volúmenes: EUR DCA vs OOIP volumétrico vs N match por método RTA
- Tabla de parámetros de yacimiento: kh, k, N vol., N match, re match, Área match — por cada método guardado
- Trazabilidad por parámetro: badges medido (lab) / estimado (correlación) / calculado (ajuste)
- Validación cuantitativa vs software comercial de referencia
- Exportación Excel (hoja Validación con score % concordancia) y PDF

---

## Stack tecnológico

```
Python 3.11 · Streamlit · Plotly · pandas · matplotlib · Pydantic v2 · xlsxwriter · pytest
```

---

## Instalación rápida

Consulta **[README_instalacion.md](README_instalacion.md)** para instrucciones completas paso a paso (Windows / macOS / Linux).

```bash
git clone https://github.com/RTApadron/RTApadron.git
cd RTApadron
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run src/ui/app.py
```

---

## Estructura del repositorio

```
src/
├── ui/
│   ├── app.py                      — Hub principal (navegación M1→M5)
│   ├── m1_well_editor.py           — Historia, Pwf, esquema mecánico
│   ├── m2_pvt_editor.py            — PVT interactivo
│   ├── m4_type_curve_overlay.py    — Match RTA con joystick SNES
│   ├── m5_results_dashboard.py     — Dashboard comparativo
│   └── components/snes_controller/ — Componente Streamlit bidireccional
├── services/
│   ├── pvt_correlations.py         — Standing, VB, Beggs-Robinson (puras)
│   ├── pvt_service.py              — Tabla PVT por presión
│   ├── well_mech_qc_service.py     — 9 QC checks mecánicos
│   ├── rta_transform_service.py    — MBT, tasa normalizada, derivada
│   ├── rta_match_params_service.py — kh, k, N match dinámico
│   ├── rta_export_service.py       — JSON match por método
│   ├── dca_service.py              — Arps multi-método
│   ├── m5_aggregator_service.py    — Consolidación M1→M4
│   ├── m5_export_service.py        — Excel / PDF / CSV / JSON
│   └── m5_comparison_service.py   — Score validación vs referencia
├── domain/
│   ├── models.py                   — WellStatic, HistoryPoint, PVTConfig
│   └── m5_models.py                — WellResultsSummary, RTASummary, ComparisonRow
└── rta_type_curves/
    ├── models.py                   — RTATypeCurveMethod, TypeCurve
    ├── overlay.py                  — build_overlay(), ManualMatchConfig
    └── blasingame.py               — Solver FD implícito radial

data/type_curves/
├── fetkovich_base.csv              — 12 curvas, 936 pts (status=validated)
├── palacio_blasingame_base.csv     — 36 curvas, 4320 pts (status=validated)
├── agarwal_gardner_base.csv        — 7 curvas, 959 pts (status=validated)
└── blasingame_base.csv             — 24 curvas, 8600 pts (status=demo)

scripts/
├── generate_type_curves.py         — Regenera los 4 CSVs analíticos
└── generate_qc_slides.py           — Genera PPTX QC técnico M4

tests/                              — 413 pruebas automatizadas (pytest)
```

---

## Comandos frecuentes

```bash
# Ejecutar todas las pruebas
pytest

# Iniciar la aplicación (puerto alternativo si 8501 está ocupado)
python -m streamlit run src/ui/app.py --server.port 8502

# Regenerar curvas tipo analíticas
python scripts/generate_type_curves.py
```

---

## Marco teórico — variables adimensionales implementadas

| Método | Ejes | Referencia |
|--------|------|------------|
| Fetkovich | qDd vs tDd — Ec. 20/21 | SPE-4629 (1980) |
| Palacio-Blasingame | qDd / qDdi / qDdid vs tDd — Ec. 12-15 | SPE-25909 (1993) |
| Agarwal-Gardner | qD vs tDA (E₁ exacto, junction ≈ 0.164) | SPE-49222 (1998) |
| Blasingame | qDd / qDdi / qDdid vs tcDd (FD numérico) | Blasingame et al. |
| Arps DCA | q(t) — Exp / Hip / Arm | Trans. AIME 160 (1945) |

---

## Estado de validación

Los resultados de kh, k y OOIP son **estimaciones preliminares** del match con curvas tipo analíticas.  
La validación cuantitativa vs software comercial de referencia (Harmony, IHS/Fekete) está en curso.  
No usar para toma de decisiones de campo sin validación previa.

---

## Licencia

GPL-3.0 — ver [LICENSE](LICENSE) para detalles.

---

## Contacto

**Robert E. Padrón García**  
📧 robert.padron@ecopetrol.com.co  
🎓 Maestría en Ingeniería de Yacimientos — Fundación Universidad de América, Bogotá
