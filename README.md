## Hi there 👋
--# 📊 RTA Project - Herramienta de Análisis de Yacimientos

Este repositorio corresponde a mi **proyecto de grado** de la Maestría en Ingeniería de Yacimientos.  
El objetivo es desarrollar una herramienta computacional para apoyar el **Análisis de Presión Transitoria (RTA)** y otras técnicas asociadas, utilizando **Python** y metodologías modernas de desarrollo ágil (Scrum).

---

## 🚀 Características principales (Sprint 0)
- Configuración de entorno en **Python 3.11**.
- Control de versiones con **Git + GitHub**.
- Entorno virtual (`venv`) y archivo `requirements.txt` para gestión de dependencias.
- Primer script de prueba: `hola.py`.

---

## 📂 Estructura del proyecto
RTApadron/
│
├── .gitignore # Archivos/carpetas ignorados (venv, VSCode, temporales)
├── README.md # Documentación principal
├── requirements.txt # Dependencias del proyecto
└── hola.py # Script inicial de prueba

---

## ⚙️ Instalación y uso

1. **Clonar este repositorio**
   ```bash
   git clone https://github.com/RTApadron/RTApadron.git
   cd RTApadron
python -m venv venv
# En Windows PowerShell
.\venv\Scripts\Activate
pip install -r requirements.txt
python hola.py

👨‍💻 Autor

Robert Padrón
GitHub @RTApadron

📌 Este proyecto se encuentra en desarrollo como parte de mi proyecto de grado. Toda contribución y retroalimentación es bienvenida.




# 🛠️ Proyecto RTA – Sprint 1

Este proyecto contiene las herramientas desarrolladas durante los sprints de un flujo de trabajo de RTA (Rate Transient Analysis).  
Los módulos se van construyendo incrementalmente: desde la información de pozo, pasando por PVT, hasta el modelado de flujo.

---

## 📂 Estructura del proyecto

```text

VS Code RTA Project/
│
├─ src/
│   └─ rta\_pvt/
│       ├─ **init**.py
│       ├─ pvt\_tools.py          # Funciones y clases principales (Standing, Beggs–Robinson)
│       ├─ make\_pvt\_excel.py     # Genera resultados básicos (CSV/XLSX)
│       ├─ make\_pvt\_outputs.py   # Genera resultados + gráficas en /output
│       ├─ make\_pvt\_cli.py       # CLI: corre con argumentos (--api, --gamma-g, --temp, --rsb…)
│       └─ app\_streamlit.py      # Interfaz gráfica con Streamlit
│
├─ output/                       # Resultados generados automáticamente
│   ├─ sprint1\_propiedades\_pvt.csv
│   ├─ sprint1\_propiedades\_pvt.xlsx
│   ├─ cli\_plot\_rs.png
│   ├─ cli\_plot\_bo.png
│   └─ cli\_plot\_mu.png
│
├─ .vscode/
│   └─ launch.json               # Configuración para correr Streamlit desde VSCode
│
├─ requirements.txt              # Dependencias (pandas, matplotlib, xlsxwriter, streamlit)
└─ README.md

````

---

## ⚙️ Instalación

1. Clona el repo o copia la carpeta del proyecto.  
2. Crea y activa un entorno virtual (Windows PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
````text

3. Instala las dependencias:

   ```powershell
   pip install -r requirements.txt
   ```

---

## ▶️ Uso de los scripts

### 1. Generar outputs simples

```powershell
python src/rta_pvt/make_pvt_excel.py
```

Genera `sprint1_propiedades_pvt.csv` y `sprint1_propiedades_pvt.xlsx`.

---

### 2. Generar outputs + gráficas en `/output`

```powershell
python src/rta_pvt/make_pvt_outputs.py
```

Resultados en la carpeta `output/`:

* CSV y Excel con tabla PVT
* Gráficas de Rs, Bo y μo (incluyendo anotación de Pb)

---

### 3. Usar CLI (argumentos por consola)

```powershell
python src/rta_pvt/make_pvt_cli.py --api 30 --gamma-g 0.80 --temp 200 --rsb 600
```

Parámetros disponibles:

* `--api` → grados API
* `--gamma-g` → gravedad específica del gas
* `--temp` → temperatura en °F
* `--rsb` → gas en solución a Pb (scf/STB)
* `--pmin`, `--pmax`, `--step` → rango de presiones

Genera:

* `output/pvt_cli_results.csv`
* `output/pvt_cli_results.xlsx`
* `output/cli_plot_rs.png`
* `output/cli_plot_bo.png`
* `output/cli_plot_mu.png`

---

### 4. Interfaz con Streamlit

```powershell
streamlit run src/rta_pvt/app_streamlit.py
```

Se abrirá en tu navegador (por defecto `http://localhost:8501`).
Desde ahí puedes:

* Ajustar inputs (API, γg, T, Rsb/Pb, rango de presiones).
* Ver tabla y gráficas.
* Descargar CSV/XLSX directamente.

⚠️ Si el puerto 8501 está ocupado, usa por ejemplo:

```powershell
streamlit run src/rta_pvt/app_streamlit.py --server.port 8502
```

---

### 5. Desde VSCode (Debug)

Con el archivo `.vscode/launch.json` puedes correr Streamlit directo desde VSCode:

* Abre la pestaña **Run and Debug** (Ctrl+Shift+D).
* Selecciona **“Streamlit: run app”**.
* ▶️ Ejecuta.

---

## 🚩 Roadmap de Sprints

### Sprint 1 — Módulo 1: Información de Pozo (2 semanas)

**Modelos de datos a implementar**

* Well
* Survey
* Estado mecánico
* Intervalos
* Levantamiento
* Historia: caudales **q-o-g-w**, Pwf (si existe), T, API

**Funcionalidades**

* Importadores desde **CSV/Excel** + validación.
* Estimador de **Pwf frente a perforados**:

  * Versión 1 (Sprint 1): gradiente hidrostático + fricción simplificada.
  * Versión 2 (Sprint 4): integrar correlación de gradiente completa.

**Criterio de éxito**

* Cargar pozos e información histórica.
* Derivar **Pwf estimada** cuando falte.
* Cumplir con el requerimiento metodológico (ID, survey, historia y Pwf).

---

### Sprint 2 — Módulo 2: PVT (2–3 semanas)

**Funcionalidades**

* Implementar correlaciones PVT: **Bo, Rs, μo, μg, ρ, Pb/Pr, Bg**.
* Selector de correlación (Standing, Beggs–Robinson, Glaso, Lee, etc.).
* Calibración con datos de laboratorio:

  * Override de parámetros.
  * Ajuste por mínimos cuadrados (“fit”).

**Criterio de éxito**

* Tablas PVT consistentes vs. laboratorio.
* Función de selección y calibración activas.

---

## ✨ Próximos sprints

* **Sprint 3 — Módulo 3:** Modelo de flujo en tuberías (Beggs & Brill, Duns & Ros).
* **Sprint 4 — Módulo 4:** Integración gradiente de presión completa (para estimador Pwf).
* **Sprint 5:** Integración de módulos + validación con dataset real + dashboards en Streamlit.

---

## ✍️ Créditos

Proyecto académico RTA – Sprint 1.
Correlaciones: **Standing (1947/1981)** y **Beggs & Robinson (1975)**.
