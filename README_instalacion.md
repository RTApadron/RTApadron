# ecoRTA — Guía de Instalación Local (Beta)

**Versión beta — Uso interno Ecopetrol / Hocol**  
Herramienta digital para Rate Transient Analysis (RTA) en pozos exploratorios.  
*Robert E. Padrón García — Maestría Ing. Yacimientos, Fundación Universidad de América*

---

## Requisitos previos

| Requisito | Versión | Descarga |
|-----------|---------|----------|
| Python | **3.11.x** (recomendado) | https://www.python.org/downloads/release/python-3110/ |
| Git | Cualquier versión reciente | https://git-scm.com/download/win |

> ⚠️ **Importante:** usa Python **3.11**, no 3.12 ni 3.13. Pydantic v2 y Streamlit tienen mejor compatibilidad probada en 3.11.

---

## Instalación paso a paso (Windows)

Abre **PowerShell** o **Símbolo del sistema (cmd)** y ejecuta los siguientes comandos en orden:

### 1 — Clonar el repositorio

```powershell
git clone https://github.com/RTApadron/RTApadron.git
cd RTApadron
```

### 2 — Crear entorno virtual

```powershell
python -m venv .venv
```

### 3 — Activar el entorno virtual

```powershell
# En PowerShell:
.venv\Scripts\Activate.ps1

# En cmd:
.venv\Scripts\activate.bat
```

> Si PowerShell muestra un error de permisos, ejecuta primero:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 4 — Instalar dependencias

```powershell
pip install -r requirements.txt
```

La instalación tarda entre 2 y 5 minutos dependiendo de la conexión.

### 5 — Iniciar la aplicación

```powershell
python -m streamlit run src/ui/app.py
```

El navegador se abre automáticamente en `http://localhost:8501`.  
Si no abre solo, copia esa dirección en Chrome o Edge.

---

## Instalación en macOS / Linux

```bash
git clone https://github.com/RTApadron/RTApadron.git
cd RTApadron
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run src/ui/app.py
```

---

## Flujo de trabajo recomendado (M1 → M5)

```
M1 — Historia    →  M2 — PVT    →  M3 — DCA    →  M4 — RTA    →  M5 — Resultados
Cargar CSV            Correlaciones     Arps             Curvas tipo       Comparativo
historia prod.        PVT + Pwf         multi-método     Fetkovich/P-B/AG  kh · k · EUR
```

### Datos de entrada mínimos

| Archivo | Formato | Columnas mínimas |
|---------|---------|-----------------|
| Historia de producción | CSV (`;` o `,`) | Fecha, Qo (STB/d), Pwh o Pwf (psi) |
| Config PVT | JSON (opcional) | Se genera automáticamente con valores por defecto |

> El mapeador de columnas en M1 detecta automáticamente los nombres de columnas más comunes.

### ID de pozo

- Cada evaluador debe usar un **ID de pozo único** para sus archivos, por ejemplo: `W001`, `CPO9_A`, `LL123_B`.
- Todos los archivos de resultados se guardan en `output/` y `data/ui_uploads/` usando ese ID como prefijo.

---

## Estructura de carpetas relevante

```
RTApadron/
├── src/ui/app.py              ← punto de entrada principal
├── data/
│   ├── ui_uploads/            ← tus archivos de entrada (historia, PVT)
│   └── type_curves/           ← curvas tipo analíticas (incluidas en repo)
├── output/                    ← resultados exportados (JSON, Excel, PDF, PNG)
├── assets/                    ← logos e imágenes de la UI
└── requirements.txt
```

---

## Solución de problemas frecuentes

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | Verifica que el entorno virtual está activado (`(.venv)` al inicio del prompt) |
| `Error: Port 8501 already in use` | Usa otro puerto: `python -m streamlit run src/ui/app.py --server.port 8502` |
| La app abre pero el sidebar está vacío | Recarga con F5 o abre en ventana de incógnito |
| Error al cargar CSV | Verifica que el separador es `;` o `,` y que la fecha está en formato DD/MM/YYYY o YYYY-MM-DD |
| `pydantic` error al cargar M4/M5 | Borra `data/ui_uploads/*.json` y vuelve a ejecutar el flujo desde M1 |
| Gráfico del SNES controller no aparece | El componente tarda ~3 segundos en cargar; espera o recarga la pestaña M4 |

---

## Actualizar a la versión más reciente

```powershell
# Con el entorno virtual activado:
git pull origin feature/m4-type-curve-overlay
pip install -r requirements.txt   # solo si requirements.txt cambió
```

---

## Contacto y soporte

**Robert E. Padrón García**  
📧 robert.padron@ecopetrol.com.co  
🎓 Maestría Ing. Yacimientos — Fundación Universidad de América, Bogotá  

> Esta es una versión **beta preliminar** para evaluación interna.  
> Los resultados de kh, k y OOIP son estimaciones del match con curvas tipo analíticas.  
> Pendiente validación cuantitativa vs software comercial de referencia.
