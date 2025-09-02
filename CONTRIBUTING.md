# Convenciones del proyecto

- Variables y módulos en **inglés** (ej.: `q_oil`, `p_wf`, `pvt.py`).
- Documentar **fórmulas** y **unidades** en docstrings.
- Estándar de código: `black` (line-length 100) + `ruff`.
- Tests con `pytest` (carpeta `tests/`).
- Unidades por defecto:
  - Presión: psi (convertir a Pa cuando se requiera).
  - Volumen: STB y m³ según contexto (declararlo siempre).
  - Temperatura: °F y °C (declarar en cada función).
- Flujo típico antes de commit:
  1. `black src tests`
  2. `ruff check --fix src tests`
  3. `pytest`
