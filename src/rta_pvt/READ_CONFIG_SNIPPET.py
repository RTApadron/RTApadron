
# Cómo consumir la configuración aceptada desde otros módulos (Python)
import json
with open("pvt_fit_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

base = cfg["base_correlation"]      # "Standing" | "Vasquez–Beggs" | "Glaso"
Pb   = cfg["Pb"]; Rsb = cfg["Rsb"]
co   = cfg["co"]; normalize = cfg["normalize_at_pb"]
s_rs = cfg["scales"]["Rs"]; s_bo = cfg["scales"]["Bo"]; s_mu = cfg["scales"]["mu"]

# ...usar estos parámetros para generar Rs/Bo/μo en tus siguientes módulos...
