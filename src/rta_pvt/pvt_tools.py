# pvt_tools.py
import math
from dataclasses import dataclass

@dataclass
class PVTInputs:
    API: float
    gamma_g: float   # gas specific gravity (air=1)
    T_F: float       # °F
    Rsb: float       # scf/STB
    pressures: list  # list of psia

def gamma_o_from_api(API: float) -> float:
    return 141.5 / (API + 131.5)

def a_standing(API: float, T_F: float) -> float:
    return 0.00091 * (T_F - 460.0) - 0.0125 * API

def pb_standing(API: float, T_F: float, gamma_g: float, Rsb: float) -> float:
    a = a_standing(API, T_F)
    return 18.2 * (((Rsb / gamma_g) ** 0.83) * (10 ** a) - 1.4)

def rs_standing(p: float, API: float, T_F: float, gamma_g: float) -> float:
    a = a_standing(API, T_F)
    return gamma_g * ((((p / 18.2) + 1.4) / (10 ** a)) ** (1.0 / 0.83))

def bo_standing(Rs: float, gamma_g: float, gamma_o: float, T_F: float) -> float:
    F = Rs * math.sqrt(gamma_g / gamma_o) + 1.25 * T_F
    return 0.972 + 0.000147 * (F ** 1.175)

def mu_od_beggs_robinson(gamma_o: float, T_F: float) -> float:
    x = (T_F ** (-1.163)) * math.exp(13.108 - 6.591 / gamma_o)
    return (10 ** x) - 1.0

def mu_o_beggs_robinson(p: float, Pb: float, Rs: float, mu_od: float) -> float:
    A = 10.715 * ((Rs + 100.0) ** (-0.515))
    B = 5.44 * ((Rs + 150.0) ** (-0.338))
    mu_os = A * (mu_od ** B)
    if p <= Pb:
        return mu_os
    # sub-saturado
    m = 2.6 * (p ** 1.187) * math.exp(-11.513 - 8.98e-5 * p)
    return mu_os * ((p / Pb) ** m)

def compute_pvt_table(inp: PVTInputs):
    go = gamma_o_from_api(inp.API)
    Pb = pb_standing(inp.API, inp.T_F, inp.gamma_g, inp.Rsb)
    mu_od = mu_od_beggs_robinson(go, inp.T_F)

    rows = []
    # incluir punto en Pb además de la grilla
    pressures = sorted(set([*inp.pressures, Pb]))
    for p in pressures:
        Rs_sat = rs_standing(p, inp.API, inp.T_F, inp.gamma_g)
        Rs = Rs_sat if p <= Pb else inp.Rsb
        Bo = bo_standing(Rs, inp.gamma_g, go, inp.T_F)
        mu_o = mu_o_beggs_robinson(p, Pb, Rs, mu_od)
        regimen = "saturado" if p <= Pb else "sub-saturado"
        is_pb_point = abs(p - Pb) < 1e-6
        rows.append({
            "pressure_psi": p,
            "Pb_psi": Pb,
            "Rs_scf_per_STB": Rs,
            "Bo_bbl_per_STB": Bo,
            "mu_o_cP": mu_o,
            "regimen": regimen,
            "is_Pb_point": is_pb_point,
        })
    return rows
