import math
from dataclasses import dataclass
from typing import Optional
from .models import MechState, Lift

BBL_TO_FT3 = 5.614583333
DAY_TO_S = 86400.0

@dataclass
class PwfInputs:
    qo_stb_d: float
    qw_stb_d: float
    api: Optional[float] = 30.0
    sg_o: Optional[float] = None     # si quieres forzar SG del aceite
    sg_w: float = 1.0
    whp_psia: float = 100.0
    tvd_perf_ft: float = 6000.0
    tubing_id_in: float = 2.375
    length_ft: Optional[float] = None   # si None, usa tvd_perf_ft
    Cf: float = 0.02  # coef. fricción simplificado (ajustable)

def api_to_sg(api: float) -> float:
    return 141.5 / (api + 131.5)

def estimate_pwf_v1(inp: PwfInputs) -> float:
    sg_o = inp.sg_o if inp.sg_o is not None else api_to_sg(inp.api or 30.0)
    ql_stb_d = max(1e-9, (inp.qo_stb_d + inp.qw_stb_d))
    sg_mix = (inp.qo_stb_d * sg_o + inp.qw_stb_d * inp.sg_w) / ql_stb_d
    grad_psi_per_ft = 0.433 * sg_mix

    L = inp.length_ft if inp.length_ft else inp.tvd_perf_ft
    ID_ft = (inp.tubing_id_in / 12.0)
    area_ft2 = math.pi * (ID_ft**2) / 4.0

    # velocidad líquida (ft/s) asumiendo líquido equivalente
    ql_ft3_s = (ql_stb_d * BBL_TO_FT3) / DAY_TO_S
    v = ql_ft3_s / max(1e-9, area_ft2)

    # fricción simplificada (lineal en v): Cf * v * (L/ID)
    dp_fric_psi = inp.Cf * v * (L / max(1e-9, ID_ft))

    pwf = inp.whp_psia + grad_psi_per_ft * inp.tvd_perf_ft + dp_fric_psi
    return pwf
