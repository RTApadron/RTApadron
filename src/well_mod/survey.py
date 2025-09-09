import math
from typing import List, Tuple
from .models import SurveyPoint

def min_curv(svy: List[SurveyPoint]) -> List[Tuple[float, float]]:
    """Devuelve lista de (MD_ft, TVD_ft)."""
    svy_sorted = sorted(svy, key=lambda s: s.md_ft)
    tvd = [0.0]
    for i in range(1, len(svy_sorted)):
        md1, inc1, azi1 = svy_sorted[i-1].md_ft, math.radians(svy_sorted[i-1].inc_deg), math.radians(svy_sorted[i-1].azi_deg)
        md2, inc2, azi2 = svy_sorted[i].md_ft, math.radians(svy_sorted[i].inc_deg), math.radians(svy_sorted[i].azi_deg)
        dmd = md2 - md1
        dog = math.acos(max(-1,min(1, math.cos(inc2-inc1) - math.sin(inc1)*math.sin(inc2)*(1-math.cos(azi2-azi1)))))
        rf = 1.0 if dog == 0 else (2/dog)*math.tan(dog/2)
        # componente vertical promedio
        cosI1, cosI2 = math.cos(inc1), math.cos(inc2)
        dTVD = 0.5 * dmd * (cosI1 + cosI2) * rf
        tvd.append(tvd[-1] + dTVD)
    return [(svy_sorted[i].md_ft, tvd[i]) for i in range(len(svy_sorted))]
