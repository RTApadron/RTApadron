"""
Utilidades de conversión de unidades para el proyecto RTA.
Todas las funciones documentan unidades de entrada/salida.
"""

PSI_TO_PA = 6894.757293168
PA_TO_PSI = 1 / PSI_TO_PA
STB_TO_M3 = 0.158987294928
M3_TO_STB = 1 / STB_TO_M3

def psi_to_pa(psi: float) -> float:
    """Convierte presión de psi a Pa."""
    return psi * PSI_TO_PA

def pa_to_psi(pa: float) -> float:
    """Convierte presión de Pa a psi."""
    return pa * PA_TO_PSI

def stb_to_m3(stb: float) -> float:
    """Convierte volumen de STB a m³."""
    return stb * STB_TO_M3

def m3_to_stb(m3: float) -> float:
    """Convierte volumen de m³ a STB."""
    return m3 * M3_TO_STB
