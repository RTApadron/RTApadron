"""Generate analytical RTA type curves and write to data/type_curves/.

Run from the project root:
    python scripts/generate_type_curves.py

Equations:
  Fetkovich   — SPE-4629 (1980) Eq. 20-21
  Palacio-Blasingame — SPE-25909 (1993) Eq. 12-15
  Agarwal-Gardner    — SPE-49222 (1998) Eq. 3
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "type_curves"

# Allow importing from src/
sys.path.insert(0, str(PROJECT_ROOT))

# Euler-Mascheroni constant: ln(γ) = 0.5772...  →  γ = e^0.5772 ≈ 1.7811
_GAMMA_EM = math.exp(0.5772156649)

# CSV column order required by TypeCurveLoader
_FIELDNAMES = [
    "method", "curve_id", "curve_family",
    "x", "y",
    "x_label", "y_label",
    "source", "status", "notes",
]


# ---------------------------------------------------------------------------
# Arps BDF (used by all three methods)
# ---------------------------------------------------------------------------

def _arps_qdd(tDd: np.ndarray, b: float) -> np.ndarray:
    """Dimensionless decline rate — Arps Eq."""
    if b == 0.0:
        return np.exp(-tDd)
    return 1.0 / (1.0 + b * tDd) ** (1.0 / b)


# ---------------------------------------------------------------------------
# Fetkovich transient stem (radial flow, bounded circular reservoir)
# ---------------------------------------------------------------------------

def _fetkovich_transient_qD_raw(tD: float) -> float:
    """Transient qD WITHOUT the BDF floor — two-approximation formula only.

    Used during CSV generation to detect the exact BDF onset point so that
    transient stems are clipped there rather than drawn flat past the junction.
    """
    if tD <= 0.0:
        return 1e6
    qD_early = 1.0 / math.sqrt(math.pi * tD) if tD > 1e-14 else 1e6
    arg = 4.0 * tD / _GAMMA_EM
    if arg > 1.0:
        log_denom = 0.5 * math.log(arg)
        qD_log = 1.0 / log_denom
        return min(qD_log, qD_early)
    return qD_early


def _fetkovich_transient_qD(tD: float, re_rw: float) -> float:
    """Dimensionless rate qD during transient phase — floored at BDF onset.

    Used at runtime (overlay rendering) where the floor prevents the curve from
    dropping below qD_pss. For CSV generation use _fetkovich_transient_qD_raw
    and filter points instead of clamping, to avoid a flat horizontal extension.
    """
    if tD <= 0.0:
        return 1e6
    qD_pss = 1.0 / (math.log(re_rw) - 0.5)
    return max(_fetkovich_transient_qD_raw(tD), qD_pss)


# ---------------------------------------------------------------------------
# Palacio-Blasingame integrals  (Eq. 14-15 SPE-25909)
# ---------------------------------------------------------------------------

def _compute_pb_integrals(
    tDd_arr: np.ndarray, qDd_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (qDdi, qDdid) from qDd on a log-spaced tDd grid.

    qDdi  = (1/tDd) * ∫₀^tDd qDd dt          (normalized cumulative)
    qDdid = −tDd * d(qDdi)/dtDd               (derivative of normalized cumulative)

    Implementation notes
    --------------------
    * Trapezoidal integration is performed in **linear** tDd space.
    * The derivative qDdid is computed via np.gradient in **log(tDd)** space.
      On log-spaced grids this is far more stable than a linear-space gradient
      because the step sizes in log space are uniform, which prevents the
      artificial oscillations that arise when np.gradient sees tiny absolute
      steps at small tDd and large steps at large tDd.

      Chain rule:  d(qDdi)/d(tDd) = d(qDdi)/d(ln tDd) * (1/tDd)
      So:          qDdid = |−tDd · d(qDdi)/d(tDd)|
                         = |d(qDdi)/d(ln tDd)|
    """
    n = len(tDd_arr)
    integral = np.zeros(n)
    for i in range(1, n):
        dt = tDd_arr[i] - tDd_arr[i - 1]
        integral[i] = integral[i - 1] + 0.5 * (qDd_arr[i] + qDd_arr[i - 1]) * dt

    # qDdi = integral / tDd  (handle tDd=0 edge: use qDd[0] as limit)
    with np.errstate(divide="ignore", invalid="ignore"):
        qDdi = np.where(tDd_arr > 0, integral / tDd_arr, qDd_arr)

    # qDdid via log-space gradient — stable on log-spaced grids
    log_tDd = np.log(np.maximum(tDd_arr, 1e-30))
    d_qDdi_log = np.gradient(qDdi, log_tDd)   # d(qDdi)/d(ln tDd)
    qDdid = np.abs(d_qDdi_log)                # = tDd · |d(qDdi)/dtDd|
    # Clamp noise floor
    qDdid = np.maximum(qDdid, 1e-10)

    return qDdi, qDdid


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    n_curves = len({r["curve_id"] for r in rows})
    print(f"  {path.name}: {n_curves} curves, {len(rows)} points")


# ---------------------------------------------------------------------------
# FETKOVICH (SPE-4629)
# ---------------------------------------------------------------------------

_RE_RW_LIST = [10, 20, 50, 100, 200, 500, 1000]
_B_VALUES   = [0.0, 0.3, 0.5, 0.8, 1.0]

_FET_SOURCE = "Fetkovich SPE-4629 1980 Eq.20-21"


def generate_fetkovich() -> list[dict]:
    rows: list[dict] = []

    # BDF — Arps decline curves
    tDd_bdf = np.logspace(-4, 3, 100)
    for b in _B_VALUES:
        b_str = str(b).replace(".", "_")
        cid = f"fetkovich_bdf_b_{b_str}"
        qDd = _arps_qdd(tDd_bdf, b)
        for x, y in zip(tDd_bdf, qDd):
            if float(y) > 1e-9:
                rows.append(dict(
                    method="fetkovich", curve_id=cid,
                    curve_family="arps_bdf",
                    x=round(float(x), 10), y=round(float(y), 10),
                    x_label="tDd", y_label="qDd",
                    source=_FET_SOURCE, status="validated",
                    notes=f"BDF Arps b={b}",
                ))

    # Transient stems — radial flow before boundary effects.
    # Stems are clipped at the BDF junction (qDd = 1.0) using the raw (unflored)
    # formula so no flat horizontal extension is drawn past the junction.
    # Start at 1e-8 so re/rw=1000 (junction near tDd≈4e-6) is still captured.
    tDd_trans = np.logspace(-8, 0, 120)
    for re_rw in _RE_RW_LIST:
        cid = f"fetkovich_transient_rerw_{re_rw}"
        F_BDF = 0.5 * (re_rw**2 - 1) * (math.log(re_rw) - 0.5)
        norm = math.log(re_rw) - 0.5
        qD_pss = 1.0 / (math.log(re_rw) - 0.5)
        for tDd in tDd_trans:
            qD_raw = _fetkovich_transient_qD_raw(float(tDd) * F_BDF)
            if qD_raw <= qD_pss:
                break  # reached BDF onset — stop stem here
            qDd = qD_raw * norm
            if qDd > 1e-9:
                rows.append(dict(
                    method="fetkovich", curve_id=cid,
                    curve_family="transient_stem",
                    x=round(float(tDd), 10), y=round(float(qDd), 10),
                    x_label="tDd", y_label="qDd",
                    source=_FET_SOURCE, status="validated",
                    notes=f"Transient radial re/rw={re_rw}",
                ))

    return rows


# ---------------------------------------------------------------------------
# PALACIO-BLASINGAME (SPE-25909)
# ---------------------------------------------------------------------------

_PB_SOURCE = "Palacio-Blasingame SPE-25909 1993 Eq.12-15"


def generate_palacio_blasingame() -> list[dict]:
    rows: list[dict] = []

    # BDF: 3 series per b value
    tDd_bdf = np.logspace(-4, 3, 200)
    for b in _B_VALUES:
        b_str = str(b).replace(".", "_")
        qDd_arr = _arps_qdd(tDd_bdf, b)
        qDdi_arr, qDdid_arr = _compute_pb_integrals(tDd_bdf, qDd_arr)

        for series, y_arr, y_lbl in [
            ("qDd",   qDd_arr,  "qDd"),
            ("qDdi",  qDdi_arr, "qDdi"),
            ("qDdid", qDdid_arr,"qDdid"),
        ]:
            cid = f"pb_bdf_b_{b_str}_{series}"
            for x, y in zip(tDd_bdf, y_arr):
                if float(y) > 1e-9:
                    rows.append(dict(
                        method="palacio_blasingame", curve_id=cid,
                        curve_family="arps_bdf",
                        x=round(float(x), 10), y=round(float(y), 10),
                        x_label="tDd", y_label=y_lbl,
                        source=_PB_SOURCE, status="validated",
                        notes=f"BDF b={b} series={series}",
                    ))

    # Transient stems — 3 series per re/rw.
    # Clip at BDF junction using raw (unflored) formula to avoid flat extension.
    tDd_trans = np.logspace(-8, 0, 120)
    for re_rw in _RE_RW_LIST:
        F_BDF = 0.5 * (re_rw**2 - 1) * (math.log(re_rw) - 0.5)
        norm  = math.log(re_rw) - 0.5
        qD_pss = 1.0 / (math.log(re_rw) - 0.5)
        _tDd_list: list[float] = []
        _qDd_list: list[float] = []
        for t in tDd_trans:
            qD_raw = _fetkovich_transient_qD_raw(float(t) * F_BDF)
            if qD_raw <= qD_pss:
                break
            _tDd_list.append(float(t))
            _qDd_list.append(qD_raw * norm)
        if len(_tDd_list) < 2:
            continue
        tDd_stem = np.array(_tDd_list)
        qDd_arr  = np.array(_qDd_list)
        qDdi_arr, qDdid_arr = _compute_pb_integrals(tDd_stem, qDd_arr)

        for series, y_arr, y_lbl in [
            ("qDd",   qDd_arr,  "qDd"),
            ("qDdi",  qDdi_arr, "qDdi"),
            ("qDdid", qDdid_arr,"qDdid"),
        ]:
            cid = f"pb_transient_rerw_{re_rw}_{series}"
            for x, y in zip(tDd_stem, y_arr):
                if float(y) > 1e-9:
                    rows.append(dict(
                        method="palacio_blasingame", curve_id=cid,
                        curve_family="transient_stem",
                        x=round(float(x), 10), y=round(float(y), 10),
                        x_label="tDd", y_label=y_lbl,
                        source=_PB_SOURCE, status="validated",
                        notes=f"Transient re/rw={re_rw} series={series}",
                    ))

    return rows


# ---------------------------------------------------------------------------
# AGARWAL-GARDNER (SPE-49222)
# ---------------------------------------------------------------------------

_AG_SOURCE  = "Agarwal-Gardner SPE-49222 1998 Eq.3"
_AG_RE_RW_REF = 100.0   # reference geometry for BDF x-axis normalization


def generate_agarwal_gardner() -> list[dict]:
    """A-G uses tDA = tD / re_rw² and qD (not normalized by ln(re/rw)-0.5)."""
    rows: list[dict] = []

    # BDF — same Arps shapes, relabelled to tDA/qD axes
    F_BDF_ref = 0.5 * (_AG_RE_RW_REF**2 - 1) * (math.log(_AG_RE_RW_REF) - 0.5)
    qD_pss_ref = 1.0 / (math.log(_AG_RE_RW_REF) - 0.5)

    tDd_bdf = np.logspace(-4, 3, 100)
    for b in _B_VALUES:
        b_str = str(b).replace(".", "_")
        cid = f"ag_bdf_b_{b_str}"
        qDd = _arps_qdd(tDd_bdf, b)
        tDA = tDd_bdf * F_BDF_ref / (_AG_RE_RW_REF**2)
        qD  = qDd * qD_pss_ref
        for x, y in zip(tDA, qD):
            if float(x) > 1e-14 and float(y) > 1e-9:
                rows.append(dict(
                    method="agarwal_gardner", curve_id=cid,
                    curve_family="radial_bdf",
                    x=round(float(x), 12), y=round(float(y), 10),
                    x_label="tDA", y_label="qD",
                    source=_AG_SOURCE, status="validated",
                    notes=f"BDF Arps b={b} re/rw_ref={int(_AG_RE_RW_REF)}",
                ))

    # Transient stems in tDA/qD space — clipped at BDF onset.
    tDd_trans = np.logspace(-8, 0, 120)
    for re_rw in _RE_RW_LIST:
        cid = f"ag_transient_rerw_{re_rw}"
        F_BDF = 0.5 * (re_rw**2 - 1) * (math.log(re_rw) - 0.5)
        qD_pss = 1.0 / (math.log(re_rw) - 0.5)
        for tDd in tDd_trans:
            tD  = float(tDd) * F_BDF
            qD_raw = _fetkovich_transient_qD_raw(tD)
            if qD_raw <= qD_pss:
                break
            tDA = tD / (re_rw**2)
            if tDA > 1e-14 and qD_raw > 1e-9:
                rows.append(dict(
                    method="agarwal_gardner", curve_id=cid,
                    curve_family="radial_transient",
                    x=round(float(tDA), 12), y=round(float(qD_raw), 10),
                    x_label="tDA", y_label="qD",
                    source=_AG_SOURCE, status="validated",
                    notes=f"Transient radial re/rw={re_rw}",
                ))

    return rows


# ---------------------------------------------------------------------------
# BLASINGAME composite (solver numérico implícito)
# ---------------------------------------------------------------------------

_BL_SOURCE = "blasingame.py implicit FD solver — radial constant-pressure (sin validación cuantitativa)"


def generate_blasingame() -> list[dict]:
    """Genera curvas tipo Blasingame compuestas (qDd, qDdi, qDdid vs tcDd).

    Usa el solver numérico implícito en blasingame.py para 8 valores de reD.
    Estado: demo — pendiente validación vs Software Comercial.
    """
    from src.rta_type_curves.blasingame import BlasingameCurveConfig, generate_blasingame_curves

    cfg = BlasingameCurveConfig()
    curve_set = generate_blasingame_curves(cfg)
    df = curve_set.curves

    rows: list[dict] = []
    eps = cfg.eps

    for re_d, grp in df.groupby("re_d", sort=True):
        re_label = int(re_d) if re_d < 1e6 else int(re_d)
        # Nombre compacto: 10, 20, 50, 100, 1e3, 1e4, 1e5, 1e6
        re_str = (
            f"{int(re_d)}"
            if re_d < 1000
            else f"{int(re_d / 10 ** int(math.log10(re_d)))}"
                  + "e" + str(int(math.log10(re_d)))
        )

        for series, col, y_lbl in [
            ("qDd",   "q_dd",   "qDd"),
            ("qDdi",  "q_ddi",  "qDdi"),
            ("qDdid", "q_ddid", "qDdid"),
        ]:
            cid = f"bl_reD_{int(re_d)}_{series}"
            for _, pt in grp.iterrows():
                x = round(float(pt["t_c_dd"]), 10)
                y = round(float(pt[col]),      10)
                # Filter AFTER rounding so 0.0 artefacts are excluded
                if x > 0 and y > 0:
                    rows.append(dict(
                        method="blasingame",
                        curve_id=cid,
                        curve_family="blasingame_composite",
                        x=x,
                        y=y,
                        x_label="tcDd",
                        y_label=y_lbl,
                        source=_BL_SOURCE,
                        status="demo",
                        notes=f"reD={int(re_d)} series={series}",
                    ))

    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Output dir: {OUTPUT_DIR}")

    print("\nFetkovich (SPE-4629)…")
    fet_rows = generate_fetkovich()
    _write_csv(fet_rows, OUTPUT_DIR / "fetkovich_base.csv")

    print("\nPalacio-Blasingame (SPE-25909)…")
    pb_rows = generate_palacio_blasingame()
    _write_csv(pb_rows, OUTPUT_DIR / "palacio_blasingame_base.csv")

    print("\nAgarwal-Gardner (SPE-49222)…")
    ag_rows = generate_agarwal_gardner()
    _write_csv(ag_rows, OUTPUT_DIR / "agarwal_gardner_base.csv")

    print("\nBlasingame composite (solver numérico)…")
    bl_rows = generate_blasingame()
    _write_csv(bl_rows, OUTPUT_DIR / "blasingame_base.csv")

    total = len(fet_rows) + len(pb_rows) + len(ag_rows) + len(bl_rows)
    n_all = sum(
        len({r["curve_id"] for r in rows})
        for rows in (fet_rows, pb_rows, ag_rows, bl_rows)
    )
    print(f"\nTotal: {n_all} curves, {total} points OK")


if __name__ == "__main__":
    main()
