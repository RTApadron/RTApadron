import pandas as pd
from well_mod.io_csv import load_wells, load_survey, load_mech, load_lift, load_history
from well_mod.survey import min_curv
from well_mod.pwf import PwfInputs, estimate_pwf_v1

def main():
    wells, e1 = load_wells("data/wells.csv")
    svy,   e2 = load_survey("data/survey.csv")
    mech,  e3 = load_mech("data/mechanical_state.csv")
    lift,  e4 = load_lift("data/lift.csv")
    hist,  e5 = load_history("data/history.csv")


    errs = e1+e2+e3+e4+e5
    if errs:
        print("VALIDATION ERRORS:")
        for e in errs: print(" -", e)

    # Ejemplo para un pozo
    wid = wells[0].well_id
    svy_w = [s for s in svy if s.well_id == wid]
    tvd_table = min_curv(svy_w)
    tvd_perf = [m for m in mech if m.well_id==wid][0].perfs_bottom_ft  # usa top/bottom según criterio

    wh = [l for l in lift if l.well_id==wid][0]
    mech_w = [m for m in mech if m.well_id==wid][0]

    # Itera historia y rellena Pwf faltante
    rows = []
    for h in [h for h in hist if h.well_id==wid]:
        if h.pwf_psia and h.pwf_psia>0:
            est = h.pwf_psia
        else:
            pin = PwfInputs(
                qo_stb_d=h.qo_stb_d,
                qw_stb_d=h.qw_stb_d,
                api=h.api or wells[0].api or 30.0,
                whp_psia=wh.whp_psia or 100.0,
                tvd_perf_ft=tvd_perf,  # simplificación
                tubing_id_in=mech_w.tubing_id_in,
                length_ft=mech_w.tubing_set_depth_ft,
                Cf=0.02
            )
            est = estimate_pwf_v1(pin)
        rows.append({
            "well_id": wid, "date": h.date, "pwf_est_psia": est,
            "qo_stb_d": h.qo_stb_d, "qw_stb_d": h.qw_stb_d, "qg_mscf_d": h.qg_mscf_d
        })

    out = pd.DataFrame(rows)
    out.to_csv("pwf_estimated.csv", index=False)
    print("Escrito: pwf_estimated.csv")

if __name__ == "__main__":
    main()
