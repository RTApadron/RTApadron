# src/rta_pvt/make_pvt_outputs.py
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from pvt_tools import PVTInputs, compute_pvt_table

def main():
    # Inputs del usuario (ajústalos libremente)
    inputs = PVTInputs(
        API=15.0,  ##<--cambia aquí
        gamma_g=0.80, ##<--cambia aquí
        T_F=250.0, ##<--cambia aquí
        Rsb=150.0, ##<--cambia aquí
        pressures=list(range(15, 5000, 300)) ##<--cambia aquí
    )

    # Crear carpeta de salida
    outdir = os.path.join(os.path.dirname(__file__), "..", "..", "output")
    os.makedirs(outdir, exist_ok=True)

    # Calcular tabla PVT
    rows = compute_pvt_table(inputs)
    df = pd.DataFrame(rows)

    # Guardar CSV
    csv_path = os.path.join(outdir, "sprint1_propiedades_pvt.csv")
    df.to_csv(csv_path, index=False)
    print(f"Escrito: {csv_path}")

    # Guardar Excel
    try:
        import xlsxwriter
        xlsx_path = os.path.join(outdir, "sprint1_propiedades_pvt.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="PVT", index=False)
        print(f"Escrito: {xlsx_path}")
    except ImportError:
        print("xlsxwriter no instalado; solo CSV generado.")

    # Graficas
    def make_plot(x, y, xlabel, ylabel, title, fname):
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.axvline(df["Pb_psi"].iloc[0], linestyle="--", color="red", label="Pb")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(outdir, fname)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Escrito: {path}")

    make_plot(df["pressure_psi"], df["Rs_scf_per_STB"], "Presion (psia)", "Rs (scf/STB)",
              "Rs vs Presion", "plot_rs_vs_p.png")
    make_plot(df["pressure_psi"], df["Bo_bbl_per_STB"], "Presion (psia)", "Bo (bbl/STB)",
              "Bo vs Presion", "plot_bo_vs_p.png")
    make_plot(df["pressure_psi"], df["mu_o_cP"], "Presion (psia)", "Viscosidad (cP)",
              "mu_o vs Presion", "plot_mu_vs_p.png")

if __name__ == "__main__":
    main()
