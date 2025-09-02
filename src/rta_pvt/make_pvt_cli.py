# src/rta_pvt/make_pvt_cli.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from pvt_tools import (
    PVTInputs,
    compute_pvt_table,
    gamma_o_from_api,
    a_standing,
    pb_standing,
)

def main():
    parser = argparse.ArgumentParser(description="Genera tabla PVT y gráficos (con anotación de Pb).")
    parser.add_argument("--api", type=float, required=True, help="Gravedad API del crudo")
    parser.add_argument("--gamma-g", type=float, required=True, help="Gravedad específica del gas (aire=1)")
    parser.add_argument("--temp", type=float, required=True, help="Temperatura en °F")
    parser.add_argument("--rsb", type=float, required=True, help="Gas disuelto a Pb (scf/STB)")
    parser.add_argument("--pmin", type=int, default=500, help="Presión mínima (psia)")
    parser.add_argument("--pmax", type=int, default=4500, help="Presión máxima (psia)")
    parser.add_argument("--step", type=int, default=250, help="Paso de presión (psia)")
    args = parser.parse_args()

    # Validaciones simples
    if args.pmin <= 0 or args.pmax <= 0 or args.step <= 0:
        raise ValueError("pmin, pmax y step deben ser positivos.")
    if args.pmax < args.pmin:
        raise ValueError("pmax debe ser mayor o igual que pmin.")

    pressures = list(range(args.pmin, args.pmax + args.step, args.step))

    # Resumen previo
    gamma_o = gamma_o_from_api(args.api)
    a = a_standing(args.api, args.temp)
    Pb = pb_standing(args.api, args.temp, args.gamma_g, args.rsb)

    print("\n=== Resumen de entradas / parámetros ===")
    print(f"API           : {args.api:.3f} °API")
    print(f"gamma_g       : {args.gamma_g:.5f} (aire=1)")
    print(f"T_F           : {args.temp:.3f} °F")
    print(f"Rsb           : {args.rsb:.3f} scf/STB")
    print(f"gamma_o       : {gamma_o:.5f} (agua=1)")
    print(f"a (Standing)  : {a:.6f}")
    print(f"Pb (Standing) : {Pb:.3f} psia")
    print(f"Presiones     : {pressures[0]} .. {pressures[-1]} (step={args.step}) -> {len(pressures)} pts (+1 en Pb)")
    print("=======================================\n")

    # Calcular tabla (incluye punto en Pb)
    inputs = PVTInputs(API=args.api, gamma_g=args.gamma_g, T_F=args.temp, Rsb=args.rsb, pressures=pressures)
    df = pd.DataFrame(compute_pvt_table(inputs))

    # Carpeta de salida
    outdir = os.path.join(os.path.dirname(__file__), "..", "..", "output")
    os.makedirs(outdir, exist_ok=True)

    # Guardar CSV
    csv_path = os.path.join(outdir, "pvt_cli_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Escrito: {csv_path}")

    # Guardar Excel
    try:
        import xlsxwriter  # noqa: F401
        xlsx_path = os.path.join(outdir, "pvt_cli_results.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="PVT", index=False)
        print(f"Escrito: {xlsx_path}")
    except ImportError:
        print("xlsxwriter no instalado; se generó solo CSV.")

    # ---------- Graficación con anotación de Pb ----------
    def make_plot(x, y, xlabel, ylabel, title, fname, Pb_value):
        plt.figure()
        plt.plot(x, y, marker="o")
        # Línea vertical Pb con etiqueta en leyenda
        plt.axvline(Pb_value, linestyle="--", color="red", label=f"Pb = {Pb_value:.1f} psia")

        # Anotación del valor de Pb (flecha + texto)
        ymin, ymax = float(min(y)), float(max(y))
        y_text = ymin + 0.05 * (ymax - ymin)  # 5% por encima del mínimo
        plt.annotate(
            f"Pb = {Pb_value:.1f} psia",
            xy=(Pb_value, y_text),
            xytext=(Pb_value + 0.06 * (max(x) - min(x)), y_text),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=9,
            color="red",
        )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(outdir, fname)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Escrito: {path}")

    Pb_value = float(df["Pb_psi"].iloc[0])

    make_plot(
        df["pressure_psi"], df["Rs_scf_per_STB"],
        "Presion (psia)", "Rs (scf/STB)",
        "Rs vs Presion", "cli_plot_rs.png", Pb_value
    )
    make_plot(
        df["pressure_psi"], df["Bo_bbl_per_STB"],
        "Presion (psia)", "Bo (bbl/STB)",
        "Bo vs Presion", "cli_plot_bo.png", Pb_value
    )
    make_plot(
        df["pressure_psi"], df["mu_o_cP"],
        "Presion (psia)", "Viscosidad (cP)",
        "mu_o vs Presion", "cli_plot_mu.png", Pb_value
    )

if __name__ == "__main__":
    main()
