# make_pvt_excel.py
import csv
import math
from pvt_tools import PVTInputs, compute_pvt_table

def main():
    inputs = PVTInputs(
        API=30.0,
        gamma_g=0.80,
        T_F=200.0,
        Rsb=600.0,
        pressures=list(range(500, 4501, 250))
    )
    rows = compute_pvt_table(inputs)

    # Guardar CSV
    csv_path = "sprint1_propiedades_pvt.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Escrito: {csv_path}")

    # Guardar Excel si quieres (requiere xlsxwriter)
    try:
        import xlsxwriter
        xlsx_path = "sprint1_propiedades_pvt.xlsx"
        wb = xlsxwriter.Workbook(xlsx_path)
        ws = wb.add_worksheet("PVT")

        headers = list(rows[0].keys())
        for j, h in enumerate(headers):
            ws.write(0, j, h)
        for i, row in enumerate(rows, start=1):
            for j, h in enumerate(headers):
                ws.write(i, j, row[h])
        wb.close()
        print(f"Escrito: {xlsx_path}")
    except ImportError:
        print("xlsxwriter no instalado; se generó solo CSV.")

if __name__ == "__main__":
    main()
