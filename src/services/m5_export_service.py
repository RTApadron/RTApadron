"""Servicio de exportación M5 — genera CSV, JSON, Excel y PDF desde WellResultsSummary.

Formatos:
    CSV   — resumen plano consolidado (una fila por pozo, una sección por módulo)
    JSON  — volcado completo de WellResultsSummary (Pydantic model_dump)
    Excel — xlsxwriter, una hoja por módulo (M1 Historia / M2 PVT / M3 DCA / M4 RTA)
    PDF   — matplotlib PdfPages (sin dependencias extra): métricas + gráfico comparativo

Todas las funciones devuelven bytes listos para st.download_button o escritura a disco.
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.domain.m5_models import WellResultsSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value: float | None, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}" if value is not None else ""


def _now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def export_csv_bytes(summary: WellResultsSummary) -> bytes:
    """Genera un CSV consolidado con una sección por módulo."""
    sections: list[dict] = []

    # ── M1 info ──────────────────────────────────────────────────────────────
    wi = summary.well_info
    sections.append({
        "seccion": "M1_bien_info",
        "campo": "well_id", "valor": wi.well_id,
    })
    for k, v in {
        "well_name": wi.well_name,
        "field_name": wi.field_name,
        "api_gravity": _fmt(wi.api_gravity, 1),
        "temp_f": _fmt(wi.temp_f, 1),
        "history_points": wi.history_points,
        "date_from": str(wi.date_from) if wi.date_from else "",
        "date_to": str(wi.date_to) if wi.date_to else "",
        "qo_avg_stb_d": _fmt(wi.qo_avg_stb_d, 1),
        "qo_max_stb_d": _fmt(wi.qo_max_stb_d, 1),
    }.items():
        sections.append({"seccion": "M1_bien_info", "campo": k, "valor": str(v)})

    # ── M2 PVT ───────────────────────────────────────────────────────────────
    if summary.pvt:
        pvt = summary.pvt
        for k, v in {
            "oil_corr": pvt.oil_corr,
            "calibrated": pvt.calibrated,
            "avg_bo_rb_stb": _fmt(pvt.avg_bo_rb_stb, 4),
            "avg_rs_scf_stb": _fmt(pvt.avg_rs_scf_stb, 1),
            "avg_mu_o_cp": _fmt(pvt.avg_mu_o_cp, 3),
            "pb_psia": _fmt(pvt.pb_psia, 0),
            "status": pvt.status,
        }.items():
            sections.append({"seccion": "M2_PVT", "campo": k, "valor": str(v or "")})

    # ── M3 DCA ───────────────────────────────────────────────────────────────
    if summary.dca:
        dca = summary.dca
        sections.append({"seccion": "M3_DCA", "campo": "best_model", "valor": dca.best_model or ""})
        for m in dca.models:
            prefix = f"dca_{m.model}"
            for k, v in {
                "qi_stb_d": _fmt(m.qi_stb_d, 1),
                "di_nominal_d": _fmt(m.di_nominal_d, 6),
                "b": _fmt(m.b, 3),
                "eur_stb": _fmt(m.eur_stb, 0),
                "r2": _fmt(m.r2, 4),
                "rmse_stb_d": _fmt(m.rmse_stb_d, 1),
                "forecast_days": str(m.forecast_days),
            }.items():
                sections.append({"seccion": "M3_DCA", "campo": f"{prefix}_{k}", "valor": v})

    # ── M4 RTA ───────────────────────────────────────────────────────────────
    if summary.rta:
        rta = summary.rta
        for k, v in {
            "method": rta.method,
            "kh_md_ft": _fmt(rta.kh_md_ft, 2),
            "k_md": _fmt(rta.k_md, 4),
            "n_vol_stb": _fmt(rta.n_vol_stb, 0),
            "re_ft": _fmt(rta.re_ft, 0),
            "area_acres": _fmt(rta.area_acres, 2),
            "x_multiplier": _fmt(rta.x_multiplier, 4),
            "y_multiplier": _fmt(rta.y_multiplier, 4),
            "status": rta.status,
        }.items():
            sections.append({"seccion": "M4_RTA", "campo": k, "valor": str(v or "")})

    # ── Metadata ─────────────────────────────────────────────────────────────
    sections.append({"seccion": "meta", "campo": "generated_at", "valor": _now_str()})
    sections.append({"seccion": "meta", "campo": "ecoRTA_version", "valor": "m5-export-0.1"})

    df = pd.DataFrame(sections)
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def export_json_bytes(summary: WellResultsSummary) -> bytes:
    """Volcado completo del WellResultsSummary como JSON indentado."""
    data = summary.model_dump(mode="json")
    data["_export_meta"] = {"generated_at": _now_str(), "ecoRTA_version": "m5-export-0.1"}
    return json.dumps(data, indent=2, default=str).encode("utf-8")


# ---------------------------------------------------------------------------
# Excel (xlsxwriter)
# ---------------------------------------------------------------------------

def export_excel_bytes(summary: WellResultsSummary) -> bytes:
    """Genera un Excel con una hoja por módulo usando xlsxwriter."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book
        # Formatos
        fmt_title = wb.add_format({
            "bold": True, "font_size": 13, "font_color": "#1a1a2e",
            "bg_color": "#e8f4f8", "border": 1,
        })
        fmt_header = wb.add_format({
            "bold": True, "bg_color": "#2c3e50", "font_color": "white",
            "border": 1, "align": "center",
        })
        fmt_demo = wb.add_format({
            "italic": True, "font_color": "#8e44ad",
        })
        fmt_num = wb.add_format({"num_format": "#,##0.0000", "border": 1})
        fmt_int = wb.add_format({"num_format": "#,##0", "border": 1})
        fmt_cell = wb.add_format({"border": 1})

        def _write_header(ws, row: int, cols: list[str]) -> None:
            for c, label in enumerate(cols):
                ws.write(row, c, label, fmt_header)

        # ── Hoja Resumen ─────────────────────────────────────────────────────
        ws_res = writer.sheets.get("Resumen") or wb.add_worksheet("Resumen")
        writer.sheets["Resumen"] = ws_res
        wi = summary.well_info
        ws_res.write(0, 0, "ecoRTA — M5 Resultados Integrados", fmt_title)
        ws_res.write(1, 0, f"Pozo: {wi.well_id}  |  Generado: {_now_str()}", fmt_cell)
        data_res = [
            ["Well ID", wi.well_id],
            ["Nombre", wi.well_name or ""],
            ["Campo", wi.field_name or ""],
            ["API (°)", wi.api_gravity],
            ["Temp. °F", wi.temp_f],
            ["Puntos historia", wi.history_points],
            ["Fecha inicio", str(wi.date_from) if wi.date_from else ""],
            ["Fecha fin", str(wi.date_to) if wi.date_to else ""],
            ["qo promedio (STB/d)", wi.qo_avg_stb_d],
            ["qo máximo (STB/d)", wi.qo_max_stb_d],
        ]
        _write_header(ws_res, 3, ["Campo", "Valor"])
        for i, (label, val) in enumerate(data_res):
            ws_res.write(4 + i, 0, label, fmt_cell)
            ws_res.write(4 + i, 1, val if val is not None else "", fmt_cell)
        ws_res.set_column(0, 0, 25)
        ws_res.set_column(1, 1, 20)

        # ── Hoja M2 PVT ──────────────────────────────────────────────────────
        ws_pvt = wb.add_worksheet("M2_PVT")
        writer.sheets["M2_PVT"] = ws_pvt
        ws_pvt.write(0, 0, "M2 — Propiedades PVT representativas", fmt_title)
        if summary.pvt:
            pvt = summary.pvt
            data_pvt = [
                ["Correlación", pvt.oil_corr or ""],
                ["Calibrado", "Sí" if pvt.calibrated else "No"],
                ["Pb (psia)", pvt.pb_psia],
                ["Bo promedio (RB/STB)", pvt.avg_bo_rb_stb],
                ["Rs promedio (scf/STB)", pvt.avg_rs_scf_stb],
                ["μo promedio (cp)", pvt.avg_mu_o_cp],
                ["Status", pvt.status],
            ]
            _write_header(ws_pvt, 2, ["Propiedad", "Valor"])
            for i, (label, val) in enumerate(data_pvt):
                ws_pvt.write(3 + i, 0, label, fmt_cell)
                ws_pvt.write(3 + i, 1, val if val is not None else "", fmt_cell)
        else:
            ws_pvt.write(2, 0, "M2 no disponible — ejecutar pipeline M1-M2.", fmt_demo)
        ws_pvt.set_column(0, 0, 28)
        ws_pvt.set_column(1, 1, 18)

        # ── Hoja M3 DCA ──────────────────────────────────────────────────────
        ws_dca = wb.add_worksheet("M3_DCA")
        writer.sheets["M3_DCA"] = ws_dca
        ws_dca.write(0, 0, "M3 — Declinación Arps (DCA)", fmt_title)
        if summary.dca and summary.dca.models:
            dca = summary.dca
            ws_dca.write(1, 0, f"Mejor modelo: {dca.best_model or '—'}", fmt_cell)
            cols_dca = ["Modelo", "qi (STB/d)", "Di (/d)", "b", "EUR (STB)", "EUR (MM STB)", "R²", "RMSE (STB/d)", "Puntos"]
            _write_header(ws_dca, 3, cols_dca)
            for i, m in enumerate(dca.models):
                row = 4 + i
                ws_dca.write(row, 0, m.model.capitalize(), fmt_cell)
                ws_dca.write(row, 1, m.qi_stb_d, fmt_num)
                ws_dca.write(row, 2, m.di_nominal_d, fmt_num)
                ws_dca.write(row, 3, m.b, fmt_num)
                ws_dca.write(row, 4, m.eur_stb, fmt_int)
                ws_dca.write(row, 5, m.eur_stb / 1e6, fmt_num)
                ws_dca.write(row, 6, m.r2, fmt_num)
                ws_dca.write(row, 7, m.rmse_stb_d, fmt_num)
                ws_dca.write(row, 8, m.n_points, fmt_int)
            ws_dca.set_column(0, 0, 16)
            ws_dca.set_column(1, 8, 14)
        else:
            ws_dca.write(2, 0, "M3 no disponible — ejecutar DCA primero.", fmt_demo)

        # ── Hoja M4 RTA ──────────────────────────────────────────────────────
        ws_rta = wb.add_worksheet("M4_RTA")
        writer.sheets["M4_RTA"] = ws_rta
        ws_rta.write(0, 0, "M4 — Parámetros RTA (DEMO — curvas tipo no validadas)", fmt_title)
        if summary.rta:
            rta = summary.rta
            data_rta = [
                ["Método", (rta.method or "").replace("_", " ").title()],
                ["kh (mD·ft)", rta.kh_md_ft],
                ["k (mD)", rta.k_md],
                ["OOIP volumétrico (STB)", rta.n_vol_stb],
                ["OOIP volumétrico (MM STB)", rta.n_vol_stb / 1e6 if rta.n_vol_stb else None],
                ["re (ft)", rta.re_ft],
                ["Área drene (acres)", rta.area_acres],
                ["Multiplicador X", rta.x_multiplier],
                ["Multiplicador Y", rta.y_multiplier],
                ["Status", rta.status],
            ]
            ws_rta.write(1, 0, "⚠️ Resultados preliminares — no usar para toma de decisiones.", fmt_demo)
            _write_header(ws_rta, 3, ["Parámetro", "Valor"])
            for i, (label, val) in enumerate(data_rta):
                ws_rta.write(4 + i, 0, label, fmt_cell)
                ws_rta.write(4 + i, 1, val if val is not None else "", fmt_cell)

            # Advertencias QC
            if rta.qc_warnings:
                row_off = 4 + len(data_rta) + 2
                ws_rta.write(row_off, 0, "Advertencias QC:", fmt_header)
                for j, w in enumerate(rta.qc_warnings):
                    ws_rta.write(row_off + 1 + j, 0, w, fmt_demo)
        else:
            ws_rta.write(2, 0, "M4 no disponible — realizar matching en UI M4.", fmt_demo)
        ws_rta.set_column(0, 0, 30)
        ws_rta.set_column(1, 1, 20)

        # ── Hoja Comparativo ─────────────────────────────────────────────────
        ws_comp = wb.add_worksheet("Comparativo")
        writer.sheets["Comparativo"] = ws_comp
        ws_comp.write(0, 0, "M5 — Dashboard comparativo de volúmenes", fmt_title)
        _write_header(ws_comp, 2, ["Fuente", "Método", "Volumen (STB)", "Volumen (MM STB)", "R² / Confianza", "Status"])
        row_c = 3
        if summary.dca:
            for m in summary.dca.models:
                ws_comp.write(row_c, 0, "DCA M3", fmt_cell)
                ws_comp.write(row_c, 1, m.model.capitalize(), fmt_cell)
                ws_comp.write(row_c, 2, m.eur_stb, fmt_int)
                ws_comp.write(row_c, 3, m.eur_stb / 1e6, fmt_num)
                ws_comp.write(row_c, 4, m.r2, fmt_num)
                ws_comp.write(row_c, 5, "calculado", fmt_cell)
                row_c += 1
        if summary.rta and summary.rta.n_vol_stb:
            ws_comp.write(row_c, 0, "RTA M4", fmt_cell)
            ws_comp.write(row_c, 1, f"OOIP vol. ({summary.rta.method or ''})", fmt_demo)
            ws_comp.write(row_c, 2, summary.rta.n_vol_stb, fmt_int)
            ws_comp.write(row_c, 3, summary.rta.n_vol_stb / 1e6, fmt_num)
            ws_comp.write(row_c, 4, "DEMO", fmt_demo)
            ws_comp.write(row_c, 5, "demo", fmt_demo)
        ws_comp.set_column(0, 0, 12)
        ws_comp.set_column(1, 1, 28)
        ws_comp.set_column(2, 5, 16)

    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# PDF (matplotlib PdfPages — sin dependencias extra)
# ---------------------------------------------------------------------------

def export_pdf_bytes(summary: WellResultsSummary) -> bytes:
    """Genera un PDF multi-página con matplotlib PdfPages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()

    with PdfPages(buf) as pdf:
        # ── Página 1: Portada + resumen del pozo ────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        wi = summary.well_info
        lines = [
            ("ecoRTA — Reporte Integrado de Pozo", 0.93, 16, "bold", "#1a1a2e"),
            (f"Pozo: {wi.well_id}  |  {wi.field_name or ''}  |  API {wi.api_gravity or '—'}°", 0.87, 12, "normal", "#2c3e50"),
            (f"Generado: {_now_str()}", 0.83, 9, "normal", "#7f8c8d"),
            ("─" * 80, 0.81, 9, "normal", "#bdc3c7"),
            ("INFORMACIÓN DEL POZO (M1)", 0.77, 11, "bold", "#2c3e50"),
            (f"Historia: {wi.history_points} puntos  |  {wi.date_from} → {wi.date_to}", 0.73, 10, "normal", "#333"),
            (f"qo promedio: {_fmt(wi.qo_avg_stb_d, 0)} STB/d  |  qo máximo: {_fmt(wi.qo_max_stb_d, 0)} STB/d", 0.69, 10, "normal", "#333"),
        ]
        if summary.pvt:
            pvt = summary.pvt
            lines += [
                ("─" * 80, 0.65, 9, "normal", "#bdc3c7"),
                ("PROPIEDADES PVT (M2)", 0.61, 11, "bold", "#2c3e50"),
                (f"Correlación: {pvt.oil_corr or '—'}  |  Calibrado: {'Sí' if pvt.calibrated else 'No'}", 0.57, 10, "normal", "#333"),
                (f"Bo = {_fmt(pvt.avg_bo_rb_stb, 4)} RB/STB  |  Rs = {_fmt(pvt.avg_rs_scf_stb, 0)} scf/STB  |  μo = {_fmt(pvt.avg_mu_o_cp, 3)} cp  |  Pb = {_fmt(pvt.pb_psia, 0)} psia", 0.53, 10, "normal", "#333"),
            ]
        y_next = 0.45 if summary.pvt else 0.61
        if summary.dca and summary.dca.models:
            dca = summary.dca
            lines += [
                ("─" * 80, y_next, 9, "normal", "#bdc3c7"),
                ("DECLINACIÓN ARPS — DCA (M3)", y_next - 0.04, 11, "bold", "#2c3e50"),
                (f"Mejor modelo: {dca.best_model or '—'}", y_next - 0.08, 10, "normal", "#333"),
            ]
            for j, m in enumerate(dca.models):
                lines.append((
                    f"  {m.model.capitalize()}: qi={_fmt(m.qi_stb_d, 0)} STB/d  EUR={m.eur_stb/1e6:.3f} MM STB  R²={_fmt(m.r2, 4)}",
                    y_next - 0.12 - j * 0.04, 9, "normal", "#333",
                ))
        if summary.rta:
            rta = summary.rta
            y_rta = y_next - 0.12 - len(summary.dca.models if summary.dca else []) * 0.04 - 0.06
            lines += [
                ("─" * 80, y_rta, 9, "normal", "#bdc3c7"),
                ("PARÁMETROS RTA (M4) — ⚠️ DEMO", y_rta - 0.04, 11, "bold", "#8e44ad"),
                (f"Método: {(rta.method or '').replace('_', ' ').title()}  |  kh = {_fmt(rta.kh_md_ft, 2)} mD·ft  |  k = {_fmt(rta.k_md, 4)} mD", y_rta - 0.08, 9, "normal", "#8e44ad"),
                (f"OOIP volumétrico = {rta.n_vol_stb/1e6:.3f} MM STB" if rta.n_vol_stb else "OOIP volumétrico: —", y_rta - 0.12, 9, "normal", "#8e44ad"),
            ]

        for text, y, size, weight, color in lines:
            ax.text(0.05, y, text, transform=ax.transAxes,
                    fontsize=size, fontweight=weight, color=color,
                    verticalalignment="top", wrap=True)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Página 2: Gráfico comparativo de volúmenes ──────────────────────
        labels: list[str] = []
        values: list[float] = []
        colors_bar: list[str] = []

        if summary.dca and summary.dca.models:
            for m in summary.dca.models:
                labels.append(f"EUR\n{m.model.capitalize()}\n(DCA)")
                values.append(m.eur_stb / 1e6)
                colors_bar.append({"exponential": "#3498db", "hyperbolic": "#2ecc71", "harmonic": "#e67e22"}.get(m.model, "#95a5a6"))
        if summary.rta and summary.rta.n_vol_stb:
            labels.append("OOIP\nVolumétr.\n(RTA)")
            values.append(summary.rta.n_vol_stb / 1e6)
            colors_bar.append("#8e44ad")

        if labels:
            fig2, ax2 = plt.subplots(figsize=(8.5, 5))
            bars = ax2.bar(labels, values, color=colors_bar, edgecolor="white", linewidth=1.2)
            for bar, val in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(values) * 0.015,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax2.set_ylabel("Volumen (MM STB)", fontsize=11)
            ax2.set_title("Comparativo de volúmenes — M3 DCA vs M4 RTA", fontsize=13, fontweight="bold", pad=15)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.set_ylim(0, max(values) * 1.2)
            ax2.annotate("⚠️ OOIP RTA es DEMO (curvas tipo no validadas)",
                         xy=(0.5, -0.18), xycoords="axes fraction",
                         ha="center", fontsize=8, color="#8e44ad", style="italic")
            fig2.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

        # Metadata del PDF
        pdf.infodict()["Title"] = f"ecoRTA — Reporte {summary.well_id}"
        pdf.infodict()["Author"] = "ecoRTA M5"
        pdf.infodict()["Subject"] = "Rate Transient Analysis — Resultados integrados"

    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Guardar a disco (helper)
# ---------------------------------------------------------------------------

def save_all_exports(summary: WellResultsSummary, output_dir: Path | str) -> dict[str, Path]:
    """Escribe CSV, JSON, Excel y PDF en output_dir. Devuelve rutas generadas."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    wid = summary.well_id

    paths: dict[str, Path] = {}

    p_csv = out / f"{wid}_m5_summary.csv"
    p_csv.write_bytes(export_csv_bytes(summary))
    paths["csv"] = p_csv

    p_json = out / f"{wid}_m5_summary.json"
    p_json.write_bytes(export_json_bytes(summary))
    paths["json"] = p_json

    p_xlsx = out / f"{wid}_m5_report.xlsx"
    p_xlsx.write_bytes(export_excel_bytes(summary))
    paths["xlsx"] = p_xlsx

    p_pdf = out / f"{wid}_m5_report.pdf"
    p_pdf.write_bytes(export_pdf_bytes(summary))
    paths["pdf"] = p_pdf

    return paths
