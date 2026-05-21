"""Tests para M5 — Servicio de exportación (CSV, JSON, Excel, PDF)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from src.domain.m5_models import (
    DCASummary,
    DCAModelSummary,
    PVTSummary,
    RTASummary,
    WellInfoSummary,
    WellResultsSummary,
)
from src.services.m5_export_service import (
    export_csv_bytes,
    export_json_bytes,
    export_excel_bytes,
    export_pdf_bytes,
    save_all_exports,
)


# ---------------------------------------------------------------------------
# Fixture: summary de prueba completo
# ---------------------------------------------------------------------------

@pytest.fixture()
def full_summary() -> WellResultsSummary:
    return WellResultsSummary(
        well_id="W001",
        well_info=WellInfoSummary(
            well_id="W001",
            well_name="Pozo Prueba",
            field_name="CPO-9",
            api_gravity=16.0,
            temp_f=185.0,
            history_points=30,
            qo_avg_stb_d=450.0,
            qo_max_stb_d=600.0,
            pwf_source_counts={"measured": 10, "estimated": 20},
            qc_warnings=["Advertencia QC de prueba"],
        ),
        pvt=PVTSummary(
            oil_corr="standing",
            calibrated=False,
            avg_bo_rb_stb=1.205,
            avg_rs_scf_stb=245.0,
            avg_mu_o_cp=2.05,
            pb_psia=3180.0,
        ),
        dca=DCASummary(
            best_model="hyperbolic",
            models=[
                DCAModelSummary(model="exponential", qi_stb_d=600.0, di_nominal_d=0.001,
                                b=0.0, eur_stb=219000.0, r2=0.96, rmse_stb_d=14.0,
                                forecast_days=365, n_points=30),
                DCAModelSummary(model="hyperbolic", qi_stb_d=600.0, di_nominal_d=0.001,
                                b=0.5, eur_stb=280000.0, r2=0.99, rmse_stb_d=7.0,
                                forecast_days=365, n_points=30),
                DCAModelSummary(model="harmonic", qi_stb_d=600.0, di_nominal_d=0.001,
                                b=1.0, eur_stb=340000.0, r2=0.93, rmse_stb_d=19.0,
                                forecast_days=365, n_points=30),
            ],
            eur_exponential_stb=219000.0,
            eur_hyperbolic_stb=280000.0,
            eur_harmonic_stb=340000.0,
        ),
        rta=RTASummary(
            method="fetkovich",
            kh_md_ft=130.0,
            k_md=2.6,
            n_vol_stb=4_800_000.0,
            re_ft=1300.0,
            area_acres=121.8,
            x_multiplier=0.88,
            y_multiplier=1.12,
            status="demo",
            qc_warnings=["Drawdown inestable"],
        ),
    )


@pytest.fixture()
def minimal_summary() -> WellResultsSummary:
    """Summary mínimo — solo M1."""
    return WellResultsSummary(
        well_id="W002",
        well_info=WellInfoSummary(well_id="W002", history_points=0),
    )


# ---------------------------------------------------------------------------
# Tests: export_csv_bytes
# ---------------------------------------------------------------------------

class TestExportCSV:
    def test_returns_bytes(self, full_summary: WellResultsSummary):
        data = export_csv_bytes(full_summary)
        assert isinstance(data, bytes)

    def test_contains_well_id(self, full_summary: WellResultsSummary):
        data = export_csv_bytes(full_summary)
        assert b"W001" in data

    def test_contains_all_sections(self, full_summary: WellResultsSummary):
        data = export_csv_bytes(full_summary)
        text = data.decode("utf-8")
        assert "M1_bien_info" in text
        assert "M2_PVT" in text
        assert "M3_DCA" in text
        assert "M4_RTA" in text
        assert "meta" in text

    def test_parseable_as_dataframe(self, full_summary: WellResultsSummary):
        data = export_csv_bytes(full_summary)
        df = pd.read_csv(io.BytesIO(data))
        assert "seccion" in df.columns
        assert "campo" in df.columns
        assert "valor" in df.columns

    def test_contains_eur_values(self, full_summary: WellResultsSummary):
        data = export_csv_bytes(full_summary)
        text = data.decode("utf-8")
        assert "eur_stb" in text
        assert "hyperbolic" in text

    def test_minimal_summary_no_crash(self, minimal_summary: WellResultsSummary):
        data = export_csv_bytes(minimal_summary)
        assert isinstance(data, bytes)
        assert b"W002" in data


# ---------------------------------------------------------------------------
# Tests: export_json_bytes
# ---------------------------------------------------------------------------

class TestExportJSON:
    def test_returns_bytes(self, full_summary: WellResultsSummary):
        data = export_json_bytes(full_summary)
        assert isinstance(data, bytes)

    def test_valid_json(self, full_summary: WellResultsSummary):
        data = export_json_bytes(full_summary)
        parsed = json.loads(data)
        assert isinstance(parsed, dict)

    def test_contains_well_id(self, full_summary: WellResultsSummary):
        data = export_json_bytes(full_summary)
        parsed = json.loads(data)
        assert parsed["well_id"] == "W001"

    def test_contains_all_modules(self, full_summary: WellResultsSummary):
        data = export_json_bytes(full_summary)
        parsed = json.loads(data)
        assert "well_info" in parsed
        assert "pvt" in parsed
        assert "dca" in parsed
        assert "rta" in parsed

    def test_contains_export_meta(self, full_summary: WellResultsSummary):
        data = export_json_bytes(full_summary)
        parsed = json.loads(data)
        assert "_export_meta" in parsed
        assert "generated_at" in parsed["_export_meta"]

    def test_dca_eur_values_correct(self, full_summary: WellResultsSummary):
        data = export_json_bytes(full_summary)
        parsed = json.loads(data)
        hyp_model = next(m for m in parsed["dca"]["models"] if m["model"] == "hyperbolic")
        assert hyp_model["eur_stb"] == pytest.approx(280000.0)

    def test_minimal_summary_no_crash(self, minimal_summary: WellResultsSummary):
        data = export_json_bytes(minimal_summary)
        parsed = json.loads(data)
        assert parsed["well_id"] == "W002"
        assert parsed["pvt"] is None
        assert parsed["dca"] is None


# ---------------------------------------------------------------------------
# Tests: export_excel_bytes
# ---------------------------------------------------------------------------

class TestExportExcel:
    # Nota: pd.read_excel() requiere openpyxl (no instalado; usamos xlsxwriter solo
    # para escritura). Los tests verifican estructura por firma ZIP y tamaño mínimo.
    _XLSX_MAGIC = b"PK\x03\x04"  # XLSX es un ZIP — empieza con esta firma

    def test_returns_bytes(self, full_summary: WellResultsSummary):
        data = export_excel_bytes(full_summary)
        assert isinstance(data, bytes)

    def test_valid_xlsx_signature(self, full_summary: WellResultsSummary):
        """Un XLSX válido es un ZIP — los primeros 4 bytes son PK\\x03\\x04."""
        data = export_excel_bytes(full_summary)
        assert data[:4] == self._XLSX_MAGIC

    def test_size_increases_with_content(self, full_summary: WellResultsSummary, minimal_summary: WellResultsSummary):
        """El summary completo produce un archivo más grande que el mínimo."""
        full_data = export_excel_bytes(full_summary)
        min_data = export_excel_bytes(minimal_summary)
        assert len(full_data) > len(min_data)

    def test_full_summary_minimum_size(self, full_summary: WellResultsSummary):
        """Esperamos al menos 5 KB de contenido para un reporte completo."""
        data = export_excel_bytes(full_summary)
        assert len(data) > 5_000

    def test_minimal_summary_no_crash(self, minimal_summary: WellResultsSummary):
        data = export_excel_bytes(minimal_summary)
        assert isinstance(data, bytes)
        assert data[:4] == self._XLSX_MAGIC


# ---------------------------------------------------------------------------
# Tests: export_pdf_bytes
# ---------------------------------------------------------------------------

class TestExportPDF:
    def test_returns_bytes(self, full_summary: WellResultsSummary):
        data = export_pdf_bytes(full_summary)
        assert isinstance(data, bytes)

    def test_pdf_header_signature(self, full_summary: WellResultsSummary):
        """Los PDFs comienzan con la firma %PDF."""
        data = export_pdf_bytes(full_summary)
        assert data[:4] == b"%PDF"

    def test_minimal_summary_no_crash(self, minimal_summary: WellResultsSummary):
        data = export_pdf_bytes(minimal_summary)
        assert data[:4] == b"%PDF"

    def test_no_dca_no_crash(self):
        summary = WellResultsSummary(
            well_id="W003",
            well_info=WellInfoSummary(well_id="W003", history_points=5),
            rta=RTASummary(method="fetkovich", n_vol_stb=3_000_000.0),
        )
        data = export_pdf_bytes(summary)
        assert data[:4] == b"%PDF"


# ---------------------------------------------------------------------------
# Tests: save_all_exports
# ---------------------------------------------------------------------------

class TestSaveAllExports:
    def test_saves_four_files(self, full_summary: WellResultsSummary, tmp_path: Path):
        paths = save_all_exports(full_summary, tmp_path)
        assert "csv" in paths
        assert "json" in paths
        assert "xlsx" in paths
        assert "pdf" in paths

    def test_files_exist_on_disk(self, full_summary: WellResultsSummary, tmp_path: Path):
        paths = save_all_exports(full_summary, tmp_path)
        for fmt, path in paths.items():
            assert path.exists(), f"Falta archivo {fmt}: {path}"

    def test_filenames_use_well_id(self, full_summary: WellResultsSummary, tmp_path: Path):
        paths = save_all_exports(full_summary, tmp_path)
        for path in paths.values():
            assert "W001" in path.name

    def test_creates_output_dir_if_missing(self, full_summary: WellResultsSummary, tmp_path: Path):
        nested = tmp_path / "new_subdir" / "output"
        paths = save_all_exports(full_summary, nested)
        assert nested.exists()
        assert len(paths) == 4
