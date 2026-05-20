"""Tests for rta_export_service — build, save JSON summary and PNG."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rta.models import RTAConfig
from src.services.rta_export_service import (
    build_match_summary,
    save_match_summary,
    save_overlay_png,
)
from src.services.rta_match_params_service import RTAMatchParams


@pytest.fixture
def config() -> RTAConfig:
    return RTAConfig(
        well_id="W-TEST",
        pi_psia=3200.0,
        phi_frac=0.18,
        h_ft=50.0,
        re_ft=1320.0,
    )


@pytest.fixture
def match_params() -> RTAMatchParams:
    return RTAMatchParams(
        well_id="W-TEST",
        method="fetkovich",
        re_ft=1320.0,
        area_acres=125.7,
        ln_re_rw_term=7.1,
        kh_md_ft=45.2,
        k_md=0.904,
        n_vol_stb=1_234_567,
        effective_x_multiplier=0.5,
        effective_y_multiplier=2.3,
        warnings=[],
    )


def test_build_match_summary_structure(config, match_params):
    summary = build_match_summary(
        match_params=match_params,
        ref_curve_id="fetkovich_re_rw_1000",
        config=config,
    )
    assert summary["well_id"] == "W-TEST"
    assert summary["method"] == "fetkovich"
    assert summary["ref_curve_id"] == "fetkovich_re_rw_1000"
    assert summary["status"] == "demo"
    assert "exported_at" in summary
    assert "config" in summary
    assert summary["match"]["x_multiplier"] == pytest.approx(0.5)
    assert summary["results"]["kh_md_ft"] == pytest.approx(45.2)
    assert summary["results"]["n_vol_mm_stb"] == pytest.approx(1.2346, abs=0.001)


def test_build_match_summary_none_results(config):
    mp = RTAMatchParams(
        well_id="W-TEST",
        method="fetkovich",
        re_ft=None,
        area_acres=None,
        ln_re_rw_term=None,
        kh_md_ft=None,
        k_md=None,
        n_vol_stb=None,
        effective_x_multiplier=1.0,
        effective_y_multiplier=1.0,
        warnings=["re no definido"],
    )
    summary = build_match_summary(
        match_params=mp, ref_curve_id="fetkovich_re_rw_100", config=config
    )
    assert summary["results"]["kh_md_ft"] is None
    assert summary["results"]["n_vol_mm_stb"] is None
    assert summary["warnings"] == ["re no definido"]


def test_save_match_summary_roundtrip(tmp_path, config, match_params):
    summary = build_match_summary(
        match_params=match_params,
        ref_curve_id="fetkovich_re_rw_1000",
        config=config,
    )
    path = save_match_summary(summary, output_dir=tmp_path)
    assert path.exists()
    assert path.name == "W-TEST_rta_match_summary.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["well_id"] == "W-TEST"
    assert loaded["results"]["kh_md_ft"] == pytest.approx(45.2)


def test_save_match_summary_creates_output_dir(tmp_path, config, match_params):
    nested = tmp_path / "deep" / "output"
    summary = build_match_summary(
        match_params=match_params,
        ref_curve_id="fetkovich_re_rw_1000",
        config=config,
    )
    path = save_match_summary(summary, output_dir=nested)
    assert path.exists()


def test_save_overlay_png(tmp_path):
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    path = save_overlay_png(fake_png, well_id="W-TEST", output_dir=tmp_path)
    assert path.exists()
    assert path.name == "W-TEST_rta_overlay.png"
    assert path.read_bytes() == fake_png
