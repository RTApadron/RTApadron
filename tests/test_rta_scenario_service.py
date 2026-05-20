"""Tests for src/services/rta_scenario_service.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rta.models import RTAConfig
from src.services.rta_scenario_service import (
    load_rta_scenario,
    save_rta_scenario,
    scenario_path,
)

WELL_ID = "W-TEST"


def _default_config() -> RTAConfig:
    return RTAConfig(well_id=WELL_ID)


def _full_config() -> RTAConfig:
    return RTAConfig(
        well_id=WELL_ID,
        pi_psia=4200.0,
        ct_1psi=1.5e-5,
        phi_frac=0.22,
        h_ft=60.0,
        rw_ft=0.354,
        re_ft=1500.0,
        area_acres=160.0,
        Bo_rb_stb=1.35,
        mu_o_cp=3.5,
        CA=30.88,
        swi_frac=0.25,
        notes="Test scenario con todos los campos.",
    )


# ---------------------------------------------------------------------------
# save_rta_scenario
# ---------------------------------------------------------------------------

def test_save_creates_json_file(tmp_path: Path) -> None:
    config = _default_config()
    path = save_rta_scenario(config, output_dir=tmp_path)

    assert path.exists()
    assert path.suffix == ".json"
    assert WELL_ID in path.name


def test_save_creates_output_dir_if_missing(tmp_path: Path) -> None:
    target_dir = tmp_path / "deep" / "output"
    assert not target_dir.exists()
    save_rta_scenario(_default_config(), output_dir=target_dir)
    assert target_dir.exists()


def test_save_writes_valid_json(tmp_path: Path) -> None:
    save_rta_scenario(_default_config(), output_dir=tmp_path)
    path = tmp_path / f"{WELL_ID}_rta_scenario.json"
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["well_id"] == WELL_ID


def test_save_overwrites_existing_file(tmp_path: Path) -> None:
    save_rta_scenario(_default_config(), output_dir=tmp_path)
    config_v2 = RTAConfig(well_id=WELL_ID, pi_psia=5000.0)
    save_rta_scenario(config_v2, output_dir=tmp_path)

    loaded = load_rta_scenario(WELL_ID, output_dir=tmp_path)
    assert loaded is not None
    assert loaded.pi_psia == 5000.0


# ---------------------------------------------------------------------------
# load_rta_scenario
# ---------------------------------------------------------------------------

def test_load_returns_none_when_file_missing(tmp_path: Path) -> None:
    result = load_rta_scenario("NONEXISTENT", output_dir=tmp_path)
    assert result is None


def test_load_returns_config_after_save(tmp_path: Path) -> None:
    config = _default_config()
    save_rta_scenario(config, output_dir=tmp_path)
    loaded = load_rta_scenario(WELL_ID, output_dir=tmp_path)
    assert loaded is not None
    assert loaded.well_id == WELL_ID


def test_load_roundtrip_all_fields(tmp_path: Path) -> None:
    config = _full_config()
    save_rta_scenario(config, output_dir=tmp_path)
    loaded = load_rta_scenario(WELL_ID, output_dir=tmp_path)

    assert loaded is not None
    assert loaded.pi_psia == config.pi_psia
    assert loaded.phi_frac == config.phi_frac
    assert loaded.h_ft == config.h_ft
    assert loaded.rw_ft == config.rw_ft
    assert loaded.re_ft == config.re_ft
    assert loaded.area_acres == config.area_acres
    assert loaded.Bo_rb_stb == config.Bo_rb_stb
    assert loaded.mu_o_cp == config.mu_o_cp
    assert loaded.CA == config.CA
    assert loaded.swi_frac == config.swi_frac
    assert loaded.notes == config.notes


def test_load_returns_none_on_corrupted_json(tmp_path: Path) -> None:
    path = tmp_path / f"{WELL_ID}_rta_scenario.json"
    path.write_text("{ this is not valid json }", encoding="utf-8")

    result = load_rta_scenario(WELL_ID, output_dir=tmp_path)
    assert result is None


def test_load_returns_none_on_invalid_model(tmp_path: Path) -> None:
    path = tmp_path / f"{WELL_ID}_rta_scenario.json"
    path.write_text(json.dumps({"well_id": ""}), encoding="utf-8")

    result = load_rta_scenario(WELL_ID, output_dir=tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# scenario_path
# ---------------------------------------------------------------------------

def test_scenario_path_returns_expected_location(tmp_path: Path) -> None:
    path = scenario_path(WELL_ID, output_dir=tmp_path)
    assert path == tmp_path / f"{WELL_ID}_rta_scenario.json"
    assert not path.exists()
