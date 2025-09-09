from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import date

class Well(BaseModel):
    well_id: str
    well_name: str
    field: Optional[str] = None
    api: Optional[float] = Field(None, ge=5, le=60)
    lat: Optional[float] = None
    lon: Optional[float] = None
    datum_elev_ft: Optional[float] = 0.0

class SurveyPoint(BaseModel):
    well_id: str
    md_ft: float = Field(ge=0)
    inc_deg: float = Field(ge=0, le=180)
    azi_deg: float = Field(ge=0, le=360)

class MechState(BaseModel):
    well_id: str
    tubing_id_in: float = Field(gt=0)
    tubing_set_depth_ft: float = Field(gt=0)
    casing_id_in: Optional[float] = None
    perfs_top_ft: float = Field(gt=0)
    perfs_bottom_ft: float = Field(gt=0)

    @validator("perfs_bottom_ft")
    def check_perf_order(cls, v, values):
        if "perfs_top_ft" in values and v < values["perfs_top_ft"]:
            raise ValueError("perfs_bottom_ft < perfs_top_ft")
        return v

class Lift(BaseModel):
    well_id: str
    method: Optional[str] = None
    whp_psia: Optional[float] = Field(None, ge=0)  # wellhead pressure
    wh_temp_F: Optional[float] = None

class HistoryRow(BaseModel):
    well_id: str
    date: date
    qo_stb_d: float = Field(ge=0)
    qw_stb_d: float = Field(ge=0)
    qg_mscf_d: float = Field(ge=0)
    pwf_psia: Optional[float] = None
    temp_F: Optional[float] = None
    api: Optional[float] = Field(None, ge=5, le=60)
