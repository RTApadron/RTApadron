import pandas as pd
from typing import Tuple, List, Type, TypeVar
from pydantic import ValidationError
from .models import Well, SurveyPoint, MechState, Lift, HistoryRow

T = TypeVar("T")

def _parse_df(df: pd.DataFrame, model: Type[T]) -> Tuple[List[T], List[str]]:
    ok, errs = [], []
    for i, row in df.iterrows():
        try:
            ok.append(model(**row.to_dict()))
        except ValidationError as e:
            errs.append(f"row {i}: {e}")
    return ok, errs

def load_wells(path: str):
    df = pd.read_csv(path)
    return _parse_df(df, Well)

def load_survey(path: str):
    df = pd.read_csv(path)
    return _parse_df(df, SurveyPoint)

def load_mech(path: str):
    df = pd.read_csv(path)
    return _parse_df(df, MechState)

def load_lift(path: str):
    df = pd.read_csv(path)
    return _parse_df(df, Lift)

def load_history(path: str):
    df = pd.read_csv(path, parse_dates=["date"])
    return _parse_df(df, HistoryRow)
