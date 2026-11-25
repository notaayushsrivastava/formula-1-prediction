"""Phase 7: the prediction dataset, one row per driver per race per MOMENT.

Moments: 'pre' (before the weekend), 'practice' (after practice) and 'quali'
(after qualifying, final grid). A column is only filled at a moment when that
information really exists then; every other feature is NaN at that moment.

OBSERVATIONS (Phase 9) are raw measurements of a race (its quali gap, its practice
long-run gap). Like LABELS they describe the race itself: models may learn from them
in PAST races, and the backtest removes them before predict().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..cache import DEFAULT_CACHE_DIR
from ..errors import DataNotAvailableError
from ..sources.base import ForecastSource, HistorySource, WeekendSource
from ._optional import require
from .circuits import circuit_features, circuit_sc_rates, circuit_stats
from .features import add_history_features, add_quali_features
from .history import History, load_history
from .weekend import (
    DRIVER_LEVEL, FORECAST_COLUMNS, OBSERVED_COLUMNS, OPENF1_FIRST_YEAR, RACE_LEVEL,
    collect_forecasts, collect_observed, collect_weekends, empty_frame,
)

MOMENTS = ("pre", "practice", "quali")
LEAD_DAYS = {"pre": 3, "practice": 2, "quali": 1}  # forecast issued this many days before
PRE_FEATURES = [
    "driver_avg_finish_5", "driver_points_5", "driver_dnf_rate_10", "champ_points_before",
    "experience", "age", "driver_avg_finish_circuit",
    "team_avg_points_5", "team_mech_dnf_rate_10", "team_pit_delta_10",
    "circuit_street", "circuit_high_speed", "circuit_high_downforce",
    "circuit_places_moved", "circuit_sc_rate", "has_sc_history",
    "air_temp_forecast", "rain_forecast_mm", "has_forecast",
    "era", "races_since_era_start", "sprint_weekend",
]
PRACTICE_FEATURES = ["has_openf1", "practice_pace_gap_pct", "has_practice_pace",
                     "wet_practice", "track_temp", "has_track_temp"]
QUALI_FEATURES = ["grid", "pit_lane_start", "quali_gap_pct", "teammate_quali_gap_pct",
                  "wet_quali"]
FEATURES = {
    "pre": PRE_FEATURES,
    "practice": PRE_FEATURES + PRACTICE_FEATURES,
    "quali": PRE_FEATURES + PRACTICE_FEATURES + QUALI_FEATURES,
}
ALL_FEATURES = FEATURES["quali"]
ID_COLUMNS = ["RaceKey", "RaceId", "Year", "Round", "Date", "EventName", "CircuitId",
              "Moment", "DriverId", "Driver", "TeamId", "Number"]
LABELS = ["Finish", "FinishOrder", "Win", "Podium", "PointsFinish", "DNF", "MechDNF", "Points"]
OBSERVATIONS = ["QualiGapObs", "PracticeGapObs"]  # this race's raw gaps (%), NaN if none
DIAGNOSTICS = ["CircuitCluster", "RaceWet", "Neutralised", "RaceRainMm"]  # never features
DEFAULT_PATH = DEFAULT_CACHE_DIR / "predict" / "dataset.parquet"
PRACTICE_GAP_CAP = 5.0  # % behind the fastest long run; more is noise (installation laps)

ROOKIE_FINISH = 15.0  # no history yet: assume a back-of-the-midfield finish
FILL: dict[str, float] = {
    "driver_points_5": 0.0,
    "driver_dnf_rate_10": 0.12,       # roughly the recent field-wide DNF rate (assumption)
    "team_avg_points_5": 0.0,
    "team_mech_dnf_rate_10": 0.06,    # assumption, as above
    "team_pit_delta_10": 0.0,         # "an average pit crew"
    "age": 25.0,
    "circuit_places_moved": 2.0,      # only the very first race in the data lacks history
    "circuit_sc_rate": 0.0,           # with has_sc_history = 0
    "has_sc_history": 0.0,
    "air_temp_forecast": 0.0,         # with has_forecast = 0
    "rain_forecast_mm": 0.0,
    "track_temp": 0.0,                # with has_track_temp = 0
    "wet_practice": 0.0,
    "wet_quali": 0.0,
    "quali_gap_pct": 0.0,
    "teammate_quali_gap_pct": 0.0,
}


def feature_columns(moment: str) -> list[str]:
    """The feature columns a model may use at a moment ('pre', 'practice' or 'quali')."""
    if moment not in FEATURES:
        raise ValueError(f"Unknown moment {moment!r}. Use one of: {', '.join(MOMENTS)}.")
    return list(FEATURES[moment])


def build_dataset(history: HistorySource, weekend: WeekendSource | None = None,
                  weather: ForecastSource | None = None, first_year: int = 2018,
                  last_year: int | None = None, history_from: int = 2000) -> pd.DataFrame:
    """Build the dataset: rows for races from first_year to last_year, 3 moments each.

    history_from: how far back to read history (form, experience, circuit stats).
    weekend/weather may be None: their features become 0 with an availability flag 0.
    """
    hist = load_history(history, history_from)
    last = int(hist.races["Year"].max()) if last_year is None else last_year
    raced = _raced(hist, last)
    base = _base_rows(hist, first_year, last)
    if weekend is None:
        race_level, driver_level = empty_frame(RACE_LEVEL), empty_frame(DRIVER_LEVEL)
    else:
        race_level, driver_level = collect_weekends(
            raced[raced["Year"] >= OPENF1_FIRST_YEAR], weekend)
    base = _add_weekend(base, raced, race_level, driver_level)
    base["PracticeGapObs"] = base["practice_pace_gap_pct"]  # raw, before any filling
    races = base.drop_duplicates("RaceId")[["RaceId", "Date", "Time", "Lat", "Long"]]
    if weather is None:
        forecasts, observed = empty_frame(FORECAST_COLUMNS), empty_frame(OBSERVED_COLUMNS)
    else:
        forecasts, observed = collect_forecasts(races, weather), collect_observed(races, weather)
    frames = [_moment_frame(base, moment, forecasts) for moment in MOMENTS]
    data = pd.concat(frames, ignore_index=True).merge(observed, on="RaceId", how="left")
    return data[ID_COLUMNS + ALL_FEATURES + LABELS + OBSERVATIONS + DIAGNOSTICS]


def save_dataset(data: pd.DataFrame, path: Path | str | None = None) -> Path:
    """Write the dataset as Parquet (needs pyarrow); returns the path."""
    require("pyarrow", "Saving the dataset as Parquet")
    target = Path(path) if path else DEFAULT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(target, index=False)
    return target


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Read a dataset written by save_dataset."""
    require("pyarrow", "Reading the Parquet dataset")
    return pd.read_parquet(Path(path) if path else DEFAULT_PATH)


# ---------- steps ----------
def _raced(hist: History, last_year: int) -> pd.DataFrame:
    """Races that already have results, up to last_year."""
    done = hist.races["RaceId"].isin(hist.results["RaceId"])
    return hist.races[done & (hist.races["Year"] <= last_year)]


def _base_rows(hist: History, first_year: int, last_year: int) -> pd.DataFrame:
    """History features (computed on ALL history), then cut to the wanted years."""
    rows = add_history_features(hist)
    rows = rows[(rows["Year"] >= first_year) & (rows["Year"] <= last_year)]
    if rows.empty:
        raise DataNotAvailableError(f"No race results between {first_year} and {last_year}.")
    rows = rows.join(circuit_features(rows, circuit_stats(hist)))
    return add_quali_features(rows)


def _add_weekend(base: pd.DataFrame, raced: pd.DataFrame, race_level: pd.DataFrame,
                 driver_level: pd.DataFrame) -> pd.DataFrame:
    """Merge OpenF1 facts and each circuit's past safety-car rate onto the rows."""
    out = base.merge(race_level, on="RaceId", how="left")
    out = out.merge(driver_level, on=["RaceId", "Number"], how="left")
    known = raced[["RaceId", "Date", "CircuitId"]].merge(
        race_level[["RaceId", "Neutralised"]], on="RaceId", how="left")
    return out.merge(circuit_sc_rates(known), on="RaceId", how="left")


def _moment_frame(base: pd.DataFrame, moment: str, forecasts: pd.DataFrame) -> pd.DataFrame:
    """The rows as seen at one moment: right forecast lead, later information hidden."""
    ahead = forecasts.loc[forecasts["Lead"] == LEAD_DAYS[moment],
                          ["RaceId", "air_temp_forecast", "rain_forecast_mm"]]
    df = base.merge(ahead, on="RaceId", how="left")
    df["Moment"] = moment
    df["has_forecast"] = df["air_temp_forecast"].notna().astype(int)
    practice_temp = df["track_temp_practice"]
    df["track_temp"] = (df["track_temp_quali"].fillna(practice_temp)
                        if moment == "quali" else practice_temp)
    df["has_track_temp"] = df["track_temp"].notna().astype(int)
    df = _fill(df)
    hidden = [c for c in ALL_FEATURES if c not in FEATURES[moment]]
    df[hidden] = np.nan
    return df


def _fill(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing values with documented defaults (flags say what was missing)."""
    out = df.copy()
    out["has_openf1"] = out["has_openf1"].fillna(0).astype(int)
    out["has_practice_pace"] = out["has_practice_pace"].fillna(0).astype(int)
    field_median = out.groupby("RaceId")["practice_pace_gap_pct"].transform("median")
    out["practice_pace_gap_pct"] = (out["practice_pace_gap_pct"].fillna(field_median)
                                    .fillna(0.0).clip(0.0, PRACTICE_GAP_CAP))
    out["driver_avg_finish_5"] = out["driver_avg_finish_5"].fillna(ROOKIE_FINISH)
    out["driver_avg_finish_circuit"] = out["driver_avg_finish_circuit"].fillna(
        out["driver_avg_finish_5"])
    for column, value in FILL.items():
        out[column] = out[column].fillna(value)
    return out
