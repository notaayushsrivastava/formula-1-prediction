"""Race history as one tidy table: one row per car that started a race, plus labels."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..errors import DataNotAvailableError
from ..sources.base import HistorySource, Record

Q_COLUMNS = ["Q1Seconds", "Q2Seconds", "Q3Seconds"]
NOT_STARTED = {"DNQ", "DNPQ", "DNP", "EX"}  # never took the start; DNS counts as a DNF
PIT_COLUMNS = ["RaceId", "DriverId", "TeamId", "Stop", "Lap", "DurationSeconds"]
STANDING_COLUMNS = ["RaceId", "DriverId", "Points"]
# Retirement reasons: mechanical unless the text also mentions an incident.
MECHANICAL = (r"\b(?:engine|gearbox|hydraulic|power|electr|brake|suspension|transmission|"
              r"turbo|ers\b|mgu|battery|oil|water|fuel|cool|overheat|clutch|exhaust|"
              r"driveshaft|throttle|steering|wheel|mechanical|technical|vibration|"
              r"pneumatic|radiator|differential|halfshaft)")
INCIDENT = r"\b(?:collision|accident|spun|crash|contact|damage|debris)"


@dataclass(frozen=True)
class History:
    """Everything the feature builders need, as DataFrames."""

    races: pd.DataFrame      # one row per race, oldest first (future races included)
    results: pd.DataFrame    # one row per starter per race, oldest first, with labels
    pitstops: pd.DataFrame   # PIT_COLUMNS
    standings: pd.DataFrame  # championship points after each race


def load_history(source: HistorySource, min_year: int = 2000) -> History:
    """Read races, results, qualifying, drivers, pit stops and standings from a source."""
    races = _races(source.get_races(min_year))
    results = _results(source.get_race_results(min_year), races)
    results = _with_qualifying(results, source.get_qualifying(min_year))
    results = _with_drivers(results, source.get_drivers())
    pitstops = _with_int_ids(pd.DataFrame(source.get_pitstops(min_year), columns=PIT_COLUMNS))
    standings = _with_int_ids(
        pd.DataFrame(source.get_driver_standings(min_year), columns=STANDING_COLUMNS))
    return History(races, results, pitstops, standings)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """What happened in the race: the things the models will learn to predict.

    DNF = not classified (DNS included, disqualification not). FinishForAverage
    treats a DNF as finishing last, for form averages.
    """
    out = df.copy()
    finish = out["Finish"]
    out["Starters"] = out.groupby("RaceId")["DriverId"].transform("size")
    out["DNF"] = (finish.isna() & (out["PositionText"].astype(str) != "DSQ")).astype(int)
    out["Win"] = (finish == 1).astype(int)
    out["Podium"] = (finish <= 3).astype(int)
    out["PointsFinish"] = (finish <= 10).astype(int)
    out["FinishForAverage"] = finish.fillna(out["Starters"].astype(float))
    out["FinishOrder"] = out.groupby("RaceId")["DisplayOrder"].rank(method="first").astype(int)
    reason = out["ReasonRetired"].fillna("").astype(str).str.lower()
    mechanical = reason.str.contains(MECHANICAL, regex=True)
    incident = reason.str.contains(INCIDENT, regex=True)
    out["MechDNF"] = (out["DNF"].astype(bool) & mechanical & ~incident).astype(int)
    return out


def _races(rows: list[Record]) -> pd.DataFrame:
    if not rows:
        raise DataNotAvailableError("The history source returned no races.")
    races = pd.DataFrame(rows)
    races["Date"] = pd.to_datetime(races["Date"])
    races[["RaceId", "Year", "Round"]] = races[["RaceId", "Year", "Round"]].astype(int)
    races["HasSprint"] = races["HasSprint"].eq(True)
    races["RaceKey"] = races["Year"].astype(str) + "-" + races["Round"].astype(str).str.zfill(2)
    return races.sort_values("Date").reset_index(drop=True)


def _results(rows: list[Record], races: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df[~df["PositionText"].astype(str).isin(NOT_STARTED)]
    df = df.astype({"RaceId": "int64"}).merge(races, on="RaceId", how="inner")
    df["Finish"] = pd.to_numeric(df["Position"], errors="coerce").astype(float)
    df["Points"] = pd.to_numeric(df["Points"], errors="coerce").fillna(0.0)
    df["Number"] = pd.to_numeric(df["Number"], errors="coerce").astype("Int64")
    df = _add_grid(add_labels(df))
    return df.sort_values(["Date", "DisplayOrder"]).reset_index(drop=True)


def _add_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Grid after penalties; a pit-lane start goes to the back of the field."""
    out = df.copy()
    grid = pd.to_numeric(out["GridPosition"], errors="coerce").astype(float)
    out["PitLaneStart"] = grid.isna().astype(int)
    out["Grid"] = grid.fillna(out["Starters"].astype(float))
    return out


def _with_qualifying(df: pd.DataFrame, rows: list[Record]) -> pd.DataFrame:
    quali = pd.DataFrame(rows, columns=["RaceId", "DriverId", "QualiPosition", *Q_COLUMNS])
    quali = _with_int_ids(quali).drop_duplicates(["RaceId", "DriverId"])
    out = df.merge(quali, on=["RaceId", "DriverId"], how="left")
    out[Q_COLUMNS] = out[Q_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return out


def _with_drivers(df: pd.DataFrame, rows: list[Record]) -> pd.DataFrame:
    """Add the 3-letter code and the driver's age on race day."""
    info = pd.DataFrame(rows, columns=["DriverId", "Driver", "FullName", "DateOfBirth"])
    info = info.drop_duplicates("DriverId")[["DriverId", "Driver", "DateOfBirth"]]
    out = df.merge(info, on="DriverId", how="left")
    out["Driver"] = out["Driver"].fillna(out["DriverId"].str[:3].str.upper())
    born = pd.to_datetime(out["DateOfBirth"], errors="coerce")
    out["Age"] = (out["Date"] - born).dt.days / 365.25
    return out


def _with_int_ids(df: pd.DataFrame) -> pd.DataFrame:
    """RaceId as int64, so merges line up even when a table is empty."""
    return df.astype({"RaceId": "int64"})
