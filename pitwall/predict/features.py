"""Driver, team, era and qualifying features. Each uses ONLY what was known before the race.

Two tools keep the future out:
- shift(1): in one driver's rows (oldest first), a row only sees the rows above it.
- as-of join: for a race on date D, take the latest value dated strictly before D.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .history import Q_COLUMNS, History

ERA_STARTS = (2014, 2017, 2022, 2026)  # hybrid engines, wider cars, ground effect, 2026 rules
Keys = str | list[str]


def running(df: pd.DataFrame, by: Keys, column: str, window: int, stat: str = "mean",
            past_only: bool = False) -> pd.Series:
    """Rolling `stat` ('mean', 'sum' or 'median') over each group's last `window` rows.

    Rows must be oldest first. past_only=True shifts by one first, so a row never sees
    its own value (the race we want to predict).
    """
    def one(values: pd.Series) -> pd.Series:
        if past_only:
            values = values.shift(1)
        return getattr(values.rolling(window, min_periods=1), stat)()

    return df.groupby(by, sort=False)[column].transform(one)


def asof(target: pd.DataFrame, table: pd.DataFrame, column: str,
         by: Keys | None = None) -> pd.Series:
    """For each target row, the latest table[column] dated strictly BEFORE its Date.

    This 'as-of join' answers "what did we know when this race started?". With `by`,
    only rows with the same key(s) count, e.g. the same team.
    """
    if by is None:
        keys: list[str] = []
    else:
        keys = [by] if isinstance(by, str) else list(by)
    if table.empty:
        return pd.Series(np.nan, index=target.index)
    left = target[[*keys, "Date"]].reset_index(names="_row").sort_values("Date")
    right = table[[*keys, "Date", column]].dropna(subset=[column]).sort_values("Date")
    merged = pd.merge_asof(left, right, on="Date", by=keys or None,
                           allow_exact_matches=False)
    return merged.set_index("_row")[column].reindex(target.index)


def add_history_features(hist: History) -> pd.DataFrame:
    """Driver form, team form, championship, experience, age and era for every result row."""
    df = hist.results.copy()
    df["driver_avg_finish_5"] = running(df, "DriverId", "FinishForAverage", 5, past_only=True)
    df["driver_points_5"] = running(df, "DriverId", "Points", 5, "sum", past_only=True)
    df["driver_dnf_rate_10"] = running(df, "DriverId", "DNF", 10, past_only=True)
    df["driver_avg_finish_circuit"] = running(
        df, ["DriverId", "CircuitId"], "FinishForAverage", 1000, past_only=True)
    df["experience"] = df.groupby("DriverId").cumcount()  # earlier starts in the data
    df["age"] = df["Age"]
    df["champ_points_before"] = champ_points_before(df, hist)
    teams = team_race_table(df)
    df["team_avg_points_5"] = asof(df, teams, "TeamPoints5", by="TeamId")
    df["team_mech_dnf_rate_10"] = asof(df, teams, "MechRate10", by="TeamId")
    df["team_pit_delta_10"] = asof(df, pit_table(hist), "PitDelta10", by="TeamId")
    df = df.merge(era_table(hist.races), on="RaceId", how="left")
    df["sprint_weekend"] = df["HasSprint"].astype(int)
    return df


def champ_points_before(df: pd.DataFrame, hist: History) -> pd.Series:
    """Championship points before the race (0 at a season opener; sprints included)."""
    table = hist.standings.merge(hist.races[["RaceId", "Date", "Year"]], on="RaceId")
    return asof(df, table, "Points", by=["DriverId", "Year"]).fillna(0.0)


def team_race_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per race, with running form INCLUDING that race (used via asof)."""
    teams = (df.groupby(["TeamId", "RaceId"], sort=False)
             .agg(Date=("Date", "first"), TeamPoints=("Points", "sum"),
                  MechRate=("MechDNF", "mean"))
             .reset_index().sort_values("Date"))
    teams["TeamPoints5"] = running(teams, "TeamId", "TeamPoints", 5)
    teams["MechRate10"] = running(teams, "TeamId", "MechRate", 10)
    return teams


def pit_table(hist: History, max_delta: float = 10.0) -> pd.DataFrame:
    """Team pit-crew speed: median of (stop time - that race's median stop time).

    Subtracting the race median removes the pit lane's length, which depends on the
    circuit, not the crew. Stops more than max_delta s off (repairs, penalties) are ignored.
    """
    stops = hist.pitstops.dropna(subset=["DurationSeconds"])
    if stops.empty:
        return pd.DataFrame(columns=["TeamId", "Date", "PitDelta10"])
    stops = stops.merge(hist.races[["RaceId", "Date"]], on="RaceId")
    seconds = stops["DurationSeconds"].astype(float)
    stops = stops.assign(Delta=seconds - seconds.groupby(stops["RaceId"]).transform("median"))
    stops = stops[stops["Delta"].abs() <= max_delta]
    teams = (stops.groupby(["TeamId", "RaceId"])
             .agg(Date=("Date", "first"), Delta=("Delta", "median"))
             .reset_index().sort_values("Date"))
    teams["PitDelta10"] = running(teams, "TeamId", "Delta", 10, "median")
    return teams


def era_table(races: pd.DataFrame) -> pd.DataFrame:
    """Regulation era (0 = before 2014, ..., 4 = 2026 rules) and races run in it so far."""
    out = races.sort_values("Date")[["RaceId", "Year"]].copy()
    out["era"] = np.searchsorted(ERA_STARTS, out["Year"].to_numpy(), side="right")
    out["races_since_era_start"] = out.groupby("era").cumcount()
    return out[["RaceId", "era", "races_since_era_start"]]


QUALI_GAP_CAP = 5.0      # % behind; beyond this a gap is noise (wet lap, no lap, red flag)
TEAMMATE_GAP_CAP = 3.0   # % either way
ROBUST_QUALI = True      # False = the Phase 7 formula (kept for the before/after diagnostic)


def add_quali_features(df: pd.DataFrame) -> pd.DataFrame:
    """Grid and qualifying pace (known after qualifying, so 'quali' moment only).

    The gap is measured inside one session: each driver against the fastest lap of the
    deepest session he set a time in (Q3, else Q2, else Q1), so a dry Q1 lap is never
    compared with a wet Q3 lap. It is then made monotone in qualifying order (nobody
    looks faster than a driver who out-qualified him) and capped at QUALI_GAP_CAP.

    Also stores QualiGapObs: the raw gap (NaN without a time), an OBSERVATION that
    models may learn from in PAST races at every moment, never in the race predicted.
    """
    out = df.copy()
    gap = session_gaps(out) if ROBUST_QUALI else _phase7_gaps(out)
    out["QualiGapObs"] = gap
    filled = gap.fillna(gap.groupby(out["RaceId"]).transform("max"))
    teammate = teammate_gaps(out)
    if ROBUST_QUALI:
        filled = filled.clip(0.0, QUALI_GAP_CAP)
        teammate = teammate.clip(-TEAMMATE_GAP_CAP, TEAMMATE_GAP_CAP)
    out["quali_gap_pct"] = filled
    out["teammate_quali_gap_pct"] = teammate
    out["grid"] = out["Grid"]
    out["pit_lane_start"] = out["PitLaneStart"]
    return out


def session_gaps(df: pd.DataFrame) -> pd.Series:
    """Gap (%) to the fastest lap of the deepest session each driver set a time in,
    made monotone in qualifying order. NaN for a driver with no time at all."""
    gap = pd.Series(np.nan, index=df.index)
    for column in Q_COLUMNS:  # Q1, Q2, Q3: a deeper session overrides an earlier one
        times = pd.to_numeric(df[column], errors="coerce").where(lambda t: t > 0)
        fastest = times.groupby(df["RaceId"]).transform("min")
        session = (times / fastest - 1) * 100
        gap = session.where(session.notna(), gap)
    if "QualiPosition" in df:
        order = pd.to_numeric(df["QualiPosition"], errors="coerce")
        ranked = pd.DataFrame({"RaceId": df["RaceId"], "order": order, "gap": gap})
        ranked = ranked.sort_values(["RaceId", "order"], na_position="last")
        known = ranked["gap"].notna() & ranked["order"].notna()
        monotone = ranked.loc[known].groupby("RaceId")["gap"].cummax()
        gap.loc[monotone.index] = monotone
    return gap


def _phase7_gaps(df: pd.DataFrame) -> pd.Series:
    """The original formula: best lap of ANY session vs the best lap of the day."""
    best = df[Q_COLUMNS].min(axis=1)
    pole = best.groupby(df["RaceId"]).transform("min")
    return (best / pole - 1) * 100


def teammate_gaps(df: pd.DataFrame) -> pd.Series:
    """Quali gap (%) to the teammate, in the deepest part both set a time in (Q3, Q2, Q1).

    Comparing in the same part is fair: the track gets faster through qualifying.
    """
    gaps = pd.Series(np.nan, index=df.index)
    for _, pair in df.groupby(["RaceId", "TeamId"]):
        if len(pair) != 2:
            continue
        for column in reversed(Q_COLUMNS):
            mine, theirs = pair[column].to_numpy(float)
            if not (np.isnan(mine) or np.isnan(theirs)):
                gaps.loc[pair.index[0]] = (mine / theirs - 1) * 100
                gaps.loc[pair.index[1]] = (theirs / mine - 1) * 100
                break
    return gaps
