"""Phase 4: enrich a race Session with F1 Live Timing data.

Adds Compound, TyreLife, FreshTyre and TrackStatus to session.laps, and replaces
the pit-stop-inferred Stint with F1's own stint numbers.
"""
from __future__ import annotations

import logging

import pandas as pd

from .core import Laps, Session
from .errors import DataNotAvailableError
from .sources.livetiming import (
    TRACK_STATUS, LiveTimingClient, parse_lap_starts, parse_stints, parse_track_status,
    track_status_by_lap,
)

log = logging.getLogger(__name__)
PER_LAP_COLUMNS = ["Number", "LapNumber", "Stint", "Compound", "TyreLife", "FreshTyre"]


def load_live_timing(session: Session, client: LiveTimingClient | None = None) -> Session:
    """Add real tyre and track-status data to a loaded race session; returns the session."""
    if session.kind != "R" or session.laps.empty or session.event is None:
        raise ValueError("load_live_timing needs a loaded race: get_session(year, event).load()")
    client = client or LiveTimingClient()
    path = client.session_path(session.year, str(session.event["Date"]), session.kind)
    stints = pd.DataFrame(parse_stints(client.keyframe(path, "TimingAppData")))
    if stints.empty:
        raise DataNotAvailableError(f"No tyre data in Live Timing for {session.name}.")
    changes = parse_track_status(client.stream(path, "TrackStatus"))
    lap_starts = parse_lap_starts(client.stream(path, "LapCount"))

    laps = _attach_tyres(session.laps, session.results, stints)
    by_lap = track_status_by_lap(lap_starts, changes)
    laps["TrackStatus"] = laps["LapNumber"].map(by_lap).fillna("")  # '' = unknown
    session.laps = Laps(laps)
    session.track_status = pd.DataFrame(
        [(t, code, TRACK_STATUS.get(code, "Unknown")) for t, code in changes],
        columns=["SessionSeconds", "Status", "Meaning"])
    return session


def stint_laps(stints: pd.DataFrame) -> pd.DataFrame:
    """One row per lap from per-stint data: Number, LapNumber, Stint, Compound, TyreLife, FreshTyre."""
    rows = []
    for number, drv in stints.sort_values(["Number", "Stint"]).groupby("Number"):
        lap = 0
        for s in drv.itertuples():
            for k in range(1, int(s.StintLaps) + 1):
                lap += 1
                rows.append({"Number": number, "LapNumber": lap, "Stint": s.Stint,
                             "Compound": s.Compound, "TyreLife": s.StartLaps + k,
                             "FreshTyre": s.New})
    return pd.DataFrame(rows, columns=PER_LAP_COLUMNS)


def _attach_tyres(laps: Laps, results: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Merge per-lap tyre data onto the laps, matching car number -> driver code."""
    numbers = results[["Driver", "Number"]].astype({"Number": str})
    per_lap = stint_laps(stints).merge(numbers, on="Number").drop(columns="Number")
    base = pd.DataFrame(laps).rename(columns={"Stint": "InferredStint"})
    merged = base.merge(per_lap, on=["Driver", "LapNumber"], how="left")
    missing = int(merged["Compound"].isna().sum())
    if missing:
        log.warning("%d laps have no Live Timing tyre data; using the pit-stop guess for "
                    "their stint.", missing)
    merged["Stint"] = merged["Stint"].fillna(merged["InferredStint"]).astype(int)
    merged["Compound"] = merged["Compound"].fillna("UNKNOWN")
    merged["FreshTyre"] = merged["FreshTyre"].astype("boolean")
    return merged.drop(columns="InferredStint")
