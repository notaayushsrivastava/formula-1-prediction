"""Parsers for the Live Timing feeds behind telemetry: TimingData, CarData.z, Position.z."""
from __future__ import annotations

import base64
import json
import zlib
from typing import Any

import pandas as pd

from .base import Record, parse_time_to_seconds

StreamEntry = tuple[float, dict[str, Any]]
CAR_CHANNELS = {"0": "RPM", "2": "Speed", "3": "Gear", "4": "Throttle", "5": "Brake", "45": "DRS"}


def parse_z_stream(text: str) -> list[StreamEntry]:
    """'00:01:02.345"<base64>"' lines -> [(62.345, {...}), ...] (raw-deflate JSON)."""
    entries: list[StreamEntry] = []
    for line in text.lstrip("\ufeff").splitlines():
        quote = line.find('"')
        seconds = parse_time_to_seconds(line[:quote]) if quote > 0 else None
        if seconds is None:
            continue
        packed = base64.b64decode(json.loads(line[quote:]))
        entries.append((seconds, json.loads(zlib.decompress(packed, -zlib.MAX_WBITS))))
    return entries


def parse_car_data(entries: list[StreamEntry], numbers: set[str] | None = None) -> pd.DataFrame:
    """CarData.z -> one row per car sample: Number, SessionSeconds, RPM, Speed, Gear, ..."""
    rows = []
    for line_seconds, msg in entries:
        for entry in msg.get("Entries", []):
            for number, car in entry.get("Cars", {}).items():
                if numbers is None or number in numbers:
                    channels = car.get("Channels", {})
                    rows.append([number, line_seconds, entry.get("Utc"),
                                 *(channels.get(key) for key in CAR_CHANNELS)])
    df = pd.DataFrame(rows, columns=["Number", "LineSeconds", "Utc", *CAR_CHANNELS.values()])
    names = list(CAR_CHANNELS.values())
    df[names] = df[names].apply(pd.to_numeric, errors="coerce")
    return _with_session_time(df)


def parse_positions(entries: list[StreamEntry], numbers: set[str] | None = None) -> pd.DataFrame:
    """Position.z -> one row per sample: Number, SessionSeconds, X, Y, Z, Status."""
    rows = []
    for line_seconds, msg in entries:
        for sample in msg.get("Position", []):
            for number, pos in sample.get("Entries", {}).items():
                if numbers is None or number in numbers:
                    rows.append([number, line_seconds, sample.get("Timestamp"),
                                 pos.get("X"), pos.get("Y"), pos.get("Z"), pos.get("Status")])
    df = pd.DataFrame(rows, columns=["Number", "LineSeconds", "Utc", "X", "Y", "Z", "Status"])
    df[["X", "Y", "Z"]] = df[["X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce")
    return _with_session_time(df)


def parse_timing_laps(entries: list[StreamEntry], window: float = 5.0) -> list[Record]:
    """TimingData -> one record per completed lap: Number, LapNumber, LapTimeSeconds, LapEndSeconds."""
    ends: dict[str, dict[int, float]] = {}
    times: dict[str, list[tuple[float, float | None]]] = {}
    for t, msg in entries:
        lines = msg.get("Lines") or {}
        for number, line in lines.items() if isinstance(lines, dict) else []:
            if not isinstance(line, dict):
                continue
            if "NumberOfLaps" in line:
                ends.setdefault(number, {}).setdefault(int(line["NumberOfLaps"]), t)
            last = line.get("LastLapTime")
            if isinstance(last, dict) and "Value" in last:
                times.setdefault(number, []).append((t, parse_time_to_seconds(last["Value"])))
    return [
        {"Number": number, "LapNumber": lap, "LapEndSeconds": end,
         "LapTimeSeconds": _nearest_time(times.get(number, []), end, window)}
        for number, laps in ends.items() for lap, end in laps.items()
    ]


def _nearest_time(events: list[tuple[float, float | None]], end: float,
                  window: float) -> float | None:
    """The lap time published closest to the lap's end (within `window` seconds)."""
    close = [(abs(t - end), value) for t, value in events
             if value is not None and abs(t - end) <= window]
    return min(close)[1] if close else None


def _with_session_time(df: pd.DataFrame) -> pd.DataFrame:
    """Convert each sample's UTC stamp to session seconds (the clock laps use)."""
    out = df.drop(columns=["LineSeconds", "Utc"])
    if df.empty:
        out["SessionSeconds"] = pd.Series(dtype=float)
        return out
    utc = pd.to_datetime(df["Utc"], utc=True, format="ISO8601")
    utc_seconds = (utc - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    newest = utc_seconds.groupby(df["LineSeconds"]).transform("max")
    offset = float((df["LineSeconds"] - newest).median())
    out["SessionSeconds"] = utc_seconds + offset
    return out.sort_values(["Number", "SessionSeconds"]).reset_index(drop=True)
