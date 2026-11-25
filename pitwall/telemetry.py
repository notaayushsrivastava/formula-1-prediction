"""Phase 5: car telemetry (speed, throttle, gear...) and positions for single laps."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .core import Session
from .errors import DataNotAvailableError
from .sources.base import Record
from .sources.livetiming import LiveTimingClient
from .sources.telemetry_feeds import parse_car_data, parse_positions, parse_timing_laps

TIMING_LAP_COLUMNS = ["Driver", "Number", "LapNumber", "LapTimeSeconds",
                      "LapStartSeconds", "LapEndSeconds"]


class Telemetry(pd.DataFrame):
    """Samples of one lap: Time (s from lap start), Distance (m), Speed (km/h), RPM,
    Gear, Throttle (%), Brake (bool), X/Y/Z (track position), and DRS up to 2025."""

    @property
    def _constructor(self) -> type[Telemetry]:
        return Telemetry

    def add_distance(self) -> Telemetry:
        """Distance in metres from the start line, by integrating speed (trapezoid rule)."""
        out = self.copy()
        speed = out["Speed"].to_numpy(float) / 3.6  # km/h -> m/s
        time = out["Time"].to_numpy(float)
        steps = (speed[1:] + speed[:-1]) / 2 * np.diff(time)
        before_first = speed[0] * time[0]  # first sample comes just after the line
        out["Distance"] = before_first + np.concatenate([[0.0], np.cumsum(steps)])
        return out


def load_telemetry(session: Session, drivers: Iterable[str] | None = None,
                   client: LiveTimingClient | None = None) -> Session:
    """Download timing laps, car data and positions (works for 'R', 'Q' and 'S').

    drivers=['VER', 'NOR'] keeps only those cars in memory (the feeds are large).
    """
    if session.event is None or session.results.empty:
        raise ValueError("Load the session first: get_session(year, event, kind).load()")
    client = client or LiveTimingClient()
    numbers = None if drivers is None else {_number(session, d) for d in drivers}
    path = client.session_path(session.year, str(session.event["Date"]), session.kind)
    session.timing_laps = _timing_laps(parse_timing_laps(client.stream(path, "TimingData")),
                                       session.results)
    car = parse_car_data(client.z_stream(path, "CarData.z"), numbers)
    pos = parse_positions(client.z_stream(path, "Position.z"), numbers)
    session.car_data = _with_drivers(car, session.results)
    session.positions = _with_drivers(pos, session.results)
    return session


def timing_lap(session: Session, driver: str, lap: int | str = "fastest") -> pd.Series:
    """One row of session.timing_laps: a lap number, or the official fastest lap."""
    code = session.get_driver(driver)["Driver"]
    laps = _require(session, "timing_laps")
    drv = laps[laps["Driver"] == code]
    if lap != "fastest":
        hit = drv[drv["LapNumber"] == int(lap)]
        if hit.empty:
            raise DataNotAvailableError(f"{code} has no lap {lap} in {session.name}.")
        return hit.iloc[0]
    timed = drv.dropna(subset=["LapTimeSeconds", "LapStartSeconds"])
    if timed.empty:
        raise DataNotAvailableError(f"{code} has no timed laps in {session.name}.")
    official = _official_best(session, code)
    if official is not None:
        match = timed[(timed["LapTimeSeconds"] - official).abs() < 0.002]
        if not match.empty:
            return match.iloc[0]
    return timed.loc[timed["LapTimeSeconds"].idxmin()]


def lap_telemetry(session: Session, driver: str, lap: int | str = "fastest") -> Telemetry:
    """Car data and track position for one lap, with Time and Distance from the line."""
    chosen = timing_lap(session, driver, lap)
    start, end = chosen["LapStartSeconds"], chosen["LapEndSeconds"]
    car = _require(session, "car_data")
    car = car[(car["Driver"] == chosen["Driver"]) & car["SessionSeconds"].between(start, end)]
    if car.empty:
        raise DataNotAvailableError(
            f"No car data for {chosen['Driver']} lap {chosen['LapNumber']}. "
            "Was this driver included in load_telemetry(drivers=...)?")
    pos = session.positions
    pos = pos.loc[pos["Driver"] == chosen["Driver"], ["SessionSeconds", "X", "Y", "Z"]]
    merged = pd.merge_asof(car.sort_values("SessionSeconds"), pos.sort_values("SessionSeconds"),
                           on="SessionSeconds", direction="nearest", tolerance=1.0)
    merged["Time"] = merged["SessionSeconds"] - start
    merged["Brake"] = merged["Brake"].fillna(0).astype(float) > 0
    merged["LapNumber"] = int(chosen["LapNumber"])
    if merged["DRS"].isna().all():
        merged = merged.drop(columns="DRS")  # no DRS channel (e.g. 2026 cars)
    return Telemetry(merged.reset_index(drop=True)).add_distance()


def delta_time(reference: Telemetry, other: Telemetry) -> pd.DataFrame:
    """Gap of `other` to `reference` along the lap (positive = other is behind)."""
    ref_time = np.interp(other["Distance"], reference["Distance"], reference["Time"])
    return pd.DataFrame({"Distance": other["Distance"].to_numpy(),
                         "Delta": other["Time"].to_numpy() - ref_time})


# ---------- helpers ----------
def _timing_laps(records: list[Record], results: pd.DataFrame) -> pd.DataFrame:
    laps = pd.DataFrame(records, columns=["Number", "LapNumber", "LapTimeSeconds", "LapEndSeconds"])
    laps["LapTimeSeconds"] = pd.to_numeric(laps["LapTimeSeconds"], errors="coerce")
    laps["Driver"] = laps["Number"].map(_number_map(results))
    laps = laps.sort_values(["Number", "LapNumber"]).reset_index(drop=True)
    previous_end = laps.groupby("Number")["LapEndSeconds"].shift()
    laps["LapStartSeconds"] = (laps["LapEndSeconds"] - laps["LapTimeSeconds"]).fillna(previous_end)
    return laps[TIMING_LAP_COLUMNS]


def _with_drivers(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "Driver", out["Number"].map(_number_map(results)))
    return out


def _number_map(results: pd.DataFrame) -> dict[str, str]:
    return dict(zip(results["Number"].astype(str), results["Driver"]))


def _number(session: Session, driver: str) -> str:
    return str(session.get_driver(driver)["Number"])


def _official_best(session: Session, code: str) -> float | None:
    """Jolpica's official best lap for the driver (valid laps only), if known."""
    res = session.results
    for column in ("BestSeconds", "FastestLapSeconds"):
        if column in res:
            value = res.loc[res["Driver"] == code, column]
            if not value.empty and pd.notna(value.iloc[0]):
                return float(value.iloc[0])
    return None


def _require(session: Session, attribute: str) -> pd.DataFrame:
    df = getattr(session, attribute, None)
    if df is None or df.empty:
        raise ValueError("No telemetry loaded: call pitwall.load_telemetry(session) first.")
    return df
