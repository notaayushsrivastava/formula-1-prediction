"""Bulk race history from the f1db CSV release (github.com/f1db/f1db, CC BY 4.0).

One zip holds every World Championship race since 1950, so the whole history costs
ONE download instead of thousands of Jolpica requests. f1db has results, grids,
qualifying, pit stops and standings, but no lap-by-lap times (we don't need them:
practice pace comes from OpenF1).
"""
from __future__ import annotations

import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from ..cache import DEFAULT_CACHE_DIR
from ..errors import DataNotAvailableError, SourceRequestError
from .base import Record
from .http_json import USER_AGENT

log = logging.getLogger(__name__)

RELEASES_URL = "https://github.com/f1db/f1db/releases"
ASSET = "f1db-csv.zip"
LATEST_MAX_AGE = 2 * 24 * 3600  # f1db releases after every race; re-check every 2 days
ATTRIBUTION = "F1DB (github.com/f1db/f1db), licensed CC BY 4.0"


class F1dbLoader:
    """Downloads the f1db CSV release once and reads it. Follows HistorySource.

    version='latest' follows new releases; pin one (e.g. '2026.20.0') for a
    backtest you can reproduce exactly.
    """

    def __init__(self, version: str = "latest", directory: Path | str | None = None,
                 timeout: float = 120.0) -> None:
        self.version = version.removeprefix("v")
        self.directory = Path(directory) if directory else DEFAULT_CACHE_DIR / "f1db"
        self.timeout = timeout
        self._tables: dict[str, pd.DataFrame] = {}

    @property
    def url(self) -> str:
        """Download URL of the release asset."""
        if self.version == "latest":
            return f"{RELEASES_URL}/latest/download/{ASSET}"
        return f"{RELEASES_URL}/download/v{self.version}/{ASSET}"

    @property
    def zip_path(self) -> Path:
        """Where the zip is kept on disk."""
        return self.directory / f"f1db-csv-{self.version}.zip"

    def ensure_downloaded(self) -> Path:
        """The local zip, downloading it if missing (or if 'latest' is 2+ days old)."""
        path = self.zip_path
        if path.exists() and not self._is_stale(path):
            return path
        try:
            content = self._download()
        except SourceRequestError:
            if path.exists():
                log.warning("Could not refresh f1db; using the copy from %s",
                            time.ctime(path.stat().st_mtime))
                return path
            raise
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(content)
        tmp.replace(path)  # atomic: never a half-written zip
        self._tables.clear()
        return path

    def table(self, name: str) -> pd.DataFrame:
        """One CSV of the release, e.g. table('races-race-results')."""
        if name not in self._tables:
            with zipfile.ZipFile(self.ensure_downloaded()) as archive:
                member = _find_member(archive.namelist(), name)
                with archive.open(member) as handle:
                    self._tables[name] = pd.read_csv(handle, low_memory=False)
        return self._tables[name]

    # ---------- HistorySource ----------
    def get_races(self, min_year: int = 2000) -> list[Record]:
        """One record per race (see HistorySource)."""
        races = self._races(min_year)
        names = self.table("grands-prix").set_index("id")["name"]
        circuits = self.table("circuits").set_index("id")
        return _records(pd.DataFrame({
            "RaceId": races["id"], "Year": races["year"], "Round": races["round"],
            "Date": races["date"], "Time": races["time"],
            "EventName": races["grandPrixId"].map(names),
            "CircuitId": races["circuitId"], "CircuitType": races["circuitType"],
            "DistanceKm": races["distance"],
            "Lat": races["circuitId"].map(circuits["latitude"]),
            "Long": races["circuitId"].map(circuits["longitude"]),
            "HasSprint": races["sprintRaceDate"].notna(),
        }))

    def get_race_results(self, min_year: int = 2000) -> list[Record]:
        """One record per car per race (see HistorySource)."""
        df = self._for_races("races-race-results", min_year)
        return _records(pd.DataFrame({
            "RaceId": df["raceId"], "DriverId": df["driverId"], "TeamId": df["constructorId"],
            "Number": df["driverNumber"], "Position": df["positionNumber"],
            "PositionText": df["positionText"], "DisplayOrder": df["positionDisplayOrder"],
            "GridPosition": df["gridPositionNumber"], "GridText": df["gridPositionText"],
            "Laps": df["laps"], "TimeMillis": df["timeMillis"],
            "ReasonRetired": df["reasonRetired"], "Points": df["points"],
        }))

    def get_qualifying(self, min_year: int = 2000) -> list[Record]:
        """Qualifying times in seconds (single-session formats count as Q1)."""
        df = self._for_races("races-qualifying-results", min_year)
        q1 = df["q1Millis"].fillna(df["timeMillis"])
        return _records(pd.DataFrame({
            "RaceId": df["raceId"], "DriverId": df["driverId"],
            "QualiPosition": df["positionNumber"], "Q1Seconds": q1 / 1000,
            "Q2Seconds": df["q2Millis"] / 1000, "Q3Seconds": df["q3Millis"] / 1000,
        }))

    def get_pitstops(self, min_year: int = 2000) -> list[Record]:
        """One record per pit stop (f1db has them from 2011)."""
        df = self._for_races("races-pit-stops", min_year)
        return _records(pd.DataFrame({
            "RaceId": df["raceId"], "DriverId": df["driverId"], "TeamId": df["constructorId"],
            "Stop": df["stop"], "Lap": df["lap"], "DurationSeconds": df["timeMillis"] / 1000,
        }))

    def get_driver_standings(self, min_year: int = 2000) -> list[Record]:
        """Championship points after each race."""
        df = self._for_races("races-driver-standings", min_year)
        return _records(pd.DataFrame({
            "RaceId": df["raceId"], "DriverId": df["driverId"], "Points": df["points"]}))

    def get_drivers(self) -> list[Record]:
        """Every driver with code and date of birth."""
        df = self.table("drivers")
        return _records(pd.DataFrame({
            "DriverId": df["id"], "Driver": df["abbreviation"],
            "FullName": df["fullName"], "DateOfBirth": df["dateOfBirth"]}))

    # ---------- helpers ----------
    def _download(self) -> bytes:
        log.info("Downloading %s", self.url)
        try:
            resp = requests.get(self.url, timeout=self.timeout,
                                headers={"User-Agent": USER_AGENT})
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise SourceRequestError(
                f"Could not reach GitHub for f1db ({type(exc).__name__}).") from None
        if not resp.ok:
            raise SourceRequestError(
                f"HTTP {resp.status_code} for {self.url}. Check {RELEASES_URL} that "
                f"the release has an '{ASSET}' asset.")
        return resp.content

    def _is_stale(self, path: Path) -> bool:
        return self.version == "latest" and time.time() - path.stat().st_mtime > LATEST_MAX_AGE

    def _races(self, min_year: int) -> pd.DataFrame:
        races = self.table("races")
        return races[races["year"] >= min_year]

    def _for_races(self, name: str, min_year: int) -> pd.DataFrame:
        df = self.table(name)
        return df[df["raceId"].isin(self._races(min_year)["id"])]


def _find_member(names: list[str], table: str) -> str:
    """The zip member 'f1db-<table>.csv' (in any folder)."""
    wanted = f"f1db-{table}.csv"
    for name in names:
        if name.rsplit("/", 1)[-1] == wanted:
            return name
    shown = ", ".join(sorted(n.rsplit("/", 1)[-1] for n in names))
    raise DataNotAvailableError(
        f"'{wanted}' is not in the f1db release. It contains: {shown}")


def _records(df: pd.DataFrame) -> list[Record]:
    """DataFrame -> list of dicts with plain Python values and None for missing."""
    clean = df.astype(object).where(df.notna(), None)
    return clean.to_dict("records")
