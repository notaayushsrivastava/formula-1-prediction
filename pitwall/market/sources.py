"""Market prices, from free sources only, read-only.

- ManualCsvSource: data/markets/<year>-<round>.csv, typed in by hand from public odds
  pages (one line per price you saw, with the time you saw it). Small, user-made files
  that are committed to the repository.
- PolymarketSource: Polymarket's public, key-free read API (Gamma for the event, CLOB
  /prices-history for each driver's "Yes" price over time). Prediction-market prices
  are crowd probabilities, not bookmaker odds; rows keep Kind so they are never mixed
  silently. Used only if you name the event's slug (data/markets/polymarket.csv or
  --polymarket-slug): pitwall never guesses which market is the right one.

Nothing here places a bet, logs in anywhere, or handles money. Paper trading only.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from ..errors import DataNotAvailableError, PitwallError
from .odds import to_decimal

log = logging.getLogger(__name__)
MARKETS_DIR = Path("data") / "markets"
POLYMARKET_INDEX = MARKETS_DIR / "polymarket.csv"
MARKETS = ("win", "podium", "points")
KINDS = ("bookmaker", "exchange", "prediction_market")
PRICE_KINDS = ("decimal", "fractional", "american", "probability")
CSV_COLUMNS = ["timestamp_utc", "market", "driver", "price", "price_kind", "source", "kind"]
PRICE_COLUMNS = ["Timestamp", "Market", "DriverId", "Odds", "Source", "Kind"]
GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


class MarketDataError(PitwallError):
    """A price file or a market could not be read (the message says which line or name)."""


class MarketSource(Protocol):
    """Anything that returns prices for one race (PRICE_COLUMNS, Odds in decimal)."""

    def get_prices(self, year: int, rnd: int, roster: Roster) -> pd.DataFrame: ...


@dataclass
class Roster:
    """Maps a driver as written in a price source to the f1db DriverId.

    Accepts the f1db id ('max-verstappen'), the 3-letter code ('VER'), the full name
    ('Max Verstappen') or the surname ('Verstappen'), case and accents aside.
    """

    drivers: dict[str, str]                       # DriverId -> code
    unmatched: list[str] = field(default_factory=list)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Roster:
        return cls({str(d["DriverId"]): str(d["Driver"]) for d in record["drivers"]})

    def match(self, name: str) -> str | None:
        text = _slug(name)
        if not text:
            return None
        for driver_id, code in self.drivers.items():
            if text in (driver_id, _slug(code)):
                return driver_id
        surname = [d for d in self.drivers if d.endswith("-" + text.split("-")[-1])]
        if len(surname) == 1:
            return surname[0]
        joined = [d for d in self.drivers if d == text or text.endswith(d) or d.endswith(text)]
        return joined[0] if len(joined) == 1 else None

    def require(self, name: str) -> str | None:
        found = self.match(name)
        if found is None:
            self.unmatched.append(name)
        return found


def _slug(text: str) -> str:
    import unicodedata
    plain = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")


def empty_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLUMNS)


def market_file(year: int, rnd: int, folder: Path | str = MARKETS_DIR) -> Path:
    return Path(folder) / f"{year}-{rnd:02d}.csv"


# ---------- typed-in prices ----------
class ManualCsvSource:
    """Prices typed in from public pages: data/markets/<year>-<round>.csv."""

    def __init__(self, folder: Path | str = MARKETS_DIR) -> None:
        self.folder = Path(folder)

    def get_prices(self, year: int, rnd: int, roster: Roster) -> pd.DataFrame:
        path = market_file(year, rnd, self.folder)
        if not path.exists():
            return empty_prices()
        rows: list[dict[str, Any]] = []
        problems: list[str] = []
        with path.open(encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip() and not line.startswith("#")]
        for number, row in enumerate(csv.DictReader(lines), start=2):
            if not (row.get("price") or "").strip():
                continue  # an empty template line
            try:
                rows.append(_csv_row(row, roster))
            except (ValueError, KeyError) as exc:
                problems.append(f"line {number}: {exc}")
        if roster.unmatched:
            problems.append("unknown drivers: " + ", ".join(sorted(set(roster.unmatched))))
        if problems:
            raise MarketDataError(f"{path}: " + "; ".join(problems))
        return pd.DataFrame(rows, columns=PRICE_COLUMNS)


def _csv_row(row: dict[str, str], roster: Roster) -> dict[str, Any]:
    market = row["market"].strip().lower()
    kind = row["kind"].strip().lower()
    price_kind = row["price_kind"].strip().lower()
    if market not in MARKETS:
        raise ValueError(f"market must be one of {MARKETS}, got {market!r}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if price_kind not in PRICE_KINDS:
        raise ValueError(f"price_kind must be one of {PRICE_KINDS}, got {price_kind!r}")
    stamp = pd.Timestamp(row["timestamp_utc"].strip())
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    driver = roster.require(row["driver"])
    if driver is None:
        raise ValueError(f"unknown driver {row['driver']!r}")
    return {"Timestamp": stamp, "Market": market, "DriverId": driver,
            "Odds": to_decimal(row["price"].strip(), price_kind),
            "Source": row["source"].strip() or "manual", "Kind": kind}


def write_template(year: int, rnd: int, roster: Roster, folder: Path | str = MARKETS_DIR,
                   event: str = "") -> Path:
    """A CSV to fill in by hand (refuses to overwrite one that exists)."""
    path = market_file(year, rnd, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    lines = [f"# {year} round {rnd} {event}: prices you saw on PUBLIC pages, with the UTC",
             "# time you saw them. One line per price. Empty price = ignored.",
             "# market: win | podium | points;  price_kind: decimal | fractional | american |",
             "# probability;  kind: bookmaker | exchange | prediction_market.",
             "# Paper trading only: never use this to place real bets.",
             ",".join(CSV_COLUMNS)]
    lines += [f"{now},win,{code},,decimal,,bookmaker" for code in roster.drivers.values()]
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except FileExistsError:
        raise MarketDataError(f"{path} already exists; edit it instead.") from None
    return path


# ---------- Polymarket (public, read-only) ----------
class PolymarketSource:
    """'Will X win the ... Grand Prix?' markets of one Polymarket event, as prices."""

    def __init__(self, http: Any = None, index: Path | str = POLYMARKET_INDEX,
                 slugs: dict[tuple[int, int], str] | None = None, fidelity: int = 60) -> None:
        if http is None:
            from ..sources.http_json import JsonHttp
            from ..sources.jolpica import RateLimiter
            http = JsonHttp("Polymarket", RateLimiter(per_second=2.0, per_hour=3000))
        self.http = http
        self.index = Path(index)
        self.slugs = dict(slugs or {})
        self.fidelity = fidelity

    def slug_for(self, year: int, rnd: int) -> str | None:
        if (year, rnd) in self.slugs:
            return self.slugs[(year, rnd)]
        if self.index.exists():
            with self.index.open(encoding="utf-8") as handle:
                lines = [line for line in handle if line.strip() and not line.startswith("#")]
            for row in csv.DictReader(lines):
                if int(row["year"]) == year and int(row["round"]) == rnd:
                    return row["event_slug"].strip()
        return None

    def get_prices(self, year: int, rnd: int, roster: Roster, market: str = "win",
                   start: dt.datetime | None = None,
                   end: dt.datetime | None = None) -> pd.DataFrame:
        slug = self.slug_for(year, rnd)
        if not slug:
            return empty_prices()
        event = self.http.get(f"{GAMMA_URL}/events/slug/{slug}", {}, 3600.0)
        if not isinstance(event, dict) or not event.get("markets"):
            log.warning("Polymarket event %r has no markets; skipped.", slug)
            return empty_prices()
        rows: list[dict[str, Any]] = []
        for item in event["markets"]:
            name = item.get("groupItemTitle") or _name_from_question(item.get("question", ""))
            driver = roster.match(name or "")
            token = _yes_token(item.get("clobTokenIds"))
            if driver is None or token is None:
                if name and name.lower() not in ("other", "field", "any other driver"):
                    log.warning("Polymarket outcome %r not matched to a driver; skipped.", name)
                continue
            params: dict[str, Any] = {"market": token, "fidelity": self.fidelity}
            if start is not None:
                params["startTs"] = int(start.timestamp())
            if end is not None:
                params["endTs"] = int(end.timestamp())
            history = self.http.get(f"{CLOB_URL}/prices-history", params, 3600.0) or {}
            for point in history.get("history", []) if isinstance(history, dict) else []:
                price = float(point.get("p", 0))
                if 0.0 < price < 1.0:
                    rows.append({"Timestamp": pd.Timestamp(int(point["t"]), unit="s", tz="UTC"),
                                 "Market": market, "DriverId": driver, "Odds": 1.0 / price,
                                 "Source": "Polymarket", "Kind": "prediction_market"})
        return pd.DataFrame(rows, columns=PRICE_COLUMNS)

    def find_events(self, query: str) -> list[dict[str, str]]:
        """Events matching a search (to find the slug to put in polymarket.csv)."""
        found = self.http.get(f"{GAMMA_URL}/public-search", {"q": query}, 3600.0) or {}
        events = found.get("events", []) if isinstance(found, dict) else []
        return [{"slug": str(e.get("slug")), "title": str(e.get("title")),
                 "closed": str(e.get("closed"))} for e in events if e.get("slug")]


def _yes_token(raw: Any) -> str | None:
    """clobTokenIds is a JSON string '["yes", "no"]' (or already a list)."""
    try:
        tokens = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    return str(tokens[0]) if isinstance(tokens, list) and tokens else None


def _name_from_question(question: str) -> str:
    match = re.match(r"\s*Will\s+(.+?)\s+win\b", question or "", re.IGNORECASE)
    return match.group(1) if match else ""


# ---------- everything together ----------
def load_prices(year: int, rnd: int, roster: Roster,
                sources: Iterable[MarketSource] | None = None) -> pd.DataFrame:
    """Prices from every source (manual CSV, then Polymarket if configured)."""
    chosen: list[MarketSource] = (list(sources) if sources is not None
                                  else [ManualCsvSource(), PolymarketSource()])
    parts = [s.get_prices(year, rnd, roster) for s in chosen]
    parts = [p for p in parts if not p.empty]
    if not parts:
        raise DataNotAvailableError(
            f"No market prices for {year} round {rnd}. Fill "
            f"{market_file(year, rnd)} (pitwall-f1 market template) or name a Polymarket "
            f"event in {POLYMARKET_INDEX}.")
    out = pd.concat(parts, ignore_index=True)
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True)
    return out.sort_values("Timestamp").reset_index(drop=True)
