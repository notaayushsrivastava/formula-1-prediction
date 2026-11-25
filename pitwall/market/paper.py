"""Paper trading: hypothetical stakes in a public ledger. NO REAL MONEY.

Real-money online betting is illegal in India (Online Gaming Act, in force since
1 May 2026), and this is an educational project. The ledger exists to answer one
question honestly: if the engine's odds had been used against real market prices,
would it have made or lost (pretend) money, and is that more than luck?

Every ticket is written BEFORE the race next to the prediction it uses
(predictions/paper/<year>-<round>-<moment>.json) and never overwritten; the git commit
time is the proof. After the race a separate settlement file
(<stem>.settled.json) records the result. Each moment (pre, practice, quali) is its own
strategy with its own pretend bankroll.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..errors import PitwallError
from ..predict.backtest import git_hash
from ..predict.records import record_frame
from .compare import LABEL_OF_MARKET, market_snapshot, race_table
from .staking import StakePolicy, stakes

PAPER_DIR = Path("predictions") / "paper"
INITIAL_BANKROLL = 1000.0   # pretend units
DEFAULT_START_UTC = "12:00"  # race start when the schedule has no time (closing line)
RUIN_FRACTION = 0.5          # "ruin" = losing half the pretend bankroll


class TicketExistsError(PitwallError):
    """A paper ticket for this race and moment already exists (never overwritten)."""


def ticket_path(stem: str, folder: Path | str = PAPER_DIR) -> Path:
    return Path(folder) / f"{stem}.json"


def settlement_path(stem: str, folder: Path | str = PAPER_DIR) -> Path:
    return Path(folder) / f"{stem}.settled.json"


def race_start(record: dict[str, Any]) -> pd.Timestamp:
    """When the race starts (UTC): the closing line is the last price before this."""
    return pd.Timestamp(f"{record['race_date']}T{DEFAULT_START_UTC}", tz="UTC")


# ---------- the ledger ----------
def bankroll(moment: str, folder: Path | str = PAPER_DIR) -> float:
    """The moment's pretend bankroll after every settled ticket so far."""
    pnl = sum(s["pnl"] for s in load_settlements(folder) if s["moment"] == moment)
    return INITIAL_BANKROLL + pnl


def make_ticket(record: dict[str, Any], prices: pd.DataFrame, policy: StakePolicy,
                bank: float, market: str = "win", method: str = "shin",
                now: dt.datetime | None = None) -> dict[str, Any]:
    """Paper bets for one prediction record, using only prices seen BEFORE it was made."""
    cutoff = pd.Timestamp(record["created_utc"])
    snapshot = market_snapshot(prices, market, cutoff, method)
    if snapshot.empty:
        raise PitwallError("No usable market prices from before the prediction was made.")
    table = race_table(record_frame(record), snapshot, market)
    fractions = stakes(table["p_engine"].to_numpy(), table["p_market"].to_numpy(),
                       table["best_odds"].to_numpy(), policy, exclusive=market == "win")
    table["stake_fraction"] = fractions
    table["stake"] = np.round(fractions * bank, 2)
    bets = table[table["stake"] > 0]
    stem = record.get("_stem") or f"{record['year']}-{record['round']:02d}-{record['moment']}"
    stamp = (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds")
    return {
        "schema": 1, "stem": stem, "year": record["year"], "round": record["round"],
        "event": record["event"], "moment": record["moment"], "market": market,
        "created_utc": stamp, "prediction_created_utc": record["created_utc"],
        "git": git_hash(), "devig": method, "policy": asdict(policy), "bankroll_before": bank,
        "race_start_utc": race_start(record).isoformat(),
        "market_table": _records(table.drop(columns=["stake_fraction", "stake"])),
        "bets": _records(bets),
        "note": ("Paper trading only: hypothetical stakes, no real money. "
                 "Not betting advice."),
    }


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].map(lambda t: t.isoformat())
    return json.loads(out.to_json(orient="records", double_precision=6))


def save_ticket(ticket: dict[str, Any], folder: Path | str = PAPER_DIR) -> Path:
    path = ticket_path(ticket["stem"], folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(ticket, handle, indent=1)
            handle.write("\n")
    except FileExistsError:
        raise TicketExistsError(f"{path} already exists; tickets are never overwritten.") \
            from None
    return path


def load_tickets(folder: Path | str = PAPER_DIR) -> list[dict[str, Any]]:
    base = Path(folder)
    if not base.exists():
        return []
    paths = [p for p in base.glob("*.json") if not p.name.endswith(".settled.json")]
    tickets = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    return sorted(tickets, key=lambda t: (t["year"], t["round"], t["moment"]))


def load_settlements(folder: Path | str = PAPER_DIR) -> list[dict[str, Any]]:
    base = Path(folder)
    if not base.exists():
        return []
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in base.glob("*.settled.json")]
    return sorted(rows, key=lambda s: (s["year"], s["round"], s["moment"]))


# ---------- after the race ----------
def settle(ticket: dict[str, Any], results: pd.DataFrame,
           closing: pd.DataFrame | None = None) -> dict[str, Any]:
    """Profit and loss of each paper bet. results: DriverId + Win/Podium/PointsFinish.

    closing: market_snapshot at the race start, for closing line value (CLV): did we
    get better odds than the last price before the start? CLV > 0 is the classic sign
    of real skill, visible long before profits are statistically meaningful.
    """
    label = LABEL_OF_MARKET[ticket["market"]]
    outcome = dict(zip(results["DriverId"], results[label]))
    close = (dict(zip(closing["DriverId"], closing["best_odds"]))
             if closing is not None and not closing.empty else {})
    bets = []
    for bet in ticket["bets"]:
        won = outcome.get(bet["DriverId"])
        if won is None or pd.isna(won):
            pnl, result = 0.0, "void"   # did not start: stake returned
        else:
            won = bool(won)
            pnl = bet["stake"] * (bet["best_odds"] - 1.0) if won else -bet["stake"]
            result = "won" if won else "lost"
        closing_odds = close.get(bet["DriverId"])
        clv = bet["best_odds"] / closing_odds - 1.0 if closing_odds else None
        bets.append({**bet, "result": result, "pnl": round(pnl, 2),
                     "closing_odds": closing_odds, "clv": clv})
    staked = sum(b["stake"] for b in bets if b["result"] != "void")
    return {"stem": ticket["stem"], "year": ticket["year"], "round": ticket["round"],
            "event": ticket["event"], "moment": ticket["moment"], "market": ticket["market"],
            "settled_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "bankroll_before": ticket["bankroll_before"], "staked": round(staked, 2),
            "pnl": round(sum(b["pnl"] for b in bets), 2), "bets": bets}


def save_settlement(settlement: dict[str, Any], folder: Path | str = PAPER_DIR) -> Path:
    path = settlement_path(settlement["stem"], folder)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(settlement, handle, indent=1)
            handle.write("\n")
    except FileExistsError:
        raise TicketExistsError(f"{path} already exists.") from None
    return path


def closing_snapshot(ticket: dict[str, Any], prices: pd.DataFrame) -> pd.DataFrame:
    return market_snapshot(prices, ticket["market"], pd.Timestamp(ticket["race_start_utc"]),
                           ticket.get("devig", "shin"))


# ---------- how it went ----------
@dataclass(frozen=True)
class PaperStats:
    """One moment's paper-trading results."""

    moment: str
    races: int
    bets: int
    staked: float
    pnl: float
    roi: float                 # profit per unit staked
    roi_low: float             # 95% bootstrap interval over races
    roi_high: float
    hit_rate: float
    average_odds: float
    mean_clv: float
    max_drawdown: float        # worst fall from a peak, as a fraction of that peak
    race_sharpe: float         # mean / sd of per-race return on bankroll
    risk_of_ruin: float        # chance of losing half the bankroll in the next 50 races
    races_needed: float        # races for the mean return to be ~2 standard errors from 0
    bankroll: float


def paper_stats(settlements: list[dict[str, Any]], moment: str, n_boot: int = 5000,
                seed: int = 0) -> PaperStats | None:
    rows = [s for s in settlements if s["moment"] == moment]
    if not rows:
        return None
    bets = [b for s in rows for b in s["bets"] if b["result"] != "void"]
    staked = sum(b["stake"] for b in bets)
    pnl = sum(s["pnl"] for s in rows)
    returns = np.array([s["pnl"] / s["bankroll_before"] for s in rows])
    rng = np.random.default_rng(seed)
    race_pnl = np.array([s["pnl"] for s in rows])
    race_staked = np.array([s["staked"] for s in rows])
    if staked > 0 and len(rows) > 1:
        picks = rng.integers(0, len(rows), size=(n_boot, len(rows)))
        boot = race_pnl[picks].sum(1) / np.maximum(race_staked[picks].sum(1), 1e-9)
        low, high = np.quantile(boot, [0.025, 0.975])
    else:
        low = high = float("nan")
    curve = INITIAL_BANKROLL + np.cumsum(race_pnl)
    peaks = np.maximum.accumulate(np.concatenate([[INITIAL_BANKROLL], curve]))[1:]
    drawdown = float(np.max((peaks - curve) / peaks)) if len(curve) else 0.0
    sd = float(returns.std(ddof=1)) if len(returns) > 1 else float("nan")
    mean = float(returns.mean())
    clvs = [b["clv"] for b in bets if b.get("clv") is not None]
    return PaperStats(
        moment, len(rows), len(bets), round(staked, 2), round(pnl, 2),
        pnl / staked if staked else 0.0, float(low), float(high),
        float(np.mean([b["result"] == "won" for b in bets])) if bets else float("nan"),
        float(np.mean([b["best_odds"] for b in bets])) if bets else float("nan"),
        float(np.mean(clvs)) if clvs else float("nan"), drawdown,
        mean / sd if sd and sd > 0 else float("nan"),
        risk_of_ruin(returns, seed=seed),
        (1.96 * sd / abs(mean)) ** 2 if sd == sd and mean != 0 else float("inf"),
        float(curve[-1]) if len(curve) else INITIAL_BANKROLL)


def risk_of_ruin(returns: np.ndarray, horizon: int = 50, n_sims: int = 5000,
                 seed: int = 0) -> float:
    """Chance the bankroll falls below RUIN_FRACTION within `horizon` races, resampling
    the per-race returns seen so far (a bootstrap; meaningless with very few races)."""
    if len(returns) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(returns, size=(n_sims, horizon), replace=True)
    paths = np.cumprod(1.0 + draws, axis=1)
    return float(np.mean(paths.min(axis=1) < RUIN_FRACTION))
