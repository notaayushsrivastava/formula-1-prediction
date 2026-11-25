"""The engine against the market.

1. market_snapshot: the market's fair probabilities at one moment, from prices seen
   BEFORE that moment only (the leakage guard), margin removed per source, then
   averaged over sources.
2. race_table: engine vs market per driver: edge, best odds, expected value.
3. score_vs_market: log loss of engine and market on the same finished races, with a
   paired bootstrap over races (is the difference real?).
4. Benter's two-stage model (Benter 1994): a conditional logit on
   log(p_engine) and log(p_market), fitted on past races only. If the engine's weight
   is clearly above zero, the engine knows something the market does not; if not,
   the honest conclusion is "no edge".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from ..predict.backtest import LeakageError
from ..predict.metrics import EPS, BootstrapResult, paired_bootstrap
from .odds import devig, expected_value

PROB_OF_MARKET = {"win": "P_win", "podium": "P_podium", "points": "P_points"}
LABEL_OF_MARKET = {"win": "Win", "podium": "Podium", "points": "PointsFinish"}
PLACES = {"win": 1, "podium": 3, "points": 10}


def latest_prices(prices: pd.DataFrame, market: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Each source's latest price per driver strictly before `cutoff` (UTC)."""
    cut = pd.Timestamp(cutoff)
    cut = cut.tz_localize("UTC") if cut.tzinfo is None else cut.tz_convert("UTC")
    rows = prices[(prices["Market"] == market) & (prices["Timestamp"] < cut)]
    if rows.empty:
        return rows
    last = rows.sort_values("Timestamp").groupby(["Source", "DriverId"], as_index=False).last()
    if (last["Timestamp"] >= cut).any():  # cannot happen; kept as the explicit guard
        raise LeakageError("A market price from after the prediction would be used.")
    return last


def market_snapshot(prices: pd.DataFrame, market: str, cutoff: pd.Timestamp,
                    method: str = "shin", min_drivers: int = 3) -> pd.DataFrame:
    """Fair market probability per driver at `cutoff`, plus the best odds on offer.

    Each source's book is de-vigged on its own (a source quoting fewer than min_drivers
    is skipped: its margin cannot be estimated), then fair probabilities are averaged
    across sources and renormalised to the market's number of places.
    """
    last = latest_prices(prices, market, cutoff)
    fair_parts = []
    usable = []
    for source, book in last.groupby("Source"):
        if len(book) < min_drivers:
            continue
        usable.append(source)
        fair = devig(book["Odds"].to_numpy(float), method, PLACES[market])
        fair_parts.append(pd.DataFrame({"DriverId": book["DriverId"].to_numpy(), "Fair": fair,
                                        "Source": source}))
    if not fair_parts:
        return pd.DataFrame(columns=["DriverId", "p_market", "best_odds", "best_source",
                                     "sources", "seen_utc"])
    fair = pd.concat(fair_parts, ignore_index=True)
    consensus = fair.groupby("DriverId")["Fair"].agg(["mean", "count"]).reset_index()
    consensus["mean"] *= PLACES[market] / consensus["mean"].sum()
    last = last[last["Source"].isin(usable)]
    best = last.loc[last.groupby("DriverId")["Odds"].idxmax(),
                    ["DriverId", "Odds", "Source", "Timestamp"]]
    out = consensus.merge(best, on="DriverId")
    return out.rename(columns={"mean": "p_market", "count": "sources", "Odds": "best_odds",
                               "Source": "best_source", "Timestamp": "seen_utc"})


def race_table(engine: pd.DataFrame, snapshot: pd.DataFrame, market: str = "win") -> pd.DataFrame:
    """Engine vs market per driver. engine: the record's driver table (P_win, ...)."""
    column = PROB_OF_MARKET[market]
    table = engine[["DriverId", "Driver", column]].rename(columns={column: "p_engine"})
    table = table.merge(snapshot, on="DriverId", how="inner")
    table["edge"] = table["p_engine"] - table["p_market"]
    table["ev"] = expected_value(table["p_engine"], table["best_odds"])
    return table.sort_values("p_engine", ascending=False).reset_index(drop=True)


# ---------- scoring the two forecasters ----------
def _row_loss(p: pd.Series, y: pd.Series) -> pd.Series:
    q = np.clip(p.to_numpy(float), EPS, 1 - EPS)
    label = y.to_numpy(float)
    return pd.Series(-(label * np.log(q) + (1 - label) * np.log(1 - q)), index=p.index)


@dataclass(frozen=True)
class MarketScore:
    """Engine vs market on the same rows (lower log loss is better)."""

    races: int
    engine_logloss: float
    market_logloss: float
    difference: BootstrapResult   # engine minus market, per race (negative = engine better)

    @property
    def verdict(self) -> str:
        d = self.difference
        if self.races < 2:
            return "not enough races yet"
        if d.high < 0:
            return "the engine is clearly better calibrated than the market"
        if d.low > 0:
            return "the market is clearly better than the engine"
        return "no clear difference between engine and market"


def score_vs_market(rows: pd.DataFrame, n_boot: int = 5000) -> MarketScore:
    """rows: one per (RaceId, DriverId) with p_engine, p_market and Label (0/1)."""
    loss_e = _row_loss(rows["p_engine"], rows["Label"])
    loss_m = _row_loss(rows["p_market"], rows["Label"])
    per_e = loss_e.groupby(rows["RaceId"]).mean()
    per_m = loss_m.groupby(rows["RaceId"]).mean()
    diff = paired_bootstrap(per_e, per_m, n_boot=n_boot)
    return MarketScore(int(rows["RaceId"].nunique()), float(loss_e.mean()),
                       float(loss_m.mean()), diff)


# ---------- Benter's two-stage model ----------
@dataclass(frozen=True)
class BenterFit:
    """Conditional-logit weights on log(p_engine) and log(p_market)."""

    engine_weight: float
    market_weight: float
    engine_se: float
    market_se: float
    races: int
    loglik: float
    loglik_market_only: float

    @property
    def engine_adds_information(self) -> bool:
        """The engine's weight is above zero with ~95% confidence."""
        return self.races >= 10 and self.engine_weight - 1.96 * self.engine_se > 0

    def combine(self, p_engine: np.ndarray, p_market: np.ndarray) -> np.ndarray:
        return _softmax(self.engine_weight * _log(p_engine) + self.market_weight * _log(p_market))


def _log(p: np.ndarray | pd.Series) -> np.ndarray:
    return np.log(np.clip(np.asarray(p, float), EPS, 1.0))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _neg_loglik(w: np.ndarray, groups: list[tuple[np.ndarray, np.ndarray, int]]) -> float:
    total = 0.0
    for le, lm, winner in groups:
        u = w[0] * le + w[1] * lm
        total -= u[winner] - (u.max() + np.log(np.exp(u - u.max()).sum()))
    return total


def _groups(rows: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, int]]:
    out = []
    for _, race in rows.groupby("RaceId", sort=False):
        winners = np.flatnonzero(race["Label"].to_numpy() == 1)
        if len(winners) != 1:
            continue  # need exactly one winner among the priced drivers
        out.append((_log(race["p_engine"]), _log(race["p_market"]), int(winners[0])))
    return out


def fit_benter(rows: pd.DataFrame) -> BenterFit:
    """Fit on WIN rows of finished races (RaceId, p_engine, p_market, Label)."""
    groups = _groups(rows)
    if len(groups) < 2:
        raise ValueError("Benter's model needs at least 2 finished races with prices.")
    fit = optimize.minimize(_neg_loglik, np.array([0.0, 1.0]), args=(groups,), method="BFGS")
    se = _standard_errors(fit.x, groups)
    market_only = optimize.minimize_scalar(lambda b: _neg_loglik(np.array([0.0, b]), groups),
                                           bounds=(0.0, 5.0), method="bounded")
    return BenterFit(float(fit.x[0]), float(fit.x[1]), se[0], se[1], len(groups),
                     -float(fit.fun), -float(market_only.fun))


def _standard_errors(w: np.ndarray, groups: list) -> tuple[float, float]:
    """From a numerical Hessian of the negative log-likelihood."""
    h = 1e-4
    hessian = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            def f(a: float, b: float) -> float:
                step = np.array(w, float)
                step[i] += a
                step[j] += b
                return _neg_loglik(step, groups)
            hessian[i, j] = (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h * h)
    try:
        cov = np.linalg.inv(hessian)
        return float(np.sqrt(max(cov[0, 0], 0))), float(np.sqrt(max(cov[1, 1], 0)))
    except np.linalg.LinAlgError:
        return float("inf"), float("inf")


def benter_walk_forward(rows: pd.DataFrame, min_races: int = 10) -> pd.DataFrame:
    """Combined probabilities for each race from a fit on EARLIER races only.

    rows need a Date column for the order. Races before min_races finished races get
    the market's probability (nothing fitted yet).
    """
    ordered = rows.sort_values(["Date", "RaceId"])
    races = ordered["RaceId"].drop_duplicates().tolist()
    out = []
    for i, race_id in enumerate(races):
        race = ordered[ordered["RaceId"] == race_id].copy()
        past = ordered[ordered["RaceId"].isin(races[:i])]
        if len(_groups(past)) >= min_races:
            fit = fit_benter(past)
            race["p_combined"] = fit.combine(race["p_engine"], race["p_market"])
        else:
            race["p_combined"] = race["p_market"] / race["p_market"].sum()
        out.append(race)
    return pd.concat(out, ignore_index=True) if out else rows.assign(p_combined=np.nan)
