"""Scoring probabilistic race predictions.

Losses (log loss, Brier, RPS, ECE): lower is better. Spearman, hit rate, overlap:
higher is better. Binary log losses are averaged over DRIVER ROWS, like the
0.075 (win) / 0.191 (podium) benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .models.base import position_matrix

EPS = 1e-6  # probabilities are clipped to [EPS, 1 - EPS] before taking logs
BENCHMARKS = {"win_logloss": 0.075, "podium_logloss": 0.191}  # best open model found
BINARY = {"win": ("P_win", "Win"), "podium": ("P_podium", "Podium"),
          "points": ("P_points", "PointsFinish"), "dnf": ("P_dnf", "DNF")}
METRIC_NAMES = [
    "win_logloss", "podium_logloss", "points_logloss", "dnf_logloss", "winner_logloss",
    "win_brier", "podium_brier", "points_brier", "dnf_brier", "rps", "spearman",
    "winner_hit_rate", "podium_overlap", "win_ece", "podium_ece", "n_races",
]


# ---------- binary ----------
def row_log_loss(p: object, y: object) -> np.ndarray:
    """-[y log p + (1 - y) log(1 - p)] for every row."""
    prob = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    label = np.asarray(y, float)
    return -(label * np.log(prob) + (1 - label) * np.log(1 - prob))


def binary_log_loss(p: object, y: object) -> float:
    """Mean binary log loss over rows."""
    return float(np.mean(row_log_loss(p, y)))


def brier(p: object, y: object) -> float:
    """Mean squared difference between probability and outcome (0 or 1)."""
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def winner_log_loss(preds: pd.DataFrame) -> float:
    """Race-level: mean over races of -log P_win of the driver who actually won."""
    winners = preds[preds["Win"] == 1]
    if winners.empty:
        return float("nan")
    return float(-np.mean(np.log(np.clip(winners["P_win"].to_numpy(float), EPS, 1.0))))


def per_race_log_loss(preds: pd.DataFrame, kind: str = "win") -> pd.Series:
    """Mean binary log loss within each race (index = RaceId), for paired bootstraps."""
    prob, label = BINARY[kind]
    losses = pd.Series(row_log_loss(preds[prob], preds[label]), index=preds.index)
    return losses.groupby(preds["RaceId"]).mean()


def race_loss_table(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per race (Phase 15 experiment log): RaceId, Date, Year, n_rows and the mean
    win / podium / points / dnf log loss within the race, plus 'winner' = -log P_win of
    the actual winner. Sorted by date. Row-weighted means (by n_rows) reproduce the
    driver-row averages of summarise()."""
    races = preds.groupby("RaceId", sort=False)
    table = pd.DataFrame({"n_rows": races.size()})
    for kind in BINARY:
        table[kind] = per_race_log_loss(preds, kind)
    winners = preds[preds["Win"] == 1].drop_duplicates("RaceId").set_index("RaceId")
    table["winner"] = -np.log(np.clip(winners["P_win"].astype(float), EPS, 1.0))
    context = [c for c in ("Date", "Year") if c in preds.columns]
    if context:
        table = table.join(races[context].first())
    table = table.reset_index()
    order = ["Date", "RaceId"] if "Date" in table else ["RaceId"]
    return table.sort_values(order).reset_index(drop=True)


def weighted_loss(table: pd.DataFrame, kind: str = "win") -> float:
    """Driver-row average of a race_loss_table column (each race weighted by n_rows)."""
    weights = table["n_rows"].to_numpy(float)
    return float(np.sum(weights * table[kind].to_numpy(float)) / np.sum(weights))


# ---------- whole order ----------
def ranked_probability_score(preds: pd.DataFrame) -> float:
    """Mean RPS per driver: squared gaps between predicted and actual cumulative
    position distributions, divided by (positions - 1). Needs the P_pos_k columns."""
    if not any(c.startswith("P_pos_") for c in preds.columns):
        return float("nan")
    scores: list[float] = []
    for _, race in preds.groupby("RaceId"):
        n = len(race)
        probs = position_matrix(race)[:, :n]
        observed = np.zeros_like(probs)
        observed[np.arange(n), race["FinishOrder"].to_numpy(int) - 1] = 1.0
        gaps = np.cumsum(probs, axis=1) - np.cumsum(observed, axis=1)
        scores.extend(np.sum(gaps**2, axis=1) / max(n - 1, 1))
    return float(np.mean(scores))


def mean_spearman(preds: pd.DataFrame) -> float:
    """Mean over races of Spearman rank correlation of ExpectedPosition vs FinishOrder."""
    values = []
    for _, race in preds.groupby("RaceId"):
        if len(race) >= 3 and race["ExpectedPosition"].nunique() > 1:
            rho = stats.spearmanr(race["ExpectedPosition"], race["FinishOrder"])[0]
            if not np.isnan(rho):
                values.append(rho)
    return float(np.mean(values)) if values else float("nan")


def winner_hit_rate(preds: pd.DataFrame) -> float:
    """Share of races where the favourite (highest P_win) actually won."""
    hits = [race.loc[race["P_win"].idxmax(), "Win"] == 1
            for _, race in preds.groupby("RaceId")]
    return float(np.mean(hits))


def podium_overlap(preds: pd.DataFrame) -> float:
    """Mean share of the real podium found among the 3 highest P_podium drivers."""
    shares = []
    for _, race in preds.groupby("RaceId"):
        real = set(race.loc[race["Podium"] == 1, "DriverId"])
        if real:
            picked = set(race.nlargest(3, "P_podium")["DriverId"])
            shares.append(len(real & picked) / len(real))
    return float(np.mean(shares)) if shares else float("nan")


# ---------- calibration ----------
def calibration_table(p: object, y: object, bins: int = 10) -> pd.DataFrame:
    """Equal-width bins: mean predicted vs observed frequency, and how many rows."""
    prob = np.clip(np.asarray(p, float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.digitize(prob, edges[1:-1])  # 0 .. bins-1
    table = (pd.DataFrame({"Bin": which, "p": prob, "y": np.asarray(y, float)})
             .groupby("Bin")
             .agg(Predicted=("p", "mean"), Observed=("y", "mean"), Count=("y", "size"))
             .reset_index())
    table["Low"] = edges[table["Bin"]]
    table["High"] = edges[table["Bin"] + 1]
    return table[["Bin", "Low", "High", "Predicted", "Observed", "Count"]]


def expected_calibration_error(p: object, y: object, bins: int = 10) -> float:
    """Count-weighted mean gap between predicted and observed frequency over bins."""
    table = calibration_table(p, y, bins)
    weights = table["Count"] / table["Count"].sum()
    return float(np.sum(weights * (table["Predicted"] - table["Observed"]).abs()))


# ---------- comparing two models ----------
@dataclass(frozen=True)
class BootstrapResult:
    """Mean of (a - b) per race and its bootstrap interval (negative = a is better)."""

    mean: float
    low: float
    high: float
    n_races: int

    @property
    def significant(self) -> bool:
        """True if the whole interval is on one side of zero."""
        return self.low > 0 or self.high < 0


def paired_bootstrap(a: pd.Series, b: pd.Series, n_boot: int = 10_000, ci: float = 0.95,
                     seed: int = 0) -> BootstrapResult:
    """Paired bootstrap over races: resample races, keeping each race's (a, b) together.

    a and b are per-race metrics indexed by RaceId (e.g. per_race_log_loss).
    """
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if both.empty:
        raise ValueError("The two models have no race in common.")
    diff = (both["a"] - both["b"]).to_numpy(float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    tail = (1 - ci) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return BootstrapResult(float(diff.mean()), float(low), float(high), len(diff))


# ---------- everything at once ----------
def summarise(preds: pd.DataFrame) -> dict[str, float]:
    """Every metric (METRIC_NAMES) for one model's predictions with labels attached."""
    out: dict[str, float] = {}
    for kind, (prob, label) in BINARY.items():
        out[f"{kind}_logloss"] = binary_log_loss(preds[prob], preds[label])
        out[f"{kind}_brier"] = brier(preds[prob], preds[label])
    out["winner_logloss"] = winner_log_loss(preds)
    out["rps"] = ranked_probability_score(preds)
    out["spearman"] = mean_spearman(preds)
    out["winner_hit_rate"] = winner_hit_rate(preds)
    out["podium_overlap"] = podium_overlap(preds)
    out["win_ece"] = expected_calibration_error(preds["P_win"], preds["Win"])
    out["podium_ece"] = expected_calibration_error(preds["P_podium"], preds["Podium"])
    out["n_races"] = float(preds["RaceId"].nunique())
    return {name: out[name] for name in METRIC_NAMES}
