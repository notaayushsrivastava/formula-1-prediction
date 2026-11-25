"""Phase 15: statistics for comparing models without fooling ourselves.

Every function works on PER-RACE numbers (one loss per race, lower = better), because
races, not driver rows, are the independent-ish units. The maths is in docs/methods.md.

- paired_bootstrap (metrics.py, reused): resample races, keep each race's pair together.
- block_bootstrap: resample RUNS of consecutive races (default 3), in case neighbouring
  races are alike (same car upgrades, same part of the season).
- diebold_mariano: a t-test on the per-race loss differences with the Harvey-Leybourne-
  Newbold small-sample correction. A second opinion to the bootstrap.
- deflated_score: "the best of 30 random tries looks good by luck". The improvement's
  mean / standard deviation (a Sharpe ratio per race) is compared with the score the
  luckiest of n_trials useless configurations would reach.
- probability_of_backtest_overfitting (PBO): how often the configuration that looks best
  on half the races is below the median on the other half.
- benjamini_hochberg: control the share of false discoveries over many gates.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import BootstrapResult, paired_bootstrap

EULER_GAMMA = 0.5772156649015329
DEFAULT_BLOCK = 3
PBO_BLOCKS = 8

__all__ = ["BootstrapResult", "DMResult", "DeflatedResult", "PBOResult", "aligned_diff",
           "benjamini_hochberg", "block_bootstrap", "bootstrap_p_value", "deflated_score",
           "diebold_mariano", "expected_max_sharpe", "paired_bootstrap",
           "probabilistic_sharpe", "probability_of_backtest_overfitting"]


def aligned_diff(a: pd.Series, b: pd.Series, order: pd.Series | None = None) -> np.ndarray:
    """a - b for the races both have (index = RaceId), in time order.

    order (optional): a Series index=RaceId with something sortable (e.g. the race Date);
    without it races are sorted by RaceId (f1db ids grow with time).
    """
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if both.empty:
        raise ValueError("The two models have no race in common.")
    if order is not None:
        both = both.join(order.rename("_order"), how="left").sort_values("_order")
    else:
        both = both.sort_index()
    return (both["a"] - both["b"]).to_numpy(float)


# ---------- bootstraps ----------
def block_bootstrap(a: pd.Series, b: pd.Series, block: int = DEFAULT_BLOCK,
                    n_boot: int = 10_000, ci: float = 0.95, seed: int = 0,
                    order: pd.Series | None = None) -> BootstrapResult:
    """Circular moving-block bootstrap of mean(a - b) over races in time order.

    Each resample glues together random runs of `block` consecutive races (wrapping
    round at the end), so dependence between neighbouring races is kept. block=1 is the
    ordinary paired bootstrap.
    """
    if block < 1:
        raise ValueError("block must be at least 1.")
    diff = aligned_diff(a, b, order)
    means = _block_means(diff, block, n_boot, seed)
    tail = (1 - ci) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return BootstrapResult(float(diff.mean()), float(low), float(high), len(diff))


def bootstrap_p_value(a: pd.Series, b: pd.Series, block: int = 1, n_boot: int = 10_000,
                      seed: int = 0, order: pd.Series | None = None) -> float:
    """One-sided p-value for "a is better (lower) than b": the share of bootstrap means
    of (a - b) that are >= 0 (plus one, to never report exactly 0). A 95% interval that
    lies entirely below 0 corresponds to p < 0.025."""
    diff = aligned_diff(a, b, order)
    means = _block_means(diff, block, n_boot, seed)
    return float((np.sum(means >= 0) + 1) / (n_boot + 1))


def _block_means(diff: np.ndarray, block: int, n_boot: int, seed: int) -> np.ndarray:
    n = len(diff)
    block = min(block, n)
    rng = np.random.default_rng(seed)
    blocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return diff[idx % n].mean(axis=1)


# ---------- Diebold-Mariano ----------
@dataclass(frozen=True)
class DMResult:
    """Diebold-Mariano test of mean(a - b) = 0 (HLN-corrected, Student t with n-1 df)."""

    statistic: float
    p_value: float     # two-sided
    p_better: float    # one-sided: a has LOWER loss than b
    mean: float
    n: int
    lag: int


def diebold_mariano(diff: Sequence[float] | np.ndarray | pd.Series, lag: int = 1) -> DMResult:
    """DM test on per-race loss differences d = loss_a - loss_b.

    lag = forecast horizon h: autocovariances up to h-1 enter the variance (h = 1: none,
    each race is one forecast). Harvey-Leybourne-Newbold (1997) multiply the statistic by
    sqrt((n + 1 - 2h + h(h-1)/n) / n) and compare it with Student's t (n - 1 df).
    """
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        raise ValueError("The Diebold-Mariano test needs at least 2 races.")
    if lag < 1 or lag >= n:
        raise ValueError("lag must be between 1 and n - 1.")
    mean = float(d.mean())
    centred = d - mean
    gamma = [float(np.dot(centred[k:], centred[:n - k]) / n) for k in range(lag)]
    variance = (gamma[0] + 2 * sum(gamma[1:])) / n
    if variance <= 1e-300:  # identical models (or a constant difference)
        stat = 0.0 if abs(mean) < 1e-15 else math.copysign(math.inf, mean)
    else:
        correction = math.sqrt((n + 1 - 2 * lag + lag * (lag - 1) / n) / n)
        stat = mean / math.sqrt(variance) * correction
    p_two = float(2 * stats.t.sf(abs(stat), n - 1))
    p_better = float(stats.t.cdf(stat, n - 1))
    return DMResult(float(stat), min(p_two, 1.0), p_better, mean, n, lag)


# ---------- the deflated score ----------
def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected best score among n_trials configurations that are all useless (true score
    0) and whose estimated scores have standard deviation sr_std (Bailey & Lopez de Prado):
        sr_std * ((1 - g) * z(1 - 1/N) + g * z(1 - 1/(N e)))     g = Euler's constant.
    One trial: 0 (no luck to correct for)."""
    if n_trials <= 1:
        return 0.0
    z = stats.norm.ppf
    return float(sr_std * ((1 - EULER_GAMMA) * z(1 - 1 / n_trials)
                           + EULER_GAMMA * z(1 - 1 / (n_trials * math.e))))


def probabilistic_sharpe(sr: float, n: int, skew: float, kurtosis: float,
                         benchmark: float = 0.0) -> float:
    """P(true score > benchmark) given an estimated per-race score sr from n races
    (non-normal correction with skewness and ordinary, non-excess, kurtosis)."""
    if n < 2:
        return float("nan")
    spread = 1 - skew * sr + (kurtosis - 1) / 4 * sr**2
    spread = max(spread, 1e-12)
    return float(stats.norm.cdf((sr - benchmark) * math.sqrt(n - 1) / math.sqrt(spread)))


@dataclass(frozen=True)
class DeflatedResult:
    """The deflated score of a per-race improvement (old minus new loss; > 0 = better)."""

    score: float         # mean / sd of the per-race improvement
    n_races: int
    n_trials: int
    luck: float          # the score the best of n_trials useless tries would reach
    probability: float   # deflated probability that the improvement is real
    p_value: float       # 1 - probability
    skew: float
    kurtosis: float

    def explain(self) -> str:
        """The result in plain words."""
        return (f"per-race score {self.score:+.3f} over {self.n_races} races; the best of "
                f"{self.n_trials} useless tries would reach about {self.luck:+.3f} by luck; "
                f"deflated p = {self.p_value:.3f}")


def deflated_score(improvement: Sequence[float] | np.ndarray, n_trials: int,
                   sr_std: float | None = None) -> DeflatedResult:
    """Adapted Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    improvement: per-race (incumbent loss - candidate loss); positive = candidate better.
    n_trials: distinct configurations tried in this family at this moment (from the log).
    sr_std: spread of the trials' scores. Default sqrt(1 / (n - 1)), the spread of a
    useless configuration's estimated score: conservative when trials are correlated.
    """
    x = np.asarray(improvement, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        raise ValueError("The deflated score needs at least 3 races.")
    trials = max(int(n_trials), 1)
    sd = float(x.std(ddof=1))
    mean = float(x.mean())
    if sd <= 1e-15:
        prob = 1.0 if mean > 0 else 0.0
        return DeflatedResult(math.copysign(math.inf, mean) if mean else 0.0, n, trials,
                              0.0, prob, 1.0 - prob, 0.0, 3.0)
    score = mean / sd
    skew = float(stats.skew(x))
    kurt = float(stats.kurtosis(x, fisher=False))
    spread = math.sqrt(1.0 / (n - 1)) if sr_std is None else float(sr_std)
    luck = expected_max_sharpe(trials, spread)
    prob = probabilistic_sharpe(score, n, skew, kurt, luck)
    return DeflatedResult(score, n, trials, luck, prob, 1.0 - prob, skew, kurt)


# ---------- PBO ----------
@dataclass(frozen=True)
class PBOResult:
    """Probability of Backtest Overfitting (combinatorially symmetric CV)."""

    pbo: float
    logits: np.ndarray        # one per split of the races into two halves
    n_combinations: int
    n_configs: int
    n_races: int
    n_blocks: int

    def explain(self) -> str:
        """The result in plain words."""
        return (f"PBO = {self.pbo:.2f}: in {self.pbo:.0%} of {self.n_combinations} ways of "
                f"splitting {self.n_races} races in two, the configuration that looked best "
                f"on one half was below the median of {self.n_configs} on the other half "
                "(0.5 = pure luck, near 0 = a real winner).")


def probability_of_backtest_overfitting(losses: pd.DataFrame,
                                        n_blocks: int = PBO_BLOCKS) -> PBOResult:
    """PBO (Bailey, Borwein, Lopez de Prado & Zhu 2017) from a races x configurations
    table of per-race LOSSES (lower = better), rows in time order.

    The races are cut into n_blocks (even) consecutive blocks. For every way of choosing
    half of the blocks as 'in-sample': pick the configuration with the lowest in-sample
    mean loss, find its rank on the other half, omega = rank / (N + 1) with rank 1 =
    worst, and logit = log(omega / (1 - omega)). PBO = share of logits below 0 (a tie at
    exactly the median counts half).
    """
    table = losses.dropna()
    n_races, n_configs = table.shape
    if n_configs < 2:
        raise ValueError("PBO needs at least 2 configurations.")
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even number >= 2.")
    if n_races < n_blocks:
        raise ValueError(f"PBO needs at least {n_blocks} races (got {n_races}).")
    values = table.to_numpy(float)
    blocks = np.array_split(np.arange(n_races), n_blocks)
    logits = []
    for chosen in combinations(range(n_blocks), n_blocks // 2):
        inside = np.concatenate([blocks[i] for i in chosen])
        outside = np.setdiff1d(np.arange(n_races), inside)
        best = int(np.argmin(values[inside].mean(axis=0)))
        # rank 1 = worst out-of-sample (highest loss), N = best
        ranks = stats.rankdata(-values[outside].mean(axis=0))
        omega = ranks[best] / (n_configs + 1)
        logits.append(math.log(omega / (1 - omega)))
    arr = np.array(logits)
    pbo = float(np.mean(np.where(np.isclose(arr, 0.0), 0.5, arr < 0)))
    return PBOResult(pbo, arr, len(arr), n_configs, n_races, n_blocks)


# ---------- many gates ----------
def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg: which of many tests can be called discoveries while keeping
    the expected share of false discoveries at most q. Returns, in the input order,
    p, adjusted p (the 'BH q-value') and rejected (True = discovery)."""
    p = np.asarray(p_values, float)
    m = len(p)
    if m == 0:
        return pd.DataFrame(columns=["p", "adjusted", "rejected"])
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    passed = p[order] <= q * np.arange(1, m + 1) / m
    cutoff = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    rejected = np.zeros(m, dtype=bool)
    rejected[order[:cutoff]] = True
    return pd.DataFrame({"p": p, "adjusted": adjusted, "rejected": rejected})
