"""How much to stake (on paper): the Kelly criterion and safer variants.

Kelly stakes the fraction of the bankroll that maximises the long-run growth rate,
E[log bankroll]. It assumes the probabilities are RIGHT; with estimated probabilities,
full Kelly over-bets, so the default is a quarter of it (fractional Kelly), with caps.

- kelly_single: one bet: f = (p * odds - 1) / (odds - 1), never below 0
- kelly_multi: several mutually exclusive outcomes (one race winner), the exact
  solution of Smoczynski & Tomkins (2010): bet on the outcomes with the highest
  p * odds while that beats the "reserve rate", stake f_i = p_i - R / odds_i
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StakePolicy:
    """Paper-trading rules. Stakes are fractions of the bankroll."""

    kelly_fraction: float = 0.25   # quarter Kelly
    min_edge: float = 0.02         # p_model - p_market must be at least this
    min_ev: float = 0.05           # expected profit per unit staked
    max_bet: float = 0.02          # at most 2% of the bankroll on one outcome
    max_race: float = 0.05         # at most 5% of the bankroll on one race
    min_odds: float = 1.2
    max_odds: float = 51.0         # very long odds are where the model is least reliable

    def __post_init__(self) -> None:
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be in (0, 1].")
        if not 0 < self.max_bet <= self.max_race <= 1:
            raise ValueError("Need 0 < max_bet <= max_race <= 1.")


def kelly_single(p: float, odds: float) -> float:
    """Full-Kelly fraction for one bet at decimal odds (0 if the bet has no edge)."""
    if not 0 <= p <= 1 or odds <= 1:
        raise ValueError("Need 0 <= p <= 1 and decimal odds > 1.")
    return max(0.0, (p * odds - 1.0) / (odds - 1.0))


def kelly_multi(p: Sequence[float] | np.ndarray,
                odds: Sequence[float] | np.ndarray) -> np.ndarray:
    """Full-Kelly fractions for mutually exclusive outcomes (exactly one wins).

    Smoczynski & Tomkins (2010): sort by expected return p_i * odds_i, add outcomes to
    the betting set S while p_k * odds_k > R(S), where
        R(S) = (1 - sum_{S} p_i) / (1 - sum_{S} 1/odds_i),
    then stake f_i = p_i - R / odds_i on S. Probabilities need not sum to 1 (outcomes
    not listed simply are not bet on).
    """
    probs, prices = np.asarray(p, float), np.asarray(odds, float)
    if probs.shape != prices.shape or np.any(prices <= 1) or np.any(probs < 0):
        raise ValueError("Need matching p >= 0 and decimal odds > 1.")
    order = np.argsort(-probs * prices)
    chosen: list[int] = []
    reserve = 1.0
    for i in order:
        if probs[i] * prices[i] <= reserve:
            break
        trial = chosen + [int(i)]
        sum_p = probs[trial].sum()
        sum_q = (1.0 / prices[trial]).sum()
        if sum_q >= 1.0:  # betting on these would be a guaranteed loss to the margin
            break
        chosen, reserve = trial, (1.0 - sum_p) / (1.0 - sum_q)
    out = np.zeros_like(probs)
    if chosen:
        idx = np.array(chosen)
        out[idx] = np.maximum(0.0, probs[idx] - reserve / prices[idx])
    return out


def growth_rate(fractions: np.ndarray, p: np.ndarray, odds: np.ndarray) -> float:
    """Expected log growth of the bankroll for mutually exclusive outcomes (+ 'none')."""
    f, probs, prices = (np.asarray(x, float) for x in (fractions, p, odds))
    kept = 1.0 - f.sum()
    wins = kept + f * prices
    none = max(0.0, 1.0 - probs.sum())
    value = float(np.sum(probs * np.log(np.maximum(wins, 1e-300))))
    return value + (none * float(np.log(kept)) if none > 0 else 0.0)


def stakes(p_model: np.ndarray, p_market: np.ndarray, odds: np.ndarray,
           policy: StakePolicy, exclusive: bool = True) -> np.ndarray:
    """Paper stakes (fractions of bankroll) after the policy's filters and caps."""
    probs, fair, prices = (np.asarray(x, float) for x in (p_model, p_market, odds))
    allowed = ((probs - fair >= policy.min_edge) & (probs * prices - 1.0 >= policy.min_ev)
               & (prices >= policy.min_odds) & (prices <= policy.max_odds))
    if not allowed.any():
        return np.zeros_like(probs)
    full = np.zeros_like(probs)
    if exclusive:
        full[allowed] = kelly_multi(probs[allowed], prices[allowed])
    else:
        full[allowed] = [kelly_single(pi, oi) for pi, oi in zip(probs[allowed], prices[allowed])]
    out = np.minimum(full * policy.kelly_fraction, policy.max_bet)
    total = out.sum()
    if total > policy.max_race:
        out *= policy.max_race / total
    return out
