"""Odds maths: formats, the bookmaker's margin, and removing it ("de-vigging").

A bookmaker's prices imply probabilities that add up to MORE than 1; the excess is the
overround (the margin, or "vig"). To compare the market with the engine we need the
market's fair probabilities, which sum to exactly 1. There is no single right way to
remove the margin, so four standard methods are provided:

- multiplicative: scale every implied probability by the same factor (simplest)
- additive: subtract the same amount from every implied probability
- power: raise implied probabilities to a power k chosen so they sum to 1; shrinks
  longshots more than favourites (the favourite-longshot bias)
- Shin (1993): assumes a share z of money comes from insiders; also corrects the
  favourite-longshot bias, with a clear economic story

All functions are pure and work on plain floats / numpy arrays.
"""
from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import numpy as np
from scipy import optimize

METHODS = ("multiplicative", "additive", "power", "shin")


# ---------- formats ----------
def decimal_from_fractional(text: str) -> float:
    """'5/2' -> 3.5 (stake back included). 'evens' / 'EVS' -> 2.0."""
    cleaned = text.strip().lower()
    if cleaned in ("evens", "evs", "even"):
        return 2.0
    return float(Fraction(cleaned)) + 1.0


def decimal_from_american(value: float) -> float:
    """+250 -> 3.5; -200 -> 1.5."""
    if value == 0 or -100 < value < 100:
        raise ValueError(f"American odds must be >= +100 or <= -100, got {value}.")
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / -value)


def american_from_decimal(odds: float) -> float:
    """3.5 -> +250; 1.5 -> -200."""
    _check_decimal(odds)
    return (odds - 1.0) * 100.0 if odds >= 2.0 else -100.0 / (odds - 1.0)


def implied_probability(odds: float | Sequence[float] | np.ndarray) -> np.ndarray:
    """Decimal odds -> implied probability 1 / odds (margin still included)."""
    values = np.asarray(odds, float)
    if np.any(values <= 1.0):
        raise ValueError("Decimal odds must be greater than 1.")
    return 1.0 / values


def decimal_from_probability(p: float | np.ndarray) -> np.ndarray:
    """Fair decimal odds of a probability (0 < p <= 1)."""
    values = np.asarray(p, float)
    if np.any((values <= 0) | (values > 1)):
        raise ValueError("Probabilities must be in (0, 1].")
    return 1.0 / values


def to_decimal(price: float | str, kind: str) -> float:
    """A price in any format -> decimal odds. kind: decimal, fractional, american,
    probability (a prediction-market price in [0, 1])."""
    if kind == "decimal":
        odds = float(price)
    elif kind == "fractional":
        odds = decimal_from_fractional(str(price))
    elif kind == "american":
        odds = decimal_from_american(float(price))
    elif kind == "probability":
        odds = float(decimal_from_probability(float(price)))
    else:
        raise ValueError(f"Unknown price kind {kind!r}: decimal, fractional, american, "
                         "probability.")
    _check_decimal(odds)
    return odds


def _check_decimal(odds: float) -> None:
    if not np.isfinite(odds) or odds <= 1.0:
        raise ValueError(f"Decimal odds must be a number above 1, got {odds}.")


# ---------- the margin ----------
def overround(odds: Sequence[float] | np.ndarray, places: int = 1) -> float:
    """Sum of implied probabilities divided by the number of places paid, minus 1.

    places = 1 for a winner market, 3 for a podium ('top 3') market. 0.05 = 5% margin.
    """
    return float(implied_probability(odds).sum() / places - 1.0)


def devig(odds: Sequence[float] | np.ndarray, method: str = "shin",
          places: int = 1) -> np.ndarray:
    """Fair probabilities (summing to `places`) from one market's decimal odds."""
    q = implied_probability(odds)
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}: {', '.join(METHODS)}.")
    if len(q) < 2:
        raise ValueError("De-vigging needs at least two outcomes.")
    if method == "multiplicative" or places != 1:
        # power and Shin are defined for single-winner markets; for top-N markets the
        # proportional method is the standard choice
        return _clip_sum(q * places / q.sum(), places)
    if method == "additive":
        return _clip_sum(q - (q.sum() - 1.0) / len(q), 1.0)
    if method == "power":
        return _power(q)
    return _shin(q)


def _power(q: np.ndarray) -> np.ndarray:
    """p_i = q_i ** k with k chosen so the p sum to 1 (k > 1 when there is a margin)."""
    if abs(q.sum() - 1.0) < 1e-12:
        return q.copy()

    def gap(k: float) -> float:
        return float(np.sum(q ** k) - 1.0)

    k = optimize.brentq(gap, 1e-6, 100.0)
    return q ** k / np.sum(q ** k)


def shin_z(q: np.ndarray) -> float:
    """Shin's insider share z for implied probabilities q (0 when there is no margin)."""
    total = q.sum()
    if total <= 1.0 + 1e-12:
        return 0.0

    def gap(z: float) -> float:
        return float(np.sum(_shin_p(q, z, total)) - 1.0)

    return float(optimize.brentq(gap, 0.0, 0.999))


def _shin_p(q: np.ndarray, z: float, total: float) -> np.ndarray:
    return (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / total) - z) / (2 * (1 - z))


def _shin(q: np.ndarray) -> np.ndarray:
    z = shin_z(q)
    p = _shin_p(q, z, q.sum())
    return p / p.sum()


def _clip_sum(p: np.ndarray, total: float) -> np.ndarray:
    """Keep probabilities in (0, 1] after the additive method and renormalise."""
    p = np.clip(p, 1e-9, 1.0)
    return p * total / p.sum()


# ---------- value ----------
def expected_value(p: float | np.ndarray, odds: float | np.ndarray) -> np.ndarray:
    """Expected profit per 1 unit staked: p * (odds - 1) - (1 - p) = p * odds - 1."""
    return np.asarray(p, float) * np.asarray(odds, float) - 1.0


def edge(p_model: float | np.ndarray, p_market: float | np.ndarray) -> np.ndarray:
    """Model probability minus fair market probability (positive = model likes it more)."""
    return np.asarray(p_model, float) - np.asarray(p_market, float)
