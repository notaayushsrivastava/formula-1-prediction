"""Bootstrap: how sure can we be that one driver was really faster?"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core import Laps, Session
from .degradation import FUEL_EFFECT_S_PER_LAP, add_fuel_corrected

Statistic = Callable[..., Any]  # e.g. np.median or np.mean (must accept axis=)


@dataclass(frozen=True)
class PaceDifference:
    """How much slower B was than A in seconds per lap (negative = B faster)."""

    driver_a: str
    driver_b: str
    difference: float
    low: float
    high: float

    @property
    def is_clear(self) -> bool:
        """True if the whole interval is on one side of zero."""
        return self.low > 0 or self.high < 0


def clean_pace_laps(laps: Laps, driver: str,
                    fuel_effect: float = FUEL_EFFECT_S_PER_LAP) -> np.ndarray:
    """Fuel-corrected times of a driver's quick laps, pit laps removed."""
    picked = laps.pick_drivers(driver).pick_wo_box().pick_quicklaps()
    if len(picked) < 2:
        raise ValueError(f"Fewer than 2 clean laps for {driver!r} (is this a loaded race?).")
    return add_fuel_corrected(picked, fuel_effect)["FuelCorrectedSeconds"].to_numpy(float)


def bootstrap_ci(values: Any, stat: Statistic = np.median, n_boot: int = 10_000,
                 ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """(estimate, low, high): the statistic of the data and its bootstrap interval."""
    data = np.asarray(values, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 2:
        raise ValueError("Need at least 2 values to bootstrap.")
    rng = np.random.default_rng(seed)
    boots = stat(rng.choice(data, size=(n_boot, len(data))), axis=1)
    low, high = _percentile_interval(boots, ci)
    return float(stat(data)), low, high


def pace_difference(laps: Laps, driver_a: str, driver_b: str, stat: Statistic = np.median,
                    n_boot: int = 10_000, ci: float = 0.95,
                    fuel_effect: float = FUEL_EFFECT_S_PER_LAP, seed: int = 0) -> PaceDifference:
    """Typical lap of B minus typical lap of A, with a bootstrap interval."""
    times_a = clean_pace_laps(laps, driver_a, fuel_effect)
    times_b = clean_pace_laps(laps, driver_b, fuel_effect)
    rng = np.random.default_rng(seed)
    boot_a = stat(rng.choice(times_a, size=(n_boot, len(times_a))), axis=1)
    boot_b = stat(rng.choice(times_b, size=(n_boot, len(times_b))), axis=1)
    low, high = _percentile_interval(boot_b - boot_a, ci)
    estimate = float(stat(times_b) - stat(times_a))
    return PaceDifference(driver_a, driver_b, estimate, low, high)


def driver_pace_table(session: Session, top_n: int = 10, stat: Statistic = np.median,
                      n_boot: int = 10_000, fuel_effect: float = FUEL_EFFECT_S_PER_LAP,
                      seed: int = 0) -> pd.DataFrame:
    """Typical fuel-corrected pace of the top N finishers with 95% intervals, fastest first."""
    rows = []
    for driver in session.results["Driver"].head(top_n):
        try:
            values = clean_pace_laps(session.laps, driver, fuel_effect)
        except ValueError:
            continue  # too few clean laps, e.g. an early retirement
        pace, low, high = bootstrap_ci(values, stat, n_boot, seed=seed)
        rows.append({"Driver": driver, "Team": session.get_driver(driver)["Team"],
                     "Pace": pace, "Low": low, "High": high, "Laps": len(values)})
    if not rows:
        raise ValueError(f"No driver in {session.name} has enough clean laps.")
    return pd.DataFrame(rows).sort_values("Pace").reset_index(drop=True)


def _percentile_interval(samples: np.ndarray, ci: float) -> tuple[float, float]:
    tail = (1 - ci) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return float(low), float(high)
