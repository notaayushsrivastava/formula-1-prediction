"""Tyre degradation: how much slower each lap gets as a stint goes on."""
from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats

from ..core import Laps

FUEL_EFFECT_S_PER_LAP = 0.05  # rough ballpark; an assumption you should vary
DEG_COLUMNS = ["Driver", "Team", "Stint", "Compound", "StartLap", "EndLap", "Laps",
               "DegSecondsPerLap", "DegLow", "DegHigh", "Intercept"]


def add_fuel_corrected(laps: Laps, fuel_effect: float = FUEL_EFFECT_S_PER_LAP) -> Laps:
    """Add 'FuelCorrectedSeconds': lap time as if every lap carried lap-1 fuel.

    Later laps are lighter and so faster; adding back fuel_effect seconds per lap
    already driven removes that trend. What's left is mostly tyres (and traffic).
    """
    out = laps.copy()
    out["FuelCorrectedSeconds"] = out["LapTimeSeconds"] + fuel_effect * (out["LapNumber"] - 1)
    return out


def add_tyre_age(laps: Laps) -> Laps:
    """Add 'TyreAge': laps on the current set of tyres.

    With Live Timing this is F1's TyreLife (used sets start above 1). Without it,
    laps are counted from the stint start, so call this BEFORE filtering laps out.
    """
    out = laps.copy()
    start = out.groupby(["DriverId", "Stint"])["LapNumber"].transform("min")
    out["TyreAge"] = out["LapNumber"] - start + 1
    if "TyreLife" in out.columns:
        out["TyreAge"] = out["TyreLife"].fillna(out["TyreAge"])
    return out


def clean_stint_laps(laps: Laps, fuel_effect: float = FUEL_EFFECT_S_PER_LAP) -> Laps:
    """Fuel-corrected racing laps with tyre age and no pit laps.

    Exact green-flag laps when Live Timing is loaded, otherwise the 107% rule.
    """
    aged = add_fuel_corrected(add_tyre_age(laps), fuel_effect).pick_wo_box()
    return aged.pick_green() if "TrackStatus" in aged.columns else aged.pick_quicklaps()


def degradation_table(laps: Laps, fuel_effect: float = FUEL_EFFECT_S_PER_LAP,
                      min_laps: int = 6) -> pd.DataFrame:
    """One row per driver stint: degradation in s/lap with a 95% range (Theil-Sen).

    Stints with fewer than min_laps clean laps are skipped (too few for a trustworthy slope).
    """
    clean = clean_stint_laps(laps, fuel_effect)
    rows = [_fit_stint(stint) for _, stint in clean.groupby(["Driver", "Stint"])
            if len(stint) >= min_laps]
    return pd.DataFrame(rows, columns=DEG_COLUMNS)


def _fit_stint(stint: pd.DataFrame) -> dict[str, Any]:
    slope, intercept, low, high = stats.theilslopes(
        stint["FuelCorrectedSeconds"].to_numpy(float),
        stint["TyreAge"].to_numpy(float), alpha=0.95)
    return {
        "Driver": stint["Driver"].iloc[0],
        "Team": stint["Team"].iloc[0],
        "Stint": int(stint["Stint"].iloc[0]),
        "Compound": stint["Compound"].iloc[0] if "Compound" in stint.columns else None,
        "StartLap": int(stint["LapNumber"].min()),
        "EndLap": int(stint["LapNumber"].max()),
        "Laps": len(stint),
        "DegSecondsPerLap": float(slope),
        "DegLow": float(low),
        "DegHigh": float(high),
        "Intercept": float(intercept),
    }
