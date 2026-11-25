"""Plots for Phase 3: degradation, pace with intervals, title odds."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from ..analysis import (
    FUEL_EFFECT_S_PER_LAP, clean_stint_laps, degradation_table, driver_pace_table,
    simulate_championship,
)
from ..core import Session
from .colours import driver_style, stint_colour, style_for
from .laptimes import format_laptime, laptime_axis, require_race_laps

if TYPE_CHECKING:
    from ..season import Season


def degradation_plot(session: Session, driver: str,
                     fuel_effect: float = FUEL_EFFECT_S_PER_LAP, min_laps: int = 6,
                     ax: Axes | None = None) -> Axes:
    """Fuel-corrected lap time vs tyre age for each stint, with the fitted slope."""
    laps = require_race_laps(session).pick_drivers(driver)
    clean = clean_stint_laps(laps, fuel_effect)
    fits = degradation_table(laps, fuel_effect, min_laps).set_index("Stint")
    if ax is None:
        _, ax = plt.subplots()

    for stint, stint_laps in clean.groupby("Stint"):
        colour = stint_colour(stint)
        ax.scatter(stint_laps["TyreAge"], stint_laps["FuelCorrectedSeconds"],
                   s=25, color=colour, edgecolors="none")
        if stint in fits.index:
            fit = fits.loc[stint]
            x = np.array([stint_laps["TyreAge"].min(), stint_laps["TyreAge"].max()], float)
            compound = fit.get("Compound")
            name = f"Stint {int(stint)}" + (f" {compound}" if isinstance(compound, str) else "")
            ax.plot(x, fit["Intercept"] + fit["DegSecondsPerLap"] * x, color=colour,
                    linewidth=2, label=f"{name}: {fit['DegSecondsPerLap']:+.3f} s/lap")

    laptime_axis(ax)
    ax.set_ylabel(f"Fuel-corrected lap time ({fuel_effect:g} s/lap)")
    ax.set_xlabel("Tyre age (laps into stint)")
    ax.set_title(f"{driver.upper()} tyre degradation · {session.name}")
    if not fits.empty:
        ax.legend(loc="best", frameon=False)
    return ax


def pace_ranking_plot(session: Session, top_n: int = 10, table: pd.DataFrame | None = None,
                      ax: Axes | None = None) -> Axes:
    """Typical fuel-corrected pace with 95% bootstrap intervals, fastest at the top."""
    table = driver_pace_table(session, top_n) if table is None else table
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 1.5 + 0.45 * len(table)))
    y = np.arange(len(table))
    pace = table["Pace"].to_numpy(float)
    err = np.vstack([pace - table["Low"].to_numpy(float), table["High"].to_numpy(float) - pace])
    ax.errorbar(pace, y, xerr=err, fmt="none", ecolor="0.8", capsize=3)
    ax.scatter(pace, y, s=60, zorder=3,
               c=[driver_style(session, d)["color"] for d in table["Driver"]])
    ax.set_yticks(y, table["Driver"])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(format_laptime))
    ax.set_xlabel("Typical fuel-corrected lap (median, 95% interval)")
    ax.set_title(f"Race pace · {session.name}")
    return ax


def title_odds_plot(season: Season, odds: pd.DataFrame | None = None, top_n: int = 8,
                    ax: Axes | None = None) -> Axes:
    """Simulated title chances of every driver who still has one."""
    odds = simulate_championship(season) if odds is None else odds
    shown = odds[odds["TitleChance"] > 0].head(top_n)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 1.5 + 0.5 * len(shown)))
    y = np.arange(len(shown))
    chance = shown["TitleChance"].to_numpy(float)
    colours = [style_for(season.results, d)["color"] for d in shown["Driver"]]
    bars = ax.barh(y, 100 * chance, color=colours)
    ax.bar_label(bars, labels=[f"{c:.1%}" for c in chance], padding=3)
    ax.set_yticks(y, shown["Driver"])
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Title chance (%)")
    ax.set_title(f"{season.year} simulated title odds")
    return ax
