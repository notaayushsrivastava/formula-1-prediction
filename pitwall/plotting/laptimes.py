"""Lap-time plots: one driver's scatter and the field's distribution."""
from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from ..core import Laps, Session
from .colours import compound_colour, driver_style, stint_colour


def format_laptime(seconds: float, _pos: object = None) -> str:
    """83.456 -> '1:23.456'. Also works as a matplotlib tick formatter."""
    if pd.isna(seconds):
        return ""
    minutes, secs = divmod(float(seconds), 60)
    return f"{int(minutes)}:{secs:06.3f}"


def laptime_axis(ax: Axes) -> None:
    """Format the y-axis as m:ss.sss with faster laps at the top."""
    ax.yaxis.set_major_formatter(FuncFormatter(format_laptime))
    ax.invert_yaxis()
    ax.set_ylabel("Lap time")


def require_race_laps(session: Session) -> Laps:
    """The session's laps, or a clear error for Q/S sessions or unloaded sessions."""
    if session.laps.empty:
        raise ValueError(f"{session.name} has no lap data. Lap plots need a loaded race "
                         "session: get_session(year, event).load()")
    return session.laps


def driver_laptimes_scatter(session: Session, driver: str,
                            ax: Axes | None = None, quicklaps: bool = True) -> Axes:
    """Scatter of one driver's lap times (FastF1 gallery plot).

    Coloured by tyre compound when Live Timing is loaded, otherwise by stint.
    quicklaps=True drops pit laps and SC/VSC/red-flag laps (exactly with Live
    Timing, via the 107% rule without it). Dotted lines mark pit stops.
    """
    all_laps = require_race_laps(session).pick_drivers(driver)
    if all_laps.empty:
        known = ", ".join(sorted(session.laps["Driver"].unique()))
        raise ValueError(f"No laps for {driver!r} in {session.name}. Drivers: {known}")
    laps = _racing_laps(all_laps) if quicklaps else all_laps
    by_compound = "Compound" in laps.columns
    if ax is None:
        _, ax = plt.subplots()

    for value, group in laps.groupby("Compound" if by_compound else "Stint"):
        colour = compound_colour(value) if by_compound else stint_colour(value)
        label = str(value).title() if by_compound else f"Stint {value}"
        ax.scatter(group["LapNumber"], group["LapTimeSeconds"], s=30,
                   color=colour, edgecolors="none", label=label)
    for lap in all_laps.loc[all_laps["PitInLap"], "LapNumber"]:
        ax.axvline(lap + 0.5, color="white", alpha=0.4, linestyle=":")

    laptime_axis(ax)
    ax.set_xlabel("Lap")
    ax.set_title(f"{all_laps['Driver'].iloc[0]} lap times · {session.name}")
    ax.legend(loc="best", frameon=False)
    return ax


def _racing_laps(laps: Laps) -> Laps:
    """No pit laps; green-flag laps with Live Timing, otherwise the 107% rule."""
    clean = laps.pick_wo_box()
    return clean.pick_green() if "TrackStatus" in clean.columns else clean.pick_quicklaps()


def laptime_distribution(session: Session, top_n: int = 10, ax: Axes | None = None) -> Axes:
    """Violin plot of race pace for the top N finishers (FastF1 gallery plot).

    Uses quick laps without pit laps. Violins are in team colours; dots show the stint.
    """
    laps, drivers = _top_driver_quicklaps(session, top_n)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    data = [laps.loc[laps["Driver"] == d, "LapTimeSeconds"].to_numpy() for d in drivers]
    violins = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, drv in zip(violins["bodies"], drivers):
        body.set_facecolor(driver_style(session, drv)["color"])
        body.set_alpha(0.5)
    _stint_dots(ax, laps, drivers)

    ax.set_xticks(range(1, len(drivers) + 1), drivers)
    laptime_axis(ax)
    ax.set_title(f"Race pace of the top {len(drivers)} · {session.name}")
    ax.legend(handles=_stint_handles(laps["Stint"].unique()), loc="best", frameon=False)
    return ax


def _top_driver_quicklaps(session: Session, top_n: int) -> tuple[Laps, list[str]]:
    """Quick laps of the top N finishers; drivers need 2+ laps (a violin needs a spread)."""
    finishers = list(session.results["Driver"].head(top_n))
    laps = require_race_laps(session).pick_drivers(finishers).pick_wo_box().pick_quicklaps()
    counts = laps["Driver"].value_counts()
    drivers = [d for d in finishers if counts.get(d, 0) >= 2]
    if not drivers:
        raise ValueError(f"Not enough quick laps for the top {top_n} in {session.name}.")
    return laps, drivers


def _stint_dots(ax: Axes, laps: Laps, drivers: list[str]) -> None:
    """Jittered dots on top of each violin, coloured by stint."""
    rng = np.random.default_rng(0)  # fixed seed: the same jitter every run
    for x, drv in enumerate(drivers, start=1):
        drv_laps = laps[laps["Driver"] == drv]
        jitter = rng.uniform(-0.15, 0.15, len(drv_laps))
        colours = [stint_colour(s) for s in drv_laps["Stint"]]
        ax.scatter(x + jitter, drv_laps["LapTimeSeconds"].to_numpy(),
                   s=10, c=colours, edgecolors="none")


def _stint_handles(stints: Iterable[int]) -> list[Line2D]:
    """Legend entries for stints (the dots themselves have no labels)."""
    return [Line2D([], [], marker="o", linestyle="", color=stint_colour(s), label=f"Stint {s}")
            for s in sorted(int(s) for s in stints)]
