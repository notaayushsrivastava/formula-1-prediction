"""Whole-race plots: position changes and team pace."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from ..core import Session
from .colours import driver_style, team_colour
from .laptimes import laptime_axis, require_race_laps


def position_changes(session: Session, ax: Axes | None = None) -> Axes:
    """Every driver's position lap by lap, starting from the grid (FastF1 gallery plot)."""
    positions = _positions_with_grid(session)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    for driver in session.results["Driver"]:  # finishing order = legend order
        drv = positions[positions["Driver"] == driver]
        if drv.empty:
            continue
        style = driver_style(session, driver)
        ax.plot(drv["LapNumber"].to_numpy(), drv["Position"].to_numpy(),
                color=style["color"], linestyle=style["linestyle"], linewidth=1.6,
                label=driver)

    n = len(session.results)
    ax.set_yticks(range(1, n + 1))
    ax.set_ylim(n + 0.5, 0.5)  # P1 at the top
    ax.set_xlabel("Lap (0 = starting grid)")
    ax.set_ylabel("Position")
    ax.set_title(f"Position changes · {session.name}")
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize="small", frameon=False)
    return ax


def team_pace_boxplot(session: Session, ax: Axes | None = None) -> Axes:
    """Race pace per team: box plot of quick laps, fastest median on the left (FastF1 gallery)."""
    laps = require_race_laps(session).pick_wo_box().pick_quicklaps()
    if laps.empty:
        raise ValueError(f"No quick laps in {session.name}.")
    order = laps.groupby("Team")["LapTimeSeconds"].median().sort_values().index.tolist()
    team_ids = laps.drop_duplicates("Team").set_index("Team")["TeamId"]
    data = [laps.loc[laps["Team"] == t, "LapTimeSeconds"].to_numpy() for t in order]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    boxes = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                       medianprops={"color": "white", "linewidth": 2},
                       whiskerprops={"color": "0.8"}, capprops={"color": "0.8"})
    for patch, team in zip(boxes["boxes"], order):
        patch.set_facecolor(team_colour(team_ids[team]))
        patch.set_edgecolor("0.8")

    ax.set_xticks(range(1, len(order) + 1), order, rotation=30, ha="right")
    laptime_axis(ax)
    ax.set_title(f"Team race pace · {session.name}")
    return ax


def _positions_with_grid(session: Session) -> pd.DataFrame:
    """Lap-by-lap positions plus a lap-0 row from the grid (pit-lane starts go to the back)."""
    laps = require_race_laps(session)
    res = session.results
    grid = pd.DataFrame({
        "Driver": res["Driver"],
        "LapNumber": 0,
        "Position": res["GridPosition"].fillna(0).astype(int).replace(0, len(res)),
    })
    per_lap = pd.DataFrame(laps[["Driver", "LapNumber", "Position"]])
    both = pd.concat([grid, per_lap], ignore_index=True)
    both["Position"] = pd.to_numeric(both["Position"], errors="coerce").astype(float)
    return both.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
