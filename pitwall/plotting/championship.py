"""Championship plots: points heatmap, the title fight over the season, who can still win."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from ..errors import DataNotAvailableError
from ..title_fight import points_still_available, who_can_still_win
from .colours import style_for

if TYPE_CHECKING:
    from ..season import Season


def points_heatmap(season: Season, ax: Axes | None = None) -> Axes:
    """Points each driver scored in each round (race + sprint), best driver at the top."""
    table = _points_table(season)
    rounds = table.drop(columns="Total")
    values = rounds.to_numpy(dtype=float)
    if ax is None:
        size = (3 + 0.45 * values.shape[1], 1.5 + 0.35 * values.shape[0])
        _, ax = plt.subplots(figsize=size)

    image = ax.imshow(values, cmap="viridis", aspect="auto")
    _annotate_cells(ax, values)
    ax.set_xticks(range(values.shape[1]), _round_labels(season, rounds.columns), rotation=90)
    ax.set_yticks(range(values.shape[0]),
                  [f"{d} ({t:g})" for d, t in zip(table.index, table["Total"])])
    ax.grid(False)
    ax.figure.colorbar(image, ax=ax, label="Points")
    ax.set_title(f"{season.year} points per round (race + sprint)")
    return ax


def championship_progress(season: Season, top_n: int = 5, ax: Axes | None = None) -> Axes:
    """Cumulative points of the top N drivers after each round: the title fight at a glance."""
    table = _points_table(season)
    cumulative = table.drop(columns="Total").head(top_n).cumsum(axis=1)
    x = np.arange(1, cumulative.shape[1] + 1)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    for driver, points in cumulative.iterrows():
        style = style_for(season.results, driver)
        ax.plot(x, points.to_numpy(dtype=float), color=style["color"],
                linestyle=style["linestyle"], marker=style["marker"], markersize=3,
                label=f"{driver} {points.iloc[-1]:g}")

    ax.set_xticks(x, _round_labels(season, cumulative.columns), rotation=90)
    ax.set_ylabel("Points (cumulative)")
    ax.set_title(f"{season.year} championship: top {len(cumulative)}")
    ax.legend(loc="upper left", frameon=False)
    return ax


def title_fight_chart(season: Season, top_n: int = 8, ax: Axes | None = None) -> Axes:
    """Points now (solid) and the best case still possible (faded) vs the leader (dashed)."""
    table = who_can_still_win(season).head(top_n)
    left = points_still_available(season)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 1.5 + 0.5 * len(table)))
    y = np.arange(len(table))
    now = table["Points"].to_numpy(dtype=float)
    best = table["MaxPoints"].to_numpy(dtype=float)
    colours = [style_for(season.results, d)["color"] for d in table["Driver"]]

    ax.barh(y, now, color=colours)
    ax.barh(y, best - now, left=now, color=colours, alpha=0.3)
    ax.axvline(now[0], color="white", linestyle="--", linewidth=1)
    ax.set_yticks(y, [f"{d} · {s}" for d, s in zip(table["Driver"], table["Status"])])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Points (solid = now, faded = best case, dashed = leader)")
    ax.set_title(f"{season.year}: who can still win? "
                 f"({left.races} races, {left.sprints} sprints left)")
    return ax


def _points_table(season: Season) -> pd.DataFrame:
    table = season.points_by_round()
    if table.empty:
        raise DataNotAvailableError(
            f"No results for {season.year} yet. Did you call season.load()?")
    return table


def _round_labels(season: Season, rounds: Iterable[int]) -> list[str]:
    """Round numbers -> short event names ('Italian Grand Prix' -> 'Italian')."""
    names = season.schedule.set_index("Round")["EventName"]
    short = names.str.replace(" Grand Prix", "", regex=False)
    return [str(short.get(int(r), r)) for r in rounds]


def _annotate_cells(ax: Axes, values: np.ndarray) -> None:
    """Write each non-zero value in its cell; dark text on bright cells."""
    top = values.max() or 1.0
    for (row, col), value in np.ndenumerate(values):
        if value:
            ax.text(col, row, f"{value:g}", ha="center", va="center", fontsize=7,
                    color="black" if value > 0.6 * top else "white")
