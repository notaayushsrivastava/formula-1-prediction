"""Phase 2b gallery: all six plots for one race weekend and its season.

Run:  python examples/plot_gallery.py 2024 Monza
Saves six PNGs in the current folder (~25 requests the first time, then cached).
"""
import logging
import sys

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import pitwall
from pitwall.plotting import (
    championship_progress, laptime_distribution, points_heatmap, position_changes,
    qualifying_overview, team_pace_boxplot, use_pitwall_style,
)


def save(ax: Axes, path: str) -> None:
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)  # free memory; we make six figures
    print(f"Saved {path}")


def main(year: int, event: int | str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # see downloads
    use_pitwall_style()
    season = pitwall.Season(year).load()
    race = season.session(event).load()
    quali = season.session(event, "Q").load()
    tag = f"{year}_r{race.round}"

    save(laptime_distribution(race), f"distribution_{tag}.png")
    save(position_changes(race), f"positions_{tag}.png")
    save(team_pace_boxplot(race), f"team_pace_{tag}.png")
    save(qualifying_overview(quali), f"qualifying_{tag}.png")
    save(points_heatmap(season), f"points_heatmap_{year}.png")
    save(championship_progress(season), f"championship_{year}.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    year = int(args[0]) if args else 2024
    event = args[1] if len(args) > 1 else "Monza"
    event = int(event) if str(event).isdigit() else event
    main(year, event)
