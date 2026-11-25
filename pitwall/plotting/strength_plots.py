"""Strength plots (Phase 9): car pace over time, driver skills, pace vs result.

Gaps are in % versus the typical car; LOWER = faster, so the y-axes are flipped and
faster is always at the top.
"""
from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy import stats

from ..predict.features import ERA_STARTS
from .colours import team_colour


def _colour(team_id: str) -> str:
    """f1db ids use dashes ('red-bull'); the colour table uses Jolpica ids ('red_bull')."""
    return team_colour(str(team_id).replace("-", "_"))


def car_strength_plot(smoothed: pd.DataFrame, teams: Iterable[str] | None = None,
                      eras: Iterable[int] = ERA_STARTS, ax: Axes | None = None) -> Axes:
    """Smoothed car pace per team with +/- 1 sd bands; dotted lines at new rules eras.

    `smoothed` comes from StrengthFilter.smooth(): hindsight estimates, for plots only.
    """
    cars = smoothed[smoothed["Kind"] == "car"]
    if teams is not None:
        cars = cars[cars["Id"].isin(list(teams))]
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    for team, group in cars.groupby("Id"):
        colour = _colour(str(team))
        ax.plot(group["Date"], group["Mean"], color=colour, linewidth=1.6, label=str(team))
        ax.fill_between(group["Date"], group["Mean"] - group["Sd"], group["Mean"] + group["Sd"],
                        color=colour, alpha=0.15, linewidth=0)
    dates = pd.to_datetime(smoothed["Date"])
    for era in eras:
        in_era = dates[dates.dt.year == era]
        if not in_era.empty:
            start = in_era.min()
            ax.axvline(start, color="white", linestyle=":", alpha=0.6)
            ax.text(start, 1.01, f" {era} rules", transform=ax.get_xaxis_transform(),
                    fontsize=8, color="white")
    ax.invert_yaxis()
    ax.set_ylabel("Car pace vs typical car (% gap, faster = up)")
    ax.set_title("Car strength (Kalman smoother, +/- 1 sd)")
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize="small", frameon=False)
    return ax


def driver_skill_plot(strengths: pd.DataFrame, current_only: bool = True,
                      ax: Axes | None = None) -> Axes:
    """Driver skill estimates with +/- 1 sd, fastest at the top (latest grid by default)."""
    skills = strengths[strengths["Kind"] == "skill"]
    if current_only:
        skills = skills[skills["InLatestRace"]]
    skills = skills.sort_values("Mean").reset_index(drop=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 1.5 + 0.35 * len(skills)))
    y = np.arange(len(skills))
    colours = [_colour(t) for t in skills["TeamId"]]
    ax.errorbar(skills["Mean"], y, xerr=skills["Sd"], fmt="none", ecolor="0.8", capsize=3)
    ax.scatter(skills["Mean"], y, c=colours, s=50, zorder=3)
    ax.set_yticks(y, skills["Id"])
    ax.invert_yaxis()
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Driver skill (% gap; negative = faster than average)")
    ax.set_title("Driver skill estimates (+/- 1 sd)")
    return ax


def strength_vs_result(table: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Predicted pace (before the race) against the actual finishing order of one race.

    table needs Driver, TeamId, pace_pred, pace_sd and FinishOrder.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    colours = [_colour(t) for t in table["TeamId"]]
    ax.errorbar(table["pace_pred"], table["FinishOrder"], xerr=table["pace_sd"], fmt="none",
                ecolor="0.6", capsize=2)
    ax.scatter(table["pace_pred"], table["FinishOrder"], c=colours, s=50, zorder=3)
    for row in table.itertuples(index=False):
        ax.annotate(str(row.Driver), (row.pace_pred, row.FinishOrder), fontsize=7,
                    xytext=(4, 2), textcoords="offset points")
    rho = stats.spearmanr(table["pace_pred"], table["FinishOrder"])[0]
    ax.invert_yaxis()
    ax.set_xlabel("Predicted pace (% gap, left = faster)")
    ax.set_ylabel("Finishing order (DNFs last)")
    ax.set_title(f"Predicted pace vs result (Spearman {rho:.2f})")
    return ax
