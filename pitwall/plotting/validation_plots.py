"""Phase 15 plots: CPCV path distributions and embargo sensitivity."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from .colours import STINT_COLOURS


def cpcv_boxplot(paths: pd.DataFrame, walk_forward: pd.Series | None = None,
                 title: str = "CPCV path log loss", ax: Axes | None = None) -> Axes:
    """One box per model: the log loss of each CPCV path (lower = better).

    paths: columns Model, Path, LogLoss. walk_forward (optional): Model -> the
    walk-forward log loss, drawn as a diamond for comparison.
    """
    models = list(dict.fromkeys(paths["Model"]))
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(models) + 2), 4.5))
    values = [paths.loc[paths["Model"] == m, "LogLoss"].to_numpy(float) for m in models]
    box = ax.boxplot(values, patch_artist=True, widths=0.55)
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(STINT_COLOURS[i % len(STINT_COLOURS)])
        patch.set_alpha(0.6)
    ax.set_xticks(range(1, len(models) + 1), labels=models, rotation=20, ha="right")
    for i, model in enumerate(models):
        ax.scatter([i + 1] * len(values[i]), values[i], s=12, color="white", zorder=3)
        if walk_forward is not None and model in walk_forward.index:
            ax.scatter([i + 1], [walk_forward[model]], marker="D", s=40, color="#e10600",
                       zorder=4, label="walk-forward" if i == 0 else None)
    ax.set_ylabel("Log loss (lower = better)")
    ax.set_title(title)
    if walk_forward is not None:
        ax.legend(frameon=False, fontsize="small")
    return ax


def embargo_plot(table: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Mean CPCV win log loss against the embargo, one line per (moment, model).

    table: columns Moment, Model, Embargo, LogLoss.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    for i, ((moment, model), group) in enumerate(table.groupby(["Moment", "Model"],
                                                               sort=False)):
        group = group.sort_values("Embargo")
        ax.plot(group["Embargo"], group["LogLoss"], marker="o",
                color=STINT_COLOURS[i % len(STINT_COLOURS)], label=f"{moment}: {model}")
    ax.set_xlabel("Embargo (race weekends removed after each test block)")
    ax.set_ylabel("Mean CPCV win log loss")
    ax.set_title("Does the score depend on the embargo? (flat = no leakage left)")
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize="x-small", frameon=False)
    return ax
