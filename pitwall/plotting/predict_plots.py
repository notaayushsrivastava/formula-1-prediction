"""Prediction plots: calibration curves (does '20%' really happen 20% of the time?)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from ..predict.metrics import calibration_table, expected_calibration_error
from .colours import STINT_COLOURS


def calibration_plot(preds: pd.DataFrame, prob: str = "P_win", label: str = "Win",
                     by: str = "Model", bins: int = 10, ax: Axes | None = None) -> Axes:
    """Predicted probability vs observed frequency per model; dot size = rows in the bin.

    Points on the dashed diagonal are perfectly calibrated. Below it: over-confident.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for i, (name, group) in enumerate(preds.groupby(by, sort=False)):
        table = calibration_table(group[prob], group[label], bins)
        ece = expected_calibration_error(group[prob], group[label], bins)
        colour = STINT_COLOURS[i % len(STINT_COLOURS)]
        sizes = 15 + 120 * table["Count"] / table["Count"].max()
        ax.plot(table["Predicted"], table["Observed"], color=colour, linewidth=1.5,
                label=f"{name} (ECE {ece:.3f})")
        ax.scatter(table["Predicted"], table["Observed"], s=sizes, color=colour, zorder=3)
    ax.plot([0, 1], [0, 1], color="white", linestyle="--", linewidth=1, alpha=0.6,
            label="Perfect")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Predicted {label.lower()} probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration: {label}")
    ax.legend(loc="upper left", frameon=False)
    return ax
