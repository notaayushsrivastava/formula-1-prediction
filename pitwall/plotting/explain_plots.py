"""Explanation plots for M2: driver embeddings in 2D, integrated-gradient attributions."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .colours import team_colour


def embedding_plot(model: Any, ax: Axes | None = None) -> Axes:
    """Driver embeddings projected to 2D with PCA (NumPy SVD), coloured by latest team.

    Drivers close together were treated as similar by the network.
    """
    table = model.embeddings()
    columns = [c for c in table.columns if c.startswith("emb_")]
    values = table[columns].to_numpy(float)
    centred = values - values.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    xy = centred @ vt[:2].T
    if xy.shape[1] < 2:
        xy = np.column_stack([xy[:, 0], np.zeros(len(xy))])
    share = singular**2 / max(float((singular**2).sum()), 1e-12)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    colours = [team_colour(str(t).replace("-", "_")) for t in table["TeamId"]]
    ax.scatter(xy[:, 0], xy[:, 1], c=colours, s=60, zorder=3)
    for (x, y), code in zip(xy, table["Driver"]):
        ax.annotate(str(code), (x, y), fontsize=7, xytext=(4, 2), textcoords="offset points")
    second = share[1] if len(share) > 1 else 0.0
    ax.set_xlabel(f"PC1 ({share[0]:.0%} of variation)")
    ax.set_ylabel(f"PC2 ({second:.0%})")
    ax.set_title(f"{model.name}: driver embeddings (seed 0, PCA)")
    return ax


def attribution_plot(attributions: pd.DataFrame, top: int = 8,
                     ax: Axes | None = None) -> Axes:
    """Heatmap of integrated gradients: drivers x the `top` most important features.

    Positive (blue) pushed the driver's score up (faster) versus an average driver.
    """
    order = attributions.abs().mean().sort_values(ascending=False).index[:top]
    values = attributions[order].to_numpy(float)
    limit = float(np.abs(values).max()) or 1.0
    if ax is None:
        _, ax = plt.subplots(figsize=(1.5 + 0.9 * len(order), 1.5 + 0.35 * len(attributions)))
    image = ax.imshow(values, cmap="coolwarm_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(order)), list(order), rotation=45, ha="right")
    ax.set_yticks(range(len(attributions)), list(attributions.index))
    ax.grid(False)
    ax.figure.colorbar(image, ax=ax, label="Contribution to score (+ = faster)")
    ax.set_title("Integrated gradients: why each driver got their score")
    return ax
