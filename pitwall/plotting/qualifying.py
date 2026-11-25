"""Qualifying plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..core import Session
from .colours import team_colour
from .laptimes import format_laptime


def qualifying_overview(session: Session, ax: Axes | None = None) -> Axes:
    """Gap from each driver's best lap to the session's fastest lap (FastF1 gallery plot).

    Bars follow the official grid order (pole at the top). Drivers without a time are skipped.
    """
    if "DeltaToFastestSeconds" not in session.results:
        raise ValueError(f"{session.name} is not a loaded qualifying session. "
                         "Use get_session(year, event, 'Q').load().")
    res = session.results.dropna(subset=["BestSeconds"]).reset_index(drop=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    gaps = res["DeltaToFastestSeconds"].to_numpy(dtype=float)
    bars = ax.barh(range(len(res)), gaps, color=[team_colour(t) for t in res["TeamId"]])
    ax.bar_label(bars, labels=[f"+{g:.3f}" for g in gaps], padding=3, fontsize=8)
    ax.set_yticks(range(len(res)), res["Driver"])
    ax.invert_yaxis()  # pole at the top
    ax.grid(axis="y", visible=False)

    fastest = res.loc[res["BestSeconds"].idxmin()]
    ax.set_xlabel("Gap to fastest lap (s)")
    ax.set_title(f"{session.name}\nFastest lap: {fastest['Driver']} "
                 f"{format_laptime(fastest['BestSeconds'])}")
    return ax
