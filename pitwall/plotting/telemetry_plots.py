"""Telemetry plots: speed traces, time delta, and track maps coloured by gear or speed."""
from __future__ import annotations

from collections.abc import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from ..core import Session
from ..telemetry import Telemetry, delta_time, lap_telemetry
from .colours import driver_style


def speed_trace(session: Session, drivers: Iterable[str], lap: int | str = "fastest",
                ax: Axes | None = None) -> Axes:
    """Speed against distance for each driver's lap (FastF1 gallery: overlaying speed traces)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    for driver in drivers:
        tel = lap_telemetry(session, driver, lap)
        style = driver_style(session, driver)
        code = session.get_driver(driver)["Driver"]
        ax.plot(tel["Distance"].to_numpy(), tel["Speed"].to_numpy(), color=style["color"],
                linestyle=style["linestyle"], label=f"{code} lap {tel['LapNumber'].iloc[0]}")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"Speed trace · {session.name}")
    ax.legend(loc="lower right", frameon=False)
    return ax


def delta_plot(session: Session, reference: str, other: str, lap: int | str = "fastest",
               ax: Axes | None = None) -> Axes:
    """Time gap of `other` to `reference` along the lap (above zero = other is behind)."""
    ref = lap_telemetry(session, reference, lap)
    delta = delta_time(ref, lap_telemetry(session, other, lap))
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3))
    ax.plot(delta["Distance"], delta["Delta"], color=driver_style(session, other)["color"])
    ax.axhline(0, color="white", linewidth=0.8)
    ref_code = session.get_driver(reference)["Driver"]
    other_code = session.get_driver(other)["Driver"]
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel(f"{other_code} gap to {ref_code} (s)")
    return ax


def gear_map(session: Session, driver: str, lap: int | str = "fastest",
             ax: Axes | None = None) -> Axes:
    """Track map coloured by gear (FastF1 gallery: gear shifts on track)."""
    tel = _track_telemetry(session, driver, lap)
    cmap = matplotlib.colormaps["Paired"].resampled(8)
    lines = LineCollection(_segments(tel), cmap=cmap, norm=plt.Normalize(1, 9), linewidth=4)
    lines.set_array(tel["Gear"].to_numpy(float)[:-1])
    ax = _draw_track(lines, ax)
    bar = ax.figure.colorbar(lines, ax=ax, label="Gear", boundaries=np.arange(1, 10))
    bar.set_ticks(np.arange(1.5, 9.5))
    bar.set_ticklabels([str(g) for g in range(1, 9)])
    ax.set_title(f"{session.get_driver(driver)['Driver']} gears · {session.name}")
    return ax


def speed_map(session: Session, driver: str, lap: int | str = "fastest",
              ax: Axes | None = None) -> Axes:
    """Track map coloured by speed."""
    tel = _track_telemetry(session, driver, lap)
    speed = tel["Speed"].to_numpy(float)
    lines = LineCollection(_segments(tel), cmap="plasma",
                           norm=plt.Normalize(speed.min(), speed.max()), linewidth=4)
    lines.set_array(speed[:-1])
    ax = _draw_track(lines, ax)
    ax.figure.colorbar(lines, ax=ax, label="Speed (km/h)")
    ax.set_title(f"{session.get_driver(driver)['Driver']} speed · {session.name}")
    return ax


def _track_telemetry(session: Session, driver: str, lap: int | str) -> Telemetry:
    tel = lap_telemetry(session, driver, lap).dropna(subset=["X", "Y"])
    if len(tel) < 2:
        raise ValueError(f"Not enough position data for {driver} to draw the track.")
    return tel


def _segments(tel: Telemetry) -> np.ndarray:
    """Consecutive X/Y points -> line segments, shape (n - 1, 2, 2)."""
    points = np.column_stack([tel["X"], tel["Y"]]).reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)


def _draw_track(lines: LineCollection, ax: Axes | None) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    ax.add_collection(lines)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
