"""Team/driver/compound colours and the pitwall matplotlib style."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from ..core import Session

# Keys are Jolpica constructorIds. Hex values are approximate.
TEAM_COLOURS: dict[str, str] = {
    "red_bull": "#3671C6", "ferrari": "#E8002D", "mercedes": "#27F4D2",
    "mclaren": "#FF8000", "aston_martin": "#229971", "alpine": "#0093CC",
    "williams": "#64C4FF", "rb": "#6692FF", "sauber": "#52E252",
    "haas": "#B6BABD", "audi": "#BB0A30", "cadillac": "#C8A951",
    # older names
    "alphatauri": "#5E8FAA", "toro_rosso": "#469BFF", "alfa": "#C92D4B",
    "racing_point": "#F596C8", "force_india": "#F596C8", "renault": "#FFF500",
}

STINT_COLOURS = ["#4E9AF1", "#F5A623", "#7ED321", "#D0021B", "#BD10E0", "#50E3C2"]

COMPOUND_COLOURS: dict[str, str] = {
    "SOFT": "#DA291C", "MEDIUM": "#FFD12E", "HARD": "#F0F0EC",
    "INTERMEDIATE": "#43B02A", "WET": "#0067AD", "UNKNOWN": "#00FFFF",
}

_FALLBACK = plt.get_cmap("tab20").colors


def team_colour(team_id: str | None) -> str:
    """Hex colour for a constructorId; a stable fallback colour if unknown."""
    key = str(team_id or "unknown")
    if key in TEAM_COLOURS:
        return TEAM_COLOURS[key]
    index = sum(map(ord, key)) % len(_FALLBACK)
    return mcolors.to_hex(_FALLBACK[index])


def stint_colour(stint: int) -> str:
    """Colour for a stint number (1-based), cycling through STINT_COLOURS."""
    return STINT_COLOURS[(int(stint) - 1) % len(STINT_COLOURS)]


def compound_colour(compound: str) -> str:
    """Colour for a tyre compound name (cyan if unknown)."""
    return COMPOUND_COLOURS.get(str(compound).upper(), COMPOUND_COLOURS["UNKNOWN"])


def driver_style(session: Session, driver: str) -> dict[str, str]:
    """Matplotlib kwargs for a driver in one session, by code ('LEC') or id ('leclerc')."""
    return style_for(session.results, session.get_driver(driver)["Driver"])


def style_for(results: pd.DataFrame, driver: str) -> dict[str, str]:
    """Team colour for a driver code; the 2nd teammate (alphabetically) gets dashes/squares.

    Works on one race's results or a whole season's (the driver's latest team counts).
    """
    latest = results.drop_duplicates("Driver", keep="last").set_index("Driver")
    if driver not in latest.index:
        return {"color": team_colour(driver), "linestyle": "-", "marker": "o"}
    team_id = latest.at[driver, "TeamId"]
    teammates = sorted(latest.index[latest["TeamId"] == team_id])
    second = teammates.index(driver) > 0
    return {
        "color": team_colour(team_id),
        "linestyle": "--" if second else "-",
        "marker": "s" if second else "o",
    }


def use_pitwall_style() -> None:
    """Dark, clean plot style (similar in spirit to FastF1's setup_mpl)."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.figsize": (9, 5),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })
