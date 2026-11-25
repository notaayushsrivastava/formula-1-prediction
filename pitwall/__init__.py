"""pitwall: a small, readable F1 data analysis library (FastF1-inspired)."""
from importlib.metadata import PackageNotFoundError, version

from .auth import TOKEN_ENV_VAR, token_status
from .core import Laps, Session, get_session
from .errors import (
    AccessBlockedError,
    AuthenticationError,
    DataNotAvailableError,
    PitwallError,
    RateLimitError,
    SourceRequestError,
)
from .points import PointsSystem, points_system
from .season import Season
from .telemetry import Telemetry, delta_time, lap_telemetry, load_telemetry, timing_lap
from .timing import load_live_timing
from .title_fight import points_still_available, title_fight_summary, who_can_still_win

try:
    __version__ = version("pitwall-f1")
except PackageNotFoundError:  # running from a source folder without `pip install -e .`
    __version__ = "0+unknown"

__all__ = [
    "Laps", "Session", "Season", "get_session", "load_live_timing",
    "Telemetry", "load_telemetry", "lap_telemetry", "timing_lap", "delta_time",
    "TOKEN_ENV_VAR", "token_status",
    "PointsSystem", "points_system",
    "points_still_available", "title_fight_summary", "who_can_still_win",
    "PitwallError", "DataNotAvailableError", "RateLimitError", "SourceRequestError",
    "AuthenticationError", "AccessBlockedError",
]
