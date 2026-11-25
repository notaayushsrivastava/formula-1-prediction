"""Phase 3 analysis: tyre degradation, bootstrap intervals, Monte Carlo title odds."""
from .bootstrap import (
    PaceDifference, bootstrap_ci, clean_pace_laps, driver_pace_table, pace_difference,
)
from .degradation import (
    FUEL_EFFECT_S_PER_LAP, add_fuel_corrected, add_tyre_age, clean_stint_laps,
    degradation_table,
)
from .montecarlo import FieldModel, build_field_model, simulate_championship

__all__ = [
    "PaceDifference", "bootstrap_ci", "clean_pace_laps", "driver_pace_table",
    "pace_difference", "FUEL_EFFECT_S_PER_LAP", "add_fuel_corrected", "add_tyre_age",
    "clean_stint_laps", "degradation_table", "FieldModel", "build_field_model",
    "simulate_championship",
]
