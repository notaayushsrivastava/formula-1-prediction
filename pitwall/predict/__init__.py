"""pitwall.predict: race prediction (v2).

Phase 7: dataset. Phase 8: B0-B2, backtest, metrics, report. Phase 9: Kalman strength
filter and M1. Phase 10: M2 neural ranker. Phase 11: M3 simulator, online stacking,
online calibration, the final pipeline.
Only pandas/numpy/scipy are imported here; heavy libraries load lazily.
"""
from .dataset import (
    ALL_FEATURES, DIAGNOSTICS, FEATURES, ID_COLUMNS, LABELS, MOMENTS, OBSERVATIONS,
    build_dataset, feature_columns, load_dataset, save_dataset,
)
from .history import History, load_history
from .backtest import (
    BacktestResult, GateResult, LeakageError, LockedSeasonError, compare_models, gate,
    run_backtest, walk_forward,
)
from .metrics import BENCHMARKS, paired_bootstrap, summarise
from .models import (
    GridSlotBaseline, LightGBMRanker, LogisticModel, NeuralRanker, RaceSimulator,
    ScoreSource, SimParams, StrengthModel, as_score_source,
)
from .report import write_report
from .strength import StrengthFilter, StrengthParams, fit_params, observation_table
from .strength_features import STRENGTH_FEATURES, StrengthFeatures
from .ensemble import (
    METHODS, build_stacks, fit_weights, pool, result_from_predictions, stack_forward,
)
from .calibration import calibrate_forward, fit_temperature, temper
from .pipeline import CONFIG_PATH, FinalPipeline, MomentChoice, PipelineConfig

__all__ = ["ALL_FEATURES", "DIAGNOSTICS", "FEATURES", "ID_COLUMNS", "LABELS", "MOMENTS",
           "OBSERVATIONS", "build_dataset", "feature_columns", "load_dataset",
           "save_dataset", "History", "load_history",
           "BacktestResult", "GateResult", "LeakageError", "LockedSeasonError",
           "compare_models", "gate", "run_backtest", "walk_forward",
           "BENCHMARKS", "paired_bootstrap", "summarise",
           "GridSlotBaseline", "LightGBMRanker", "LogisticModel", "NeuralRanker",
           "RaceSimulator", "ScoreSource", "SimParams", "StrengthModel", "as_score_source",
           "write_report", "StrengthFilter", "StrengthParams", "fit_params",
           "observation_table", "STRENGTH_FEATURES", "StrengthFeatures",
           "METHODS", "build_stacks", "fit_weights", "pool", "result_from_predictions",
           "stack_forward", "calibrate_forward", "fit_temperature", "temper",
           "CONFIG_PATH", "FinalPipeline", "MomentChoice", "PipelineConfig"]
