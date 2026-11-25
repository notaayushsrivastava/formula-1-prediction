"""Kalman strength features shared by M1 and M2 (Phase 10 refactor of StrengthModel).

fit(train): run the filter over the training rows' OBSERVATIONS and give every training
row the filter's ONE-STEP-AHEAD estimate (made before that race).
for_race(race): move the filter forward to a future race (no update) and give its rows
the same columns. Nothing here ever sees the race being predicted.

Phase 15 fix: in CPCV the training rows can include races AFTER the race being
predicted. A Kalman filter's state is a running summary of the past, so the forecast for
such a race now comes from a filter run only over the training races dated BEFORE it
(cached per date). Walk-forward predictions (always after the last training race) are
unchanged.
"""
from __future__ import annotations

import pandas as pd

from .strength import (
    OUTPUT_COLUMNS, StrengthFilter, StrengthParams, fitted_params, hyper_subset,
    observation_table,
)

STRENGTH_FEATURES = ["car_mean", "skill_mean", "pace_sd"]  # pace_pred = car + skill
STRENGTH_COLUMNS = [c for c in OUTPUT_COLUMNS if c not in ("RaceId", "DriverId", "TeamId")]


class StrengthFeatures:
    """Fit once on training rows, then add the same strength columns to any future race."""

    def __init__(self, params: StrengthParams | None = None, maxiter: int = 60) -> None:
        self.fixed_params = params  # None: fit hyperparameters (cached, training data only)
        self.maxiter = maxiter

    def fit(self, train: pd.DataFrame) -> pd.DataFrame:
        """Training rows plus one-step-ahead strength columns (new index)."""
        obs = observation_table(train)
        self.params_ = self.fixed_params or fitted_params(hyper_subset(obs), self.maxiter)
        self.filter_ = StrengthFilter(self.params_).run(obs)
        self.obs_ = obs
        self._past_filters: dict[pd.Timestamp, StrengthFilter] = {}
        strength = self.filter_.predictions.drop(columns="TeamId")
        return train.merge(strength, on=["RaceId", "DriverId"], how="left")

    def for_race(self, race: pd.DataFrame) -> pd.DataFrame:
        """The race rows (same order) plus the filter's forecast for each driver."""
        forecast = self.filter_before(pd.Timestamp(race["Date"].iloc[0])).forecast(race)
        rows = race.reset_index(drop=True).copy()
        for column in STRENGTH_COLUMNS:
            rows[column] = forecast[column].to_numpy()
        return rows

    def filter_before(self, date: pd.Timestamp) -> StrengthFilter:
        """The filter as it stood before `date`: the fitted filter for a race after the
        training data (walk-forward), else one run over the training races before `date`."""
        last = self.filter_.last_date
        if last is None or date > last:
            return self.filter_
        if date not in self._past_filters:
            earlier = self.obs_[self.obs_["Date"] < date]
            self._past_filters[date] = StrengthFilter(self.params_).run(earlier, record=False)
        return self._past_filters[date]

    def strengths(self) -> pd.DataFrame:
        """Latest car and driver estimates (lower Mean = faster)."""
        return self.filter_.strengths()
