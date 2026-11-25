"""ScoreSource: one score per driver (higher = faster) plus its uncertainty.

M3 and the final pipeline need a model's *scores*, not only its probabilities. These
adapters read them from FITTED B1 / M1 / M1-pure / M2 models without changing how those
models predict: predict_from_scores(race, score, p_dnf, 1.0, ..., score_sd=sd) gives
exactly the model's own prediction (tested). Scores already include the temperature.
training_scores(train): scores for the training rows (one-step-ahead for M1, so honest;
in-sample for B1 and M2).
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from .base import design_matrix
from .logistic import LogisticModel
from .neural import NeuralRanker
from .strength_model import StrengthModel

Scores = tuple[np.ndarray, np.ndarray]


class ScoreSource(Protocol):
    """Anything that gives (score, score_sd) per driver of a race."""

    def scores(self, race: pd.DataFrame) -> Scores:
        """Scores for one race's rows (same order)."""
        ...

    def training_scores(self, train: pd.DataFrame) -> Scores:
        """Scores for the rows the model was trained on (same order)."""
        ...


class LinearScores:
    """B1: score = standardised features @ ranking weights (no uncertainty)."""

    def __init__(self, model: LogisticModel) -> None:
        self.model = model

    def scores(self, race: pd.DataFrame) -> Scores:
        m = self.model
        score = m.scaler_.transform(design_matrix(race, m.features_)) @ m.weights_
        return score, np.zeros(len(score))

    def training_scores(self, train: pd.DataFrame) -> Scores:
        return self.scores(train)


class StrengthScores:
    """M1 / M1-pure: scores from the Kalman strengths (and features, for M1)."""

    def __init__(self, model: StrengthModel) -> None:
        self.model = model

    def scores(self, race: pd.DataFrame) -> Scores:
        return self._from_rows(self.model.strength_features(race))

    def training_scores(self, train: pd.DataFrame) -> Scores:
        strength = self.model.filter_.predictions.drop(columns="TeamId")
        return self._from_rows(train.merge(strength, on=["RaceId", "DriverId"], how="left"))

    def _from_rows(self, rows: pd.DataFrame) -> Scores:
        m = self.model
        if m.pure:
            return (m.scale_ * -rows["pace_pred"].to_numpy(float),
                    m.scale_ * rows["pace_sd"].to_numpy(float))
        z = m.scaler_.transform(design_matrix(rows, m.features_))
        return z @ m.weights_, m._score_sd(rows)


class NeuralScores:
    """M2: mean of the seed networks' scores; spread = seed disagreement (+ M1 uncertainty)."""

    def __init__(self, model: NeuralRanker) -> None:
        self.model = model

    def scores(self, race: pd.DataFrame) -> Scores:
        return self._from_rows(self.model._prepare(race))

    def training_scores(self, train: pd.DataFrame) -> Scores:
        m = self.model
        if m.use_strength:
            strength = m.strength_.filter_.predictions.drop(columns="TeamId")
            rows = train.merge(strength, on=["RaceId", "DriverId"], how="left")
        else:
            rows = train.reset_index(drop=True)
        return self._from_rows(rows)

    def _from_rows(self, rows: pd.DataFrame) -> Scores:
        m = self.model
        seeds, _ = m._ensemble(rows)
        spread = seeds.std(axis=0)
        if m.use_strength and m.combine_pace_sd:
            spread = np.sqrt(spread**2 + m._strength_var(rows))
        return m.scale_ * seeds.mean(axis=0), m.scale_ * spread


def as_score_source(model: Any) -> Any:
    """A ScoreSource for a fitted model (objects with their own scores() pass through)."""
    if hasattr(model, "scores") and hasattr(model, "training_scores"):
        return model
    if isinstance(model, LogisticModel):
        return LinearScores(model)
    if isinstance(model, StrengthModel):
        return StrengthScores(model)
    if isinstance(model, NeuralRanker):
        return NeuralScores(model)
    raise TypeError(f"{getattr(model, 'name', model)!r} cannot provide scores.")
