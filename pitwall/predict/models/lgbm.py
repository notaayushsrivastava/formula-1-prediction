"""B2: a LightGBM lambdarank ranker plus a LightGBM DNF classifier.

- The ranker learns, race by race, which drivers finish ahead (relevance = how many
  drivers you beat). Monotone constraints say a better grid, a smaller quali gap or a
  smaller practice gap can never make the score worse.
- Ranker scores have no natural scale, so a temperature is fitted by Plackett-Luce
  likelihood on OUT-OF-FOLD scores (races the model did not train on).
- Small trees, few rounds and fixed seeds: a sensible, reproducible default.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._optional import require
from .base import (
    N_SIMS, design_matrix, features_of, fit_binary, moment_of, out_of_fold_scale,
    predict_from_scores,
)

# -1: a bigger value may never increase the score (bigger grid number = further back)
MONOTONE = {"grid": -1, "quali_gap_pct": -1, "practice_pace_gap_pct": -1,
            "teammate_quali_gap_pct": -1}


class LightGBMRanker:
    """B2: lambdarank scores -> Plackett-Luce with a fitted temperature -> Monte Carlo."""

    name = "B2 LightGBM"

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 num_leaves: int = 15, min_child_samples: int = 20, folds: int = 3,
                 n_sims: int = N_SIMS, seed: int = 0) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.folds = folds
        self.n_sims = n_sims
        self.seed = seed

    def params(self) -> dict[str, Any]:
        """Settings logged with each experiment."""
        return {"n_estimators": self.n_estimators, "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves, "min_child_samples": self.min_child_samples,
                "folds": self.folds, "n_sims": self.n_sims, "seed": self.seed}

    def fit(self, train: pd.DataFrame) -> LightGBMRanker:
        """Ranker on all rows, temperature out-of-fold, then the DNF classifier."""
        lgb = require("lightgbm", "B2 LightGBMRanker")
        self.moment_ = moment_of(train)
        self.features_ = features_of(train)
        self.ranker_ = self._fit_ranker(train)
        self.scale_ = out_of_fold_scale(train, self._fit_and_score, self.folds, self.seed)
        classifier = lgb.LGBMClassifier(n_estimators=150, num_leaves=7, min_child_samples=40,
                                        **self._common())
        self.dnf_ = fit_binary(classifier, design_matrix(train, self.features_),
                               train["DNF"].to_numpy())
        return self

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """Probabilities for one race of the moment the model was fitted on."""
        if moment_of(race) != self.moment_:
            raise ValueError(f"{self.name} was fitted on '{self.moment_}' rows.")
        features = design_matrix(race, self.features_)
        scores = self.ranker_.predict(features)
        p_dnf = self.dnf_.predict_proba(features)[:, 1]
        return predict_from_scores(race, scores, p_dnf, self.scale_, self.n_sims, self.seed)

    def feature_importance(self) -> pd.Series:
        """Total gain per feature in the ranker, biggest first."""
        gains = self.ranker_.booster_.feature_importance(importance_type="gain")
        return pd.Series(gains, index=self.features_).sort_values(ascending=False)

    # ---------- helpers ----------
    def _common(self) -> dict[str, Any]:
        """Settings shared by ranker and classifier (fixed seeds, deterministic, quiet)."""
        return {"learning_rate": self.learning_rate, "subsample": 0.8, "subsample_freq": 1,
                "colsample_bytree": 0.8, "random_state": self.seed, "deterministic": True,
                "force_row_wise": True, "n_jobs": 1, "verbose": -1}

    def _fit_ranker(self, rows: pd.DataFrame) -> Any:
        """One lambdarank model; each race is a group, relevance = drivers beaten."""
        lgb = require("lightgbm", "B2 LightGBMRanker")
        ordered = rows.sort_values(["RaceId", "FinishOrder"])  # groups must be contiguous
        field = ordered.groupby("RaceId")["FinishOrder"].transform("max")
        relevance = (field - ordered["FinishOrder"]).to_numpy(int)
        groups = ordered.groupby("RaceId").size().to_numpy()
        ranker = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=self.n_estimators,
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples,
            label_gain=list(range(int(relevance.max()) + 1)),  # linear gain per place
            monotone_constraints=[MONOTONE.get(f, 0) for f in self.features_],
            **self._common())
        ranker.fit(design_matrix(ordered, self.features_), relevance, group=groups)
        return ranker

    def _fit_and_score(self, fit_rows: pd.DataFrame, score_rows: pd.DataFrame) -> np.ndarray:
        """Train a fresh ranker on fit_rows and score score_rows (for out-of-fold scaling)."""
        ranker = self._fit_ranker(fit_rows)
        return np.asarray(ranker.predict(design_matrix(score_rows, self.features_)), float)
