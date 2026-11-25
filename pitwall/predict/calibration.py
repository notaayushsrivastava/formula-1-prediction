"""Online calibration of a pipeline's probabilities (fitted on earlier races only).

- Matrix temperature: P ** T, then Sinkhorn. T < 1 flattens (less confident), T > 1
  sharpens. T is fitted by position log loss.
- Platt scaling for P_dnf: sigmoid(a * logit(p) + b).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from .dataset import LABELS
from .ensemble import CONTEXT, MIN_TRAIN_RACES
from .models.base import position_matrix, prediction_frame, sinkhorn

TEMPERATURE_BOUNDS = (0.2, 5.0)
PLATT_L2 = 1e-3
EPS = 1e-9


def temper(matrix: np.ndarray, t: float) -> np.ndarray:
    """P ** t, rebalanced so rows and columns sum to 1."""
    return sinkhorn(np.clip(matrix, EPS, 1.0) ** t)


def position_log_loss(matrix: np.ndarray, order: np.ndarray) -> float:
    """Mean -log P(driver i finished where he did)."""
    idx = np.arange(len(order))
    return -float(np.mean(np.log(np.clip(matrix[idx, order], EPS, 1.0))))


def fit_temperature(matrices: list[np.ndarray], orders: list[np.ndarray]) -> float:
    """The T minimising mean position log loss of temper(P, T) over earlier races."""
    def loss(t: float) -> float:
        return float(np.mean([position_log_loss(temper(m, t), o)
                              for m, o in zip(matrices, orders)]))

    result = optimize.minimize_scalar(loss, bounds=TEMPERATURE_BOUNDS, method="bounded")
    return float(result.x)


def fit_platt(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(a, b) of sigmoid(a * logit(p) + b), gently pulled toward (1, 0) = unchanged."""
    z = _logit(np.asarray(p, float))
    labels = np.asarray(y, float)

    def loss(v: np.ndarray) -> float:
        s = v[0] * z + v[1]
        bce = float(np.mean(np.logaddexp(0.0, s) - labels * s))
        return bce + PLATT_L2 * float((v[0] - 1.0) ** 2 + v[1] ** 2)

    result = optimize.minimize(loss, np.array([1.0, 0.0]), method="L-BFGS-B")
    return float(result.x[0]), float(result.x[1])


def apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """Calibrated probabilities."""
    return 1.0 / (1.0 + np.exp(-(a * _logit(np.asarray(p, float)) + b)))


def calibrate_forward(preds: pd.DataFrame,
                      min_train_races: int = MIN_TRAIN_RACES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate one pipeline's predictions race by race, using earlier races only.

    preds: one Model at one Moment, with labels. Races with fewer than min_train_races
    earlier races are skipped. Returns (calibrated predictions '<Model>+cal', history).
    """
    name = f"{preds['Model'].iloc[0]}+cal"
    races = (preds.drop_duplicates("RaceId").sort_values(["Date", "RaceId"])
             [["RaceId", "Date"]].reset_index(drop=True))
    groups = {race: g.reset_index(drop=True) for race, g in preds.groupby("RaceId")}
    extra = ["Eval"] if "Eval" in preds.columns else []
    out, history = [], []
    for race in races.itertuples(index=False):
        past = [groups[r] for r in races.loc[races["Date"] < race.Date, "RaceId"]]
        if len(past) < min_train_races:
            continue
        t = fit_temperature([position_matrix(g)[:, :len(g)] for g in past],
                            [g["FinishOrder"].to_numpy(int) - 1 for g in past])
        a, b = fit_platt(np.concatenate([g["P_dnf"].to_numpy(float) for g in past]),
                         np.concatenate([g["DNF"].to_numpy(float) for g in past]))
        rows = groups[race.RaceId]
        n = len(rows)
        frame = prediction_frame(rows, temper(position_matrix(rows)[:, :n], t),
                                 apply_platt(rows["P_dnf"].to_numpy(float), a, b))
        truth = rows[["RaceId", "DriverId", *CONTEXT, *LABELS, *extra]]
        frame = frame.merge(truth, on=["RaceId", "DriverId"], how="left")
        frame.insert(0, "Model", name)
        out.append(frame)
        history.append({"RaceId": race.RaceId, "Date": race.Date, "Temperature": t,
                        "PlattA": a, "PlattB": b})
    table = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return table, pd.DataFrame(history)


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))
