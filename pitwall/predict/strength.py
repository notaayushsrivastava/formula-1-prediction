"""Phase 9: car pace and driver skill from qualifying/practice gaps, with a Kalman filter.

The model, in % gap to the typical car (LOWER = faster):
    observed gap   y[d, r] = car[t, r] + skill[d] + noise     noise ~ N(0, R)
    car[t, r]      = car[t, r-1] + drift                      drift ~ N(0, Q_car per step)
    skill[d]       = skill[d] + tiny drift                    ~ N(0, Q_skill per step)
Between seasons cars get extra variance Q_winter; at a new rules era (ERA_STARTS)
also Q_era, so the filter adapts fast when the order resets.

A Kalman filter is "predict, then correct", race by race:
1. PREDICT: time has passed, so every estimate becomes a bit less certain.
2. CORRECT: compare what we expected with the gaps we saw; move each estimate toward
   the data in proportion to how unsure we were (the Kalman gain).
Teammates share a car, so their gap difference is pure skill; drivers who change teams
carry their skill into a new car. That is how car and driver are told apart.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg, optimize

from .features import ERA_STARTS

OBS_CLIP = 5.0          # centred gaps beyond +/-5% (crashes, chaos) are clipped
NEW_CAR_MEAN = 0.0      # a team we have never seen: about average...
NEW_CAR_VAR = 1.0       # ...give or take ~1% (wide prior)
DAYS_PER_STEP = 14.0    # one drift step = the usual time between two races
MIN_HYPER_RACES = 10
LOG_2PI = math.log(2 * math.pi)
VARIANCES = ("q_car", "q_skill", "q_winter", "q_era", "r_quali", "r_practice", "p_skill")
OUTPUT_COLUMNS = ["RaceId", "DriverId", "TeamId", "car_mean", "car_sd", "skill_mean",
                  "skill_sd", "car_skill_cov", "pace_pred", "pace_sd"]


@dataclass(frozen=True)
class StrengthParams:
    """Hyperparameters (variances in %^2). Fitted by fit_params; these are start values."""

    q_car: float = 0.01        # car drift per race step (sd 0.1%)
    q_skill: float = 1e-4      # skill drift per race step (much smaller than cars)
    q_winter: float = 0.1      # extra car variance between seasons
    q_era: float = 0.6         # extra car variance at a new rules era
    r_quali: float = 0.06      # noise of one qualifying gap
    r_practice: float = 0.2    # noise of one practice long-run gap
    mu_rookie: float = 0.2     # a rookie's expected skill (positive = slower than average)
    p_skill: float = 0.1       # how unsure we are about a rookie's skill

    def to_vector(self) -> np.ndarray:
        """Variances as logs (so the optimiser can never make them negative)."""
        return np.array([math.log(getattr(self, n)) for n in VARIANCES] + [self.mu_rookie])

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> StrengthParams:
        """Inverse of to_vector."""
        values = {n: float(math.exp(x)) for n, x in zip(VARIANCES, vector[:-1])}
        return cls(**values, mu_rookie=float(vector[-1]))


@dataclass(frozen=True)
class RaceObs:
    """One race's observations as arrays (prepared once, reused in every filter run)."""

    race_id: int
    date: pd.Timestamp
    year: int
    teams: list[str]
    drivers: list[str]
    quali: np.ndarray      # centred gaps, NaN = no observation
    practice: np.ndarray


# ---------- data ----------
def observation_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (race, driver), oldest first, with centred and clipped gaps.

    QualiObs / PracticeObs = gap minus that race's median gap: pace versus the typical
    car is comparable across races, while 'gap to pole' depends on who took pole.
    """
    obs = rows.drop_duplicates(["RaceId", "DriverId"]).sort_values(["Date", "RaceId",
                                                                    "DriverId"])
    out = obs[["RaceId", "Date", "Year", "DriverId", "TeamId"]].copy()
    for source, target in (("QualiGapObs", "QualiObs"), ("PracticeGapObs", "PracticeObs")):
        values = (obs[source].astype(float) if source in obs.columns
                  else pd.Series(np.nan, index=obs.index))
        centred = values - values.groupby(obs["RaceId"]).transform("median")
        out[target] = centred.clip(-OBS_CLIP, OBS_CLIP)
    return out.reset_index(drop=True)


def prepare(obs: pd.DataFrame) -> list[RaceObs]:
    """observation_table rows -> one RaceObs per race, in date order."""
    races = []
    for race_id, race in obs.groupby("RaceId", sort=False):
        races.append(RaceObs(int(race_id), pd.Timestamp(race["Date"].iloc[0]),
                             int(race["Year"].iloc[0]), race["TeamId"].astype(str).tolist(),
                             race["DriverId"].astype(str).tolist(),
                             race["QualiObs"].to_numpy(float),
                             race["PracticeObs"].to_numpy(float)))
    return races


def hyper_subset(obs: pd.DataFrame) -> pd.DataFrame:
    """Rows for fitting hyperparameters: complete seasons only, when there are enough.

    Hyperparameters change slowly. Fitting them once per season instead of at every
    retrain saves minutes, and still uses training data only.
    """
    earlier = obs[obs["Year"] < obs["Year"].max()]
    return earlier if earlier["RaceId"].nunique() >= MIN_HYPER_RACES else obs


# ---------- the maths ----------
def kalman_update(mean: np.ndarray, cov: np.ndarray, h: np.ndarray, y: np.ndarray,
                  r: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """CORRECT step for observations y = h @ state + noise (noise variances r).

    Returns the new mean, the new covariance and log p(y | everything before), the
    'innovation' log-likelihood used to fit the hyperparameters.
    """
    pht = cov @ h.T
    s = h @ pht + np.diag(r)                     # how uncertain the predicted y is
    innovation = y - h @ mean                    # surprise: seen minus expected
    factor = linalg.cho_factor(s, lower=True)
    solved = linalg.cho_solve(factor, np.column_stack([innovation, pht.T]))
    alpha, gain_t = solved[:, 0], solved[:, 1:]  # S^-1 v  and  S^-1 H P
    new_mean = mean + pht @ alpha                # = mean + K v,  K = P H' S^-1
    new_cov = cov - pht @ gain_t                 # = P - K H P
    new_cov = 0.5 * (new_cov + new_cov.T)        # keep it exactly symmetric
    logdet = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
    loglik = -0.5 * (float(innovation @ alpha) + logdet + len(y) * LOG_2PI)
    return new_mean, new_cov, loglik


class StrengthFilter:
    """A joint Kalman filter over every car (team) and driver skill.

    State order = order of first appearance, so adding later races never changes the
    numbers for earlier ones. Cars and drivers not yet seen sit at their prior and do
    not drift until they first appear. Use a fresh filter for each run.
    """

    def __init__(self, params: StrengthParams) -> None:
        self.params = params
        self.keys: list[tuple[str, str]] = []        # ("car", TeamId) or ("skill", DriverId)
        self.index: dict[tuple[str, str], int] = {}
        self.mean = np.zeros(0)
        self.cov = np.zeros((0, 0))
        self.active = np.zeros(0, dtype=bool)
        self.is_car = np.zeros(0, dtype=bool)
        self.team_of: dict[str, str] = {}
        self.latest_drivers: list[str] = []
        self.last_date: pd.Timestamp | None = None
        self.last_year: int | None = None
        self.loglik = 0.0
        self.history: list[dict[str, Any]] = []
        self.predictions = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # ----- running -----
    def run(self, obs: pd.DataFrame, record: bool = True,
            keep_history: bool = False) -> StrengthFilter:
        """Filter every race of an observation_table, oldest first."""
        return self.run_races(prepare(obs), record, keep_history)

    def run_races(self, races: list[RaceObs], record: bool = True,
                  keep_history: bool = False) -> StrengthFilter:
        """Filter prepared races. predictions = one-step-ahead values (before each race)."""
        for race in races:
            self._ensure(race.teams, race.drivers)
        outputs = []
        for race in races:
            out = self._step(race, record, keep_history)
            if out is not None:
                outputs.append(out)
        if outputs:
            self.predictions = pd.concat(outputs, ignore_index=True)
        return self

    def forecast(self, race: pd.DataFrame) -> pd.DataFrame:
        """One-step-ahead strengths for a future race (rows need RaceId, Date, Year,
        DriverId, TeamId). The filter itself is not changed."""
        twin = self.copy()
        teams = race["TeamId"].astype(str).tolist()
        drivers = race["DriverId"].astype(str).tolist()
        twin._ensure(teams, drivers)
        car, skill = twin._indices(teams, drivers)
        twin._advance(pd.Timestamp(race["Date"].iloc[0]), int(race["Year"].iloc[0]))
        return twin._outputs(int(race["RaceId"].iloc[0]), teams, drivers, car, skill)

    def strengths(self) -> pd.DataFrame:
        """Latest filtered estimates of every car and driver seen (lower Mean = faster)."""
        rows = []
        for (kind, ident), i in self.index.items():
            if self.active[i]:
                team = ident if kind == "car" else self.team_of.get(ident, "")
                rows.append({"Kind": kind, "Id": ident, "TeamId": team,
                             "Mean": float(self.mean[i]), "Sd": math.sqrt(self.cov[i, i]),
                             "InLatestRace": kind == "car" and ident in
                             {self.team_of.get(d) for d in self.latest_drivers}
                             or ident in self.latest_drivers})
        return pd.DataFrame(rows).sort_values(["Kind", "Mean"]).reset_index(drop=True)

    def smooth(self) -> pd.DataFrame:
        """Rauch-Tung-Striebel smoother: estimates that use ALL races, earlier and later.

        For plots ONLY: a smoothed value at race r has seen the races after r, so it
        must never be used as a feature. Needs run(..., keep_history=True).
        """
        if not self.history:
            raise ValueError("Run the filter with keep_history=True before smoothing.")
        steps = self.history
        mean_s, cov_s = steps[-1]["mean"], steps[-1]["cov"]
        tables = [_smoothed_rows(self, steps[-1], mean_s, cov_s)]
        for k in range(len(steps) - 2, -1, -1):
            later = steps[k + 1]
            gain = linalg.solve(later["pred_cov"], steps[k]["cov"], assume_a="pos").T
            mean_s = steps[k]["mean"] + gain @ (mean_s - later["pred_mean"])
            cov_s = steps[k]["cov"] + gain @ (cov_s - later["pred_cov"]) @ gain.T
            tables.append(_smoothed_rows(self, steps[k], mean_s, cov_s))
        return pd.concat(tables[::-1], ignore_index=True)

    def copy(self) -> StrengthFilter:
        """An independent copy of the current state (history not copied)."""
        twin = StrengthFilter(self.params)
        twin.keys, twin.index = list(self.keys), dict(self.index)
        twin.mean, twin.cov = self.mean.copy(), self.cov.copy()
        twin.active, twin.is_car = self.active.copy(), self.is_car.copy()
        twin.team_of, twin.latest_drivers = dict(self.team_of), list(self.latest_drivers)
        twin.last_date, twin.last_year, twin.loglik = self.last_date, self.last_year, self.loglik
        return twin

    # ----- one race: predict, then correct -----
    def _step(self, race: RaceObs, record: bool, keep_history: bool) -> pd.DataFrame | None:
        car, skill = self._indices(race.teams, race.drivers)
        self._advance(race.date, race.year)                     # 1. PREDICT
        self.active[car] = True
        self.active[skill] = True
        out = (self._outputs(race.race_id, race.teams, race.drivers, car, skill)
               if record else None)                             # what we expect, before
        pred_mean, pred_cov = self.mean.copy(), self.cov.copy()
        h, y, r = self._design(car, skill, race.quali, race.practice)
        if len(y):                                              # 2. CORRECT
            self.mean, self.cov, loglik = kalman_update(self.mean, self.cov, h, y, r)
            self.loglik += loglik
        if keep_history:
            self.history.append({"RaceId": race.race_id, "Date": race.date,
                                 "teams": race.teams, "drivers": race.drivers,
                                 "pred_mean": pred_mean, "pred_cov": pred_cov,
                                 "mean": self.mean.copy(), "cov": self.cov.copy()})
        self.team_of.update(zip(race.drivers, race.teams))
        self.latest_drivers = list(race.drivers)
        self.last_date, self.last_year = race.date, race.year
        return out

    def _advance(self, date: pd.Timestamp, year: int) -> None:
        """PREDICT: add drift variance for the time since the last race."""
        if self.last_date is None or self.last_year is None:
            return
        p = self.params
        if year != self.last_year:
            steps = 1.0
            extra = p.q_winter * max(1, year - self.last_year)
            if any(self.last_year < era <= year for era in ERA_STARTS):
                extra += p.q_era                         # new rules: the order may reset
        else:
            steps = max(1.0, (date - self.last_date).days / DAYS_PER_STEP)
            extra = 0.0
        added = np.where(self.is_car, p.q_car * steps + extra, p.q_skill * steps)
        added[~self.active] = 0.0                       # not seen yet: stays at its prior
        self.cov[np.diag_indices_from(self.cov)] += added

    def _design(self, car: np.ndarray, skill: np.ndarray, quali: np.ndarray,
                practice: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Observation matrix h (each row: 1 at the car, 1 at the skill), values y, noise r."""
        cars, skills, values, noise = [], [], [], []
        for gaps, variance in ((quali, self.params.r_quali),
                               (practice, self.params.r_practice)):
            have = ~np.isnan(gaps)
            cars.append(car[have])
            skills.append(skill[have])
            values.append(gaps[have])
            noise.append(np.full(int(have.sum()), variance))
        y = np.concatenate(values)
        h = np.zeros((len(y), len(self.mean)))
        rows = np.arange(len(y))
        h[rows, np.concatenate(cars)] = 1.0
        h[rows, np.concatenate(skills)] = 1.0
        return h, y, np.concatenate(noise)

    def _outputs(self, race_id: int, teams: list[str], drivers: list[str],
                 car: np.ndarray, skill: np.ndarray) -> pd.DataFrame:
        """Current estimates for the drivers of one race."""
        var_car, var_skill = self.cov[car, car], self.cov[skill, skill]
        both = self.cov[car, skill]
        pace_var = np.maximum(var_car + var_skill + 2 * both, 0.0)
        return pd.DataFrame({
            "RaceId": race_id, "DriverId": drivers, "TeamId": teams,
            "car_mean": self.mean[car], "car_sd": np.sqrt(var_car),
            "skill_mean": self.mean[skill], "skill_sd": np.sqrt(var_skill),
            "car_skill_cov": both, "pace_pred": self.mean[car] + self.mean[skill],
            "pace_sd": np.sqrt(pace_var)}, columns=OUTPUT_COLUMNS)

    def _indices(self, teams: list[str], drivers: list[str]) -> tuple[np.ndarray, np.ndarray]:
        car = np.array([self.index[("car", t)] for t in teams], dtype=int)
        skill = np.array([self.index[("skill", d)] for d in drivers], dtype=int)
        return car, skill

    def _ensure(self, teams: list[str], drivers: list[str]) -> None:
        for team in teams:
            if ("car", team) not in self.index:
                self._add("car", team)
        for driver in drivers:
            if ("skill", driver) not in self.index:
                self._add("skill", driver)

    def _add(self, kind: str, ident: str) -> None:
        """A new car or driver enters the state at its prior (no links to anything)."""
        p = self.params
        mean, var = (NEW_CAR_MEAN, NEW_CAR_VAR) if kind == "car" else (p.mu_rookie, p.p_skill)
        n = len(self.mean)
        self.index[(kind, ident)] = n
        self.keys.append((kind, ident))
        self.mean = np.append(self.mean, mean)
        grown = np.zeros((n + 1, n + 1))
        grown[:n, :n] = self.cov
        grown[n, n] = var
        self.cov = grown
        self.active = np.append(self.active, False)
        self.is_car = np.append(self.is_car, kind == "car")


def _smoothed_rows(flt: StrengthFilter, step: dict[str, Any], mean: np.ndarray,
                   cov: np.ndarray) -> pd.DataFrame:
    """Smoothed car and skill estimates for the teams and drivers of one race."""
    rows = []
    for team in dict.fromkeys(step["teams"]):
        i = flt.index[("car", team)]
        rows.append(("car", team, team, mean[i], math.sqrt(max(cov[i, i], 0.0))))
    for driver, team in zip(step["drivers"], step["teams"]):
        i = flt.index[("skill", driver)]
        rows.append(("skill", driver, team, mean[i], math.sqrt(max(cov[i, i], 0.0))))
    table = pd.DataFrame(rows, columns=["Kind", "Id", "TeamId", "Mean", "Sd"])
    table.insert(0, "Date", step["Date"])
    table.insert(0, "RaceId", step["RaceId"])
    return table


# ---------- fitting the hyperparameters ----------
def fit_params(obs: pd.DataFrame, start: StrengthParams | None = None,
               maxiter: int = 60) -> StrengthParams:
    """Hyperparameters that maximise the innovation log-likelihood of the observations.

    "Which drift and noise sizes make the gaps we actually saw least surprising, one
    race ahead?" Variances are optimised as logs, so they stay positive.
    """
    races = prepare(obs)
    begin = (start or StrengthParams()).to_vector()

    def loss(vector: np.ndarray) -> float:
        try:
            params = StrengthParams.from_vector(vector)
            return -StrengthFilter(params).run_races(races, record=False).loglik
        except (linalg.LinAlgError, ValueError):
            return 1e12

    bounds = [(math.log(1e-6), math.log(10.0))] * len(VARIANCES) + [(-2.0, 2.0)]
    result = optimize.minimize(loss, begin, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": maxiter})
    return StrengthParams.from_vector(result.x)


_CACHE: dict[str, StrengthParams] = {}


def fitted_params(obs: pd.DataFrame, maxiter: int = 60) -> StrengthParams:
    """fit_params, remembered per exact data set (models and moments share the work)."""
    key = f"{fingerprint(obs)}:{maxiter}"
    if key not in _CACHE:
        _CACHE[key] = fit_params(obs, maxiter=maxiter)
    return _CACHE[key]


def fingerprint(obs: pd.DataFrame) -> str:
    """A short hash of the observations (same data -> same key)."""
    columns = ["RaceId", "DriverId", "TeamId", "QualiObs", "PracticeObs"]
    hashed = pd.util.hash_pandas_object(obs[columns], index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:24]
