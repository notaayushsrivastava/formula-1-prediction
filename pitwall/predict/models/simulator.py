"""M3: a race-level structural Monte Carlo ("what could happen on Sunday").

Each simulated race:
1. Pace: every driver's base score from a ScoreSource (M1-pure by default), redrawn per
   simulation with its uncertainty score_sd.
2. Start (quali moment): key = a_c * pace + (1 - a_c) * grid_key, with
   a_c = sigmoid(c0 + c1 * (circuit_places_moved - 2)) and grid_key = -g * (grid - 1).
   Where cars rarely change places (small places_moved), a_c is small: the grid holds.
3. Chaos: with probability circuit_sc_rate the race is neutralised: keys / kappa_sc
   (more random) and, with probability CRASH_PROB, two cars crash out together.
4. Wet: P(wet) from rain_forecast_mm (logistic fitted on PAST races' RaceWet); keys / kappa_wet.
5. DNFs: mechanical (team rate, shrunk; a shared team shock makes teammates' failures
   correlated) + incidents (driver rate minus team rate, shrunk).
6. Add Gumbel noise, sort: one finishing order. Repeat n_sims times (vectorised).
Parameters (c0, c1, g, kappa_sc, kappa_wet) are fitted on training rows only, by the
likelihood of the real finishing orders: a mixture of Plackett-Luce models over the
(neutralised?, wet?) states, using the TRUE state where it is known (past races).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import logsumexp

from .base import N_SIMS, SMOOTHING, moment_of, prediction_frame
from .logistic import LogisticModel
from .neural import NeuralRanker
from .scores import as_score_source
from .strength_model import StrengthModel

SHRINK_RACES = 5.0     # pseudo-races pulling DNF rates toward the field rate
WINDOW_RACES = 10.0    # the DNF features are rates over the last 10 races
TEAM_SHARE = 0.3       # share of mechanical risk that hits both teammates at once
CRASH_PROB = 0.2       # chance that a neutralised race includes a two-car crash
MOVED_CENTRE = 2.0     # typical circuit_places_moved
DEFAULT_SC_RATE = 0.3
DNF_CLIP = (1e-4, 1 - 1e-4)
SOURCES: dict[str, Callable[[], Any]] = {
    "M1-pure": lambda: StrengthModel(pure=True),
    "M1": lambda: StrengthModel(),
    "B1": lambda: LogisticModel(),
    "M2": lambda: NeuralRanker(),
}
BOUNDS = [(-5.0, 5.0), (-5.0, 5.0), (math.log(1e-3), math.log(5.0)),
          (math.log(1e-3), math.log(20.0)), (math.log(1e-3), math.log(20.0))]


@dataclass(frozen=True)
class SimParams:
    """M3's fitted settings."""

    c0: float = 0.0          # start weight at a typical circuit (sigmoid(c0))
    c1: float = 0.5          # how fast pace takes over as places_moved grows
    g: float = 0.1           # score value of one grid place
    kappa_sc: float = 1.5    # randomness multiplier in a neutralised race (>= 1)
    kappa_wet: float = 1.5   # randomness multiplier in a wet race (>= 1)

    def to_vector(self) -> np.ndarray:
        """Unbounded optimiser coordinates (logs keep g > 0 and kappas > 1)."""
        return np.array([self.c0, self.c1, math.log(self.g), math.log(self.kappa_sc - 1),
                         math.log(self.kappa_wet - 1)])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> SimParams:
        """Inverse of to_vector."""
        return cls(float(v[0]), float(v[1]), math.exp(v[2]), 1 + math.exp(v[3]),
                   1 + math.exp(v[4]))

    def start_weight(self, places_moved: Any) -> np.ndarray:
        """a_c: how much pace (vs grid order) decides the result at a circuit."""
        x = np.asarray(places_moved, float) - MOVED_CENTRE
        return 1.0 / (1.0 + np.exp(-(self.c0 + self.c1 * x)))


@dataclass(frozen=True)
class WetModel:
    """P(wet race) = sigmoid(b0 + b1 * forecast rain mm)."""

    b0: float
    b1: float

    def prob(self, rain_mm: float) -> float:
        return float(1.0 / (1.0 + math.exp(-(self.b0 + self.b1 * float(rain_mm)))))


@dataclass
class Draws:
    """What the simulations produced (positions is the smoothed drivers x places matrix)."""

    positions: np.ndarray
    dnf: np.ndarray           # (n_sims, drivers) bool
    neutralised: np.ndarray   # (n_sims,) bool
    wet: np.ndarray           # (n_sims,) bool


@dataclass
class FitTable:
    """Training races as padded arrays (finishers in finishing order)."""

    pace: np.ndarray     # (races, n)
    grid: np.ndarray     # (races, n): -(grid - 1)
    moved: np.ndarray    # (races,)
    mask: np.ndarray     # (races, n)
    weights: np.ndarray  # (races, 4): P(calm dry), P(sc dry), P(calm wet), P(sc wet)


# ---------- pieces ----------
def fit_wet_model(rows: pd.DataFrame) -> WetModel:
    """Logistic of RaceWet on rain_forecast_mm over PAST races (unknown RaceWet ignored)."""
    races = rows.drop_duplicates("RaceId")
    known = races.dropna(subset=["RaceWet"]) if "RaceWet" in races.columns else races.iloc[:0]
    y = known["RaceWet"].to_numpy(float)
    x = known["rain_forecast_mm"].fillna(0.0).to_numpy(float)
    rate = (y.sum() + 1.0) / (len(y) + 2.0)
    if len(y) < 5 or len(np.unique(y)) < 2:
        return WetModel(_logit(rate), 0.0)

    def loss(b: np.ndarray) -> float:
        z = b[0] + b[1] * x
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.01 * float(b @ b))

    result = optimize.minimize(loss, np.array([_logit(rate), 0.0]), method="L-BFGS-B")
    return WetModel(float(result.x[0]), float(result.x[1]))


def dnf_rates(rows: pd.DataFrame, field_mech: float,
              field_incident: float) -> tuple[np.ndarray, np.ndarray]:
    """(mechanical, incident) DNF chance per driver, shrunk toward the field rates."""
    mech = rows["team_mech_dnf_rate_10"].fillna(field_mech).to_numpy(float)
    total = rows["driver_dnf_rate_10"].fillna(field_mech + field_incident).to_numpy(float)
    incident = np.clip(total - mech, 0.0, 1.0)
    weight = WINDOW_RACES / (WINDOW_RACES + SHRINK_RACES)
    p_mech = weight * mech + (1 - weight) * field_mech
    p_incident = weight * incident + (1 - weight) * field_incident
    return np.clip(p_mech, 0.0, 0.95), np.clip(p_incident, 0.0, 0.95)


def simulate_race(pace: np.ndarray, pace_sd: np.ndarray, grid_key: np.ndarray, a_c: float,
                  p_sc: float, p_wet: float, p_mech: np.ndarray, p_incident: np.ndarray,
                  teams: Any, params: SimParams, n_sims: int,
                  rng: np.random.Generator) -> Draws:
    """n_sims races at once (every array is simulations x drivers)."""
    n = len(pace)
    neutralised = rng.random(n_sims) < p_sc
    wet = rng.random(n_sims) < p_wet
    drawn = pace[None, :] + pace_sd[None, :] * rng.standard_normal((n_sims, n))
    key = a_c * drawn + (1.0 - a_c) * grid_key[None, :]
    calm = np.where(neutralised, params.kappa_sc, 1.0) * np.where(wet, params.kappa_wet, 1.0)
    key = key / calm[:, None] + rng.gumbel(size=(n_sims, n))
    dnf = _dnfs(rng, p_mech, p_incident, teams, neutralised)
    key = np.where(dnf, key - 1e9, key)
    return Draws(_position_matrix(key), dnf, neutralised, wet)


def fit_sim_params(table: FitTable, use_grid: bool, start: SimParams | None = None,
                   maxiter: int = 100) -> SimParams:
    """Maximise the mixture-of-Plackett-Luce likelihood of the training finishing orders."""
    first = start or SimParams()
    if table.mask.size == 0:
        return first
    log_w = np.log(np.clip(table.weights, 1e-12, 1.0))

    def loss(v: np.ndarray) -> float:
        p = SimParams.from_vector(v)
        if use_grid:
            a = p.start_weight(table.moved)[:, None]
            u = a * table.pace + (1 - a) * p.g * table.grid
        else:
            u = table.pace
        kappas = (1.0, p.kappa_sc, p.kappa_wet, p.kappa_sc * p.kappa_wet)
        per_state = np.column_stack([_pl_per_race(u / k, table.mask) for k in kappas]) + log_w
        return -float(np.sum(logsumexp(per_state, axis=1)))

    result = optimize.minimize(loss, first.to_vector(), method="L-BFGS-B", bounds=BOUNDS,
                               options={"maxiter": maxiter})
    return SimParams.from_vector(result.x)


# ---------- the model ----------
class RaceSimulator:
    """M3 as a RaceModel. source: 'M1-pure' (default), 'M1', 'B1', 'M2', or pass
    source_model (any fitted-able model with scores). params: fixed SimParams (skip fit)."""

    def __init__(self, source: str = "M1-pure", n_sims: int = N_SIMS, seed: int = 0,
                 maxiter: int = 100, source_model: Any = None,
                 params: SimParams | None = None, name: str = "M3 simulator") -> None:
        self.name = name
        self.source = source
        self.n_sims = n_sims
        self.seed = seed
        self.maxiter = maxiter
        self.source_model = source_model
        self.fixed_params = params

    def params(self) -> dict[str, Any]:
        """Settings and, once fitted, the fitted parameters."""
        out: dict[str, Any] = {"source": self.source, "n_sims": self.n_sims,
                               "seed": self.seed, "maxiter": self.maxiter}
        if hasattr(self, "sim_params_"):
            out["sim"] = {k: round(v, 5) for k, v in asdict(self.sim_params_).items()}
            out["wet"] = {"b0": round(self.wet_.b0, 4), "b1": round(self.wet_.b1, 4)}
        return out

    def fit(self, train: pd.DataFrame) -> RaceSimulator:
        """Fit the pace source, field rates, wet model, then M3's own parameters."""
        self.moment_ = moment_of(train)
        base = self.source_model if self.source_model is not None else SOURCES[self.source]()
        self.source_ = as_score_source(base.fit(train))
        rows = train.reset_index(drop=True)
        pace, _ = self.source_.training_scores(rows)
        self.field_mech_ = float(rows["MechDNF"].mean())
        self.field_incident_ = float((rows["DNF"] - rows["MechDNF"]).clip(lower=0).mean())
        races = rows.drop_duplicates("RaceId")
        known_sc = races["Neutralised"].dropna() if "Neutralised" in races else pd.Series([])
        self.field_sc_ = float(known_sc.mean()) if len(known_sc) else DEFAULT_SC_RATE
        self.wet_ = fit_wet_model(rows)
        self.sim_params_ = self.fixed_params or fit_sim_params(
            self._fit_table(rows, np.asarray(pace, float)), self.moment_ == "quali",
            maxiter=self.maxiter)
        return self

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """The standard prediction table from the simulations."""
        if moment_of(race) != self.moment_:
            raise ValueError(f"{self.name} was fitted on '{self.moment_}' rows.")
        draws = self.simulate(race)
        return prediction_frame(race, draws.positions, np.clip(draws.dnf.mean(axis=0), *DNF_CLIP))

    def simulate(self, race: pd.DataFrame, n_sims: int | None = None) -> Draws:
        """All simulated races for one race (seeded by (seed, RaceId))."""
        rows = race.reset_index(drop=True)
        pace, spread = self.source_.scores(race)
        a_c, grid_key = self._start(rows)
        p_mech, p_incident = dnf_rates(rows, self.field_mech_, self.field_incident_)
        rng = np.random.default_rng([self.seed, int(rows["RaceId"].iloc[0])])
        rain = float(rows["rain_forecast_mm"].fillna(0.0).iloc[0])
        return simulate_race(np.asarray(pace, float), np.asarray(spread, float), grid_key, a_c,
                             self._sc_prob(rows.iloc[0]), self.wet_.prob(rain), p_mech,
                             p_incident, rows["TeamId"].to_numpy(), self.sim_params_,
                             n_sims or self.n_sims, rng)

    def start_curve(self, moved: Any = None) -> pd.DataFrame:
        """a_c against circuit_places_moved (for plots)."""
        moved = np.linspace(0.0, 5.0, 51) if moved is None else np.asarray(moved, float)
        return pd.DataFrame({"places_moved": moved,
                             "a_c": self.sim_params_.start_weight(moved)})

    # ---------- helpers ----------
    def _start(self, rows: pd.DataFrame) -> tuple[float, np.ndarray]:
        if self.moment_ != "quali":
            return 1.0, np.zeros(len(rows))  # grid unknown: pace decides
        moved = float(rows["circuit_places_moved"].fillna(MOVED_CENTRE).iloc[0])
        grid = rows["grid"].fillna(len(rows)).to_numpy(float)
        a_c = float(self.sim_params_.start_weight(moved))
        return a_c, -self.sim_params_.g * (grid - 1.0)

    def _sc_prob(self, row: pd.Series) -> float:
        if float(row.get("has_sc_history", 0) or 0) >= 1:
            return float(np.clip(row["circuit_sc_rate"], 0.02, 0.98))
        return self.field_sc_

    def _fit_table(self, rows: pd.DataFrame, pace: np.ndarray) -> FitTable:
        races = rows["RaceId"].to_numpy()
        finish = rows["Finish"].to_numpy(float)
        grid = rows["grid"].to_numpy(float) if "grid" in rows else np.zeros(len(rows))
        groups = []
        for race in np.unique(races):
            idx = np.flatnonzero((races == race) & ~np.isnan(finish))
            if len(idx) >= 2:
                groups.append(idx[np.argsort(finish[idx], kind="stable")])
        width = max((len(g) for g in groups), default=0)
        shape = (len(groups), width)
        table = FitTable(np.zeros(shape), np.zeros(shape), np.zeros(len(groups)),
                         np.zeros(shape, bool), np.zeros((len(groups), 4)))
        for r, idx in enumerate(groups):
            first = rows.iloc[idx[0]]
            table.pace[r, :len(idx)] = pace[idx]
            table.grid[r, :len(idx)] = -(np.nan_to_num(grid[idx], nan=len(idx)) - 1.0)
            table.mask[r, :len(idx)] = True
            table.moved[r] = float(first.get("circuit_places_moved", MOVED_CENTRE) or 0.0)
            sc = first.get("Neutralised", np.nan)
            wet = first.get("RaceWet", np.nan)
            ps = float(sc) if pd.notna(sc) else self._sc_prob(first)
            pw = float(wet) if pd.notna(wet) else self.wet_.prob(
                float(first.get("rain_forecast_mm", 0.0) or 0.0))
            table.weights[r] = [(1 - ps) * (1 - pw), ps * (1 - pw), (1 - ps) * pw, ps * pw]
        return table


def _dnfs(rng: np.random.Generator, p_mech: np.ndarray, p_incident: np.ndarray, teams: Any,
          neutralised: np.ndarray) -> np.ndarray:
    """Mechanical (shared team shock + own), incident and crash retirements."""
    n_sims, n = len(neutralised), len(p_mech)
    codes = pd.factorize(pd.Series(teams))[0]
    shared_p = TEAM_SHARE * p_mech
    own_p = np.clip((p_mech - shared_p) / np.maximum(1 - shared_p, 1e-9), 0.0, 1.0)
    team_draw = rng.random((n_sims, int(codes.max()) + 1))
    shared = team_draw[:, codes] < shared_p[None, :]     # same draw for both teammates
    own = rng.random((n_sims, n)) < own_p[None, :]
    incident = rng.random((n_sims, n)) < p_incident[None, :]
    crash = np.zeros((n_sims, n), dtype=bool)
    hit = neutralised & (rng.random(n_sims) < CRASH_PROB)
    if n >= 2:
        first = rng.integers(0, n, n_sims)
        second = (first + 1 + rng.integers(0, n - 1, n_sims)) % n
        rows = np.flatnonzero(hit)
        crash[rows, first[rows]] = True
        crash[rows, second[rows]] = True
    return shared | own | incident | crash


def _position_matrix(keys: np.ndarray) -> np.ndarray:
    """Sorted keys -> smoothed drivers x positions matrix (rows and columns sum to 1)."""
    n_sims, n = keys.shape
    order = np.argsort(-keys, axis=1)
    place = np.empty_like(order)
    np.put_along_axis(place, order, np.arange(n)[None, :], axis=1)
    cells = np.arange(n)[None, :] * n + place
    counts = np.bincount(cells.ravel(), minlength=n * n).reshape(n, n)
    return (counts + SMOOTHING) / (n_sims + SMOOTHING * n)


def _pl_per_race(u: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Plackett-Luce log-likelihood of each race's observed order (one value per race)."""
    u = np.where(mask, u, -np.inf)
    top = np.max(u, axis=1, keepdims=True)
    e = np.exp(u - top)
    denom = np.flip(np.cumsum(np.flip(e, axis=1), axis=1), axis=1)
    safe = np.where(mask, denom, 1.0)
    return np.sum(np.where(mask, u - top - np.log(safe), 0.0), axis=1)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))
