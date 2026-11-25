"""Monte Carlo title simulation: turn 'can still win' into 'chance of winning'."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import DataNotAvailableError
from ..points import points_system
from ..season import Season
from ..title_fight import remaining_sessions, who_can_still_win


@dataclass(frozen=True)
class FieldModel:
    """What the simulation believes about each active driver (same order in every array)."""

    drivers: list[str]
    strength: np.ndarray  # relative chance of finishing ahead (higher = faster)
    dnf_prob: np.ndarray  # chance of not finishing a session


def build_field_model(season: Season, form_window: int = 5, sharpness: float = 1.0,
                      prior_races: float = 5.0) -> FieldModel:
    """Strength and DNF chance for every driver who raced in the latest round."""
    rounds = season.completed_rounds
    if not rounds:
        raise DataNotAvailableError(f"No {season.year} race results yet, so no form to use.")
    res = season.results
    drivers = res.loc[res["Round"] == rounds[-1], "Driver"].tolist()
    recent = res[res["Round"].isin(rounds[-form_window:])]
    form = recent.groupby("Driver")["Points"].mean().reindex(drivers).fillna(0.0)
    strength = (form.to_numpy(float) + 0.5) ** sharpness
    return FieldModel(drivers, strength, _dnf_probabilities(res, drivers, prior_races))


def simulate_championship(season: Season, n_sims: int = 20_000, form_window: int = 5,
                          sharpness: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Simulate the rest of the season n_sims times.

    Returns Driver, Points (now), ExpectedPoints, TitleChance (0-1) and Status
    (from the best-case calculator), most likely champion first.
    """
    model = build_field_model(season, form_window, sharpness)
    system = points_system(season.year)
    start = _start_table(season, model.drivers)
    points = np.tile(start["Points"].to_numpy(), (n_sims, 1))  # one row per simulation
    wins = np.tile(start["Wins"].to_numpy(), (n_sims, 1))
    cols = start.index.get_indexer(model.drivers)  # active drivers' columns
    rng = np.random.default_rng(seed)
    last_round = int(season.schedule["Round"].max())

    for rnd, kind in remaining_sessions(season):
        if kind == "sprint":
            scored, _ = _simulate_session(model, system.sprint, 0, n_sims, rng)
        else:
            scored, won = _simulate_session(model, system.race, system.fastest_lap, n_sims, rng)
            if system.double_points_last_round and rnd == last_round:
                scored *= 2
            wins[:, cols] += won  # only Grand Prix wins count for countback
        points[:, cols] += scored
    return _summarise(season, start, points, wins)


def _simulate_session(model: FieldModel, paid: Sequence[int], fastest_lap: int,
                      n_sims: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Points and wins per (simulation, driver) for one race or sprint."""
    n = len(model.drivers)
    keys = np.log(model.strength) + rng.gumbel(size=(n_sims, n))
    dnf = rng.random((n_sims, n)) < model.dnf_prob
    keys[dnf] = -np.inf                        # retirements sort to the back
    order = np.argsort(-keys, axis=1)          # order[s, p] = driver in position p + 1

    by_position = np.zeros(n)
    k = min(n, len(paid))
    by_position[:k] = paid[:k]
    scored = np.zeros((n_sims, n))
    np.put_along_axis(scored, order, by_position[None, :], axis=1)
    scored[dnf] = 0.0                          # retired drivers never score

    rows = np.arange(n_sims)
    if fastest_lap:  # crude guess: a random top-10 finisher sets fastest lap
        who = order[rows, rng.integers(0, min(10, n), n_sims)]
        scored[rows, who] += fastest_lap * ~dnf[rows, who]
    won = np.zeros((n_sims, n))
    winner = order[:, 0]
    won[rows, winner] = ~dnf[rows, winner]
    return scored, won


def _dnf_probabilities(results: pd.DataFrame, drivers: list[str],
                       prior_races: float) -> np.ndarray:
    """Each driver's retirement rate, shrunk towards the field average."""
    dnf = ~results["ClassifiedPosition"].astype(str).str.isdigit()
    field_rate = float(dnf.mean())
    per = dnf.groupby(results["Driver"]).agg(["sum", "count"]).reindex(drivers).fillna(0)
    rate = (per["sum"] + prior_races * field_rate) / (per["count"] + prior_races)
    return rate.to_numpy(float)


def _start_table(season: Season, active: list[str]) -> pd.DataFrame:
    """Current Points and Wins for everyone in the standings, plus any active newcomer."""
    standings = season.driver_standings.set_index("Driver")[["Points", "Wins"]]
    index = standings.index.union(pd.Index(active), sort=False)
    return standings.reindex(index, fill_value=0).astype(float)


def _summarise(season: Season, start: pd.DataFrame,
               points: np.ndarray, wins: np.ndarray) -> pd.DataFrame:
    champion = (points + 1e-3 * wins).argmax(axis=1)  # tie on points -> most wins
    chance = np.bincount(champion, minlength=len(start)) / len(points)
    status = who_can_still_win(season).set_index("Driver")["Status"]
    table = pd.DataFrame({
        "Driver": start.index,
        "Points": start["Points"].to_numpy(),
        "ExpectedPoints": points.mean(axis=0),
        "TitleChance": chance,
        "Status": status.reindex(start.index).fillna("Out").to_numpy(),
    })
    return table.sort_values(["TitleChance", "Points"], ascending=False).reset_index(drop=True)
