"""Small synthetic F1 worlds for offline tests (same columns as build_dataset).

make_dataset: random races where driver d00 is fastest (Phase 8).
make_strength_world: KNOWN car random walks, an era jump in 2022, fixed driver skills
and two drivers swapping teams (Phase 9); the true values are returned for checking.
"""
import math

import numpy as np
import pandas as pd

from pitwall.predict import (
    ALL_FEATURES, DIAGNOSTICS, FEATURES, ID_COLUMNS, LABELS, MOMENTS, OBSERVATIONS,
)
from pitwall.predict.features import ERA_STARTS
from pitwall.predict.strength import StrengthParams

POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
DNF_RATE = 0.08
COLUMNS = ID_COLUMNS + ALL_FEATURES + LABELS + OBSERVATIONS + DIAGNOSTICS

WORLD_CARS = [-1.0, -0.5, 0.0, 0.5, 1.0]  # % gap at the start; lower = faster
WORLD_SKILLS = [-0.4, 0.4, 0.3, -0.3, -0.2, 0.2, 0.1, -0.1, -0.25, 0.25]
WORLD_PARAMS = StrengthParams(q_car=0.01, q_skill=1e-6, q_winter=0.09, q_era=1.0,
                              r_quali=0.0225, r_practice=0.09, mu_rookie=0.0, p_skill=0.1)


def rank_desc(values):
    """1 for the largest value, 2 for the next, ... (ties by position)."""
    return np.argsort(np.argsort(-np.asarray(values, float), kind="stable")) + 1


# ---------- Phase 8 world ----------
def make_dataset(seasons=(2023, 2024, 2025, 2026), races_per_season=6, n_drivers=12,
                 seed=0, deterministic=False):
    """Races every 2 weeks from 1 March. Driver d00 is fastest, the last one slowest.

    deterministic=True: grid follows skill exactly, everyone finishes where they start.
    """
    rng = np.random.default_rng(seed)
    skill = np.linspace(1.5, -1.5, n_drivers)
    rows, race_id = [], 0
    for year in seasons:
        for rnd in range(1, races_per_season + 1):
            race_id += 1
            date = pd.Timestamp(year, 3, 1) + pd.Timedelta(days=14 * (rnd - 1))
            rows += _race(rng, race_id, year, rnd, date, skill, deterministic)
    return _with_moments(rows)


def _race(rng, race_id, year, rnd, date, skill, deterministic):
    n = len(skill)
    noise = np.zeros(n) if deterministic else rng.normal(0, 0.5, n)
    grid = rank_desc(skill + noise)
    if deterministic:
        order, dnf = grid.copy(), np.zeros(n, bool)
    else:
        dnf = rng.random(n) < DNF_RATE
        keys = 1.5 * skill + rng.gumbel(size=n)
        order = rank_desc(np.where(dnf, keys - 1e9, keys))  # retirements go to the back
    rows = []
    for i in range(n):
        row = _row(race_id, year, rnd, date, i, i // 2, int(order[i]), bool(dnf[i]))
        practice = max(0.0, 0.05 * i + float(rng.normal(0, 0.05)))
        quali_gap = 0.1 * (grid[i] - 1) + abs(noise[i]) * 0.05
        row.update({
            "grid": float(grid[i]), "quali_gap_pct": quali_gap,
            "practice_pace_gap_pct": practice, "has_practice_pace": 1.0, "has_openf1": 1.0,
            "driver_avg_finish_5": (i + 1) + float(rng.normal(0, 1.0)),
            "QualiGapObs": quali_gap, "PracticeGapObs": practice,
        })
        rows.append(row)
    return rows


# ---------- Phase 9 world ----------
def make_strength_world(seed=0, seasons=(2020, 2021, 2022, 2023), races_per_season=10,
                        params=WORLD_PARAMS, cars=WORLD_CARS):
    """Returns (dataset, truth). truth = {'cars': RaceId/TeamId/Car table,
    'skills': {DriverId: skill}, 'params': the true StrengthParams}."""
    rng = np.random.default_rng(seed)
    cars = np.array(cars, float)
    skills = np.array(WORLD_SKILLS, float)
    rows, car_truth, race_id, previous_year = [], [], 0, None
    for year in seasons:
        if previous_year is not None:
            cars += rng.normal(0, math.sqrt(params.q_winter), len(cars))
            if year in ERA_STARTS:
                cars += rng.normal(0, math.sqrt(params.q_era), len(cars))  # order reset
        previous_year = year
        teams = _teams_for(year, len(skills))
        for rnd in range(1, races_per_season + 1):
            race_id += 1
            date = pd.Timestamp(year, 3, 1) + pd.Timedelta(days=14 * (rnd - 1))
            if rnd > 1:
                cars += rng.normal(0, math.sqrt(params.q_car), len(cars))
            pace = cars[teams] + skills
            rows += _world_race(rng, race_id, year, rnd, date, teams, pace, params)
            car_truth += [{"RaceId": race_id, "TeamId": f"t{k}", "Car": float(c)}
                          for k, c in enumerate(cars)]
    truth = {"cars": pd.DataFrame(car_truth),
             "skills": {f"d{i:02d}": float(s) for i, s in enumerate(skills)},
             "params": params}
    return _with_moments(rows), truth


def _teams_for(year, n_drivers):
    """Two drivers per team; from 2022 d00 and d02 swap teams."""
    teams = np.arange(n_drivers) // 2
    if year >= 2022:
        teams[0], teams[2] = 1, 0
    return teams


def _world_race(rng, race_id, year, rnd, date, teams, pace, params):
    n = len(pace)
    quali = pace + rng.normal(0, math.sqrt(params.r_quali), n)
    practice = (pace + rng.normal(0, math.sqrt(params.r_practice), n) if year >= 2023
                else np.full(n, np.nan))
    order = rank_desc(-2.0 * pace + rng.gumbel(size=n))  # lower pace = faster
    grid = rank_desc(-quali)
    rows = []
    for i in range(n):
        row = _row(race_id, year, rnd, date, i, int(teams[i]), int(order[i]), False)
        practice_gap = practice[i] - np.nanmin(practice) if year >= 2023 else np.nan
        row.update({
            "grid": float(grid[i]), "quali_gap_pct": float(quali[i] - quali.min()),
            "practice_pace_gap_pct": 0.0 if np.isnan(practice_gap) else practice_gap,
            "has_practice_pace": float(not np.isnan(practice_gap)),
            "QualiGapObs": float(quali[i] - quali.min()), "PracticeGapObs": practice_gap,
        })
        rows.append(row)
    return rows


# ---------- shared ----------
def _row(race_id, year, rnd, date, i, team, order, dnf):
    """One driver's row with every column; features 0 unless set later."""
    finish = np.nan if dnf else float(order)
    row = {feature: 0.0 for feature in ALL_FEATURES}
    row.update({
        "RaceKey": f"{year}-{rnd:02d}", "RaceId": race_id, "Year": year, "Round": rnd,
        "Date": date, "EventName": f"Race {rnd}", "CircuitId": f"c{rnd}",
        "DriverId": f"d{i:02d}", "Driver": f"D{i:02d}", "TeamId": f"t{team}",
        "Number": i + 1, "experience": float(race_id), "age": 25.0, "era": 3.0,
        "Finish": finish, "FinishOrder": order, "Win": int(finish == 1),
        "Podium": int(finish <= 3), "PointsFinish": int(finish <= 10), "DNF": int(dnf),
        "MechDNF": int(dnf),
        "Points": float(POINTS[order - 1]) if not dnf and order <= 10 else 0.0,
        "CircuitCluster": "mixed", "RaceWet": 0.0, "Neutralised": 0.0, "RaceRainMm": np.nan,
    })
    return row


def _with_moments(rows):
    """Copy every row to the 3 moments, hiding features not known yet (like the dataset)."""
    base = pd.DataFrame(rows)
    frames = []
    for moment in MOMENTS:
        frame = base.copy()
        frame["Moment"] = moment
        hidden = [c for c in ALL_FEATURES if c not in FEATURES[moment]]
        frame[hidden] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[COLUMNS]
