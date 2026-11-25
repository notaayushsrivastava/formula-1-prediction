"""Offline tests for Phase 3: degradation, bootstrap and Monte Carlo."""
import matplotlib

matplotlib.use("Agg")  # draw off-screen; must come before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from pitwall import Laps, Season, Session  # noqa: E402
from pitwall.analysis import (  # noqa: E402
    bootstrap_ci, build_field_model, clean_stint_laps, degradation_table,
    driver_pace_table, pace_difference, simulate_championship,
)
from pitwall.plotting import degradation_plot, title_odds_plot  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------- degradation ----------
def _synthetic_laps(deg=(0.10, 0.05), fuel=0.05, traffic_lap=6):
    """20 laps, pit stop on lap 10. Times follow an exact formula plus one traffic lap."""
    rows = []
    for lap in range(1, 21):
        stint = 1 if lap <= 10 else 2
        age = lap if stint == 1 else lap - 10
        t = 90.0 + deg[stint - 1] * age - fuel * (lap - 1)
        if lap == traffic_lap:
            t += 1.5  # stuck behind a slower car
        rows.append({"Driver": "ALP", "DriverId": "alp", "Team": "Red", "TeamId": "red",
                     "LapNumber": lap, "LapTimeSeconds": t, "Stint": stint,
                     "PitInLap": lap == 10, "PitOutLap": lap == 11})
    return Laps(rows)


def test_theil_sen_recovers_degradation_despite_traffic():
    table = degradation_table(_synthetic_laps())
    assert table["DegSecondsPerLap"].tolist() == pytest.approx([0.10, 0.05])
    assert (table["DegLow"] <= table["DegSecondsPerLap"]).all()
    assert (table["DegSecondsPerLap"] <= table["DegHigh"]).all()


def test_least_squares_is_pulled_by_the_traffic_lap():
    clean = clean_stint_laps(_synthetic_laps())
    stint1 = clean[clean["Stint"] == 1]
    ols_slope = np.polyfit(stint1["TyreAge"], stint1["FuelCorrectedSeconds"], 1)[0]
    assert ols_slope == pytest.approx(0.125)  # 0.10 + 1.5 * (6 - 5) / 60


def test_short_stints_are_skipped():
    assert degradation_table(_synthetic_laps(), min_laps=10).empty


def test_degradation_plot_labels():
    session = Session(2024, 1, source=None)
    session.laps = _synthetic_laps()
    ax = degradation_plot(session, "ALP")
    labels = [line.get_label() for line in ax.get_lines()]
    assert labels == ["Stint 1: +0.100 s/lap", "Stint 2: +0.050 s/lap"]


# ---------- bootstrap ----------
BASE = 90 + 0.3 * np.sin(np.arange(30))  # 30 laps with some natural spread


def _pace_laps():
    """ALP and CHA identical; BRA exactly 1 s slower on every lap."""
    rows = []
    for code, offset in (("ALP", 0.0), ("BRA", 1.0), ("CHA", 0.0)):
        for i, t in enumerate(BASE + offset):
            rows.append({"Driver": code, "DriverId": code.lower(), "Team": "T", "TeamId": "t",
                         "LapNumber": i + 1, "LapTimeSeconds": float(t), "Stint": 1,
                         "PitInLap": False, "PitOutLap": False})
    return Laps(rows)


def test_bootstrap_of_constant_data_has_no_width():
    assert bootstrap_ci([90.0] * 10) == (90.0, 90.0, 90.0)


def test_clear_and_unclear_pace_differences():
    laps = _pace_laps()
    slower = pace_difference(laps, "ALP", "BRA", n_boot=4000)
    assert slower.difference == pytest.approx(1.0) and slower.is_clear
    same = pace_difference(laps, "ALP", "CHA", n_boot=4000)
    assert same.low < 0 < same.high and not same.is_clear


def test_driver_pace_table_orders_fastest_first():
    session = Session(2024, 1, source=None)
    session.laps = _pace_laps()
    session.results = pd.DataFrame({"Driver": ["BRA", "ALP", "CHA"],
                                    "DriverId": ["bra", "alp", "cha"],
                                    "Team": ["T"] * 3, "TeamId": ["t"] * 3})
    table = driver_pace_table(session, n_boot=2000)
    assert table["Driver"].iloc[-1] == "BRA"


# ---------- Monte Carlo ----------
FORM_DRIVERS = [("ALP", "red"), ("BRA", "blue"), ("CHA", "red"), ("DEL", "blue")]
RACE_POINTS = {1: 25.0, 2: 18.0, 3: 15.0, 4: 12.0}


class FakeFormSource:
    """2025 rules, 4 rounds, no sprints. Every race finishes ALP, BRA, CHA, DEL,
    except that DEL retires in round 3."""

    def __init__(self, done_rounds=(1, 2, 3)):
        self.done_rounds = done_rounds

    def get_schedule(self, year):
        return [{"Round": r, "EventName": f"Round {r} Grand Prix", "Date": "2025-01-01",
                 "CircuitId": f"c{r}", "Circuit": f"C{r}", "Locality": "", "Country": "",
                 "Lat": 0.0, "Long": 0.0, "HasSprint": False} for r in (1, 2, 3, 4)]

    def get_season_results(self, year, sprint=False):
        if sprint:
            return []
        return [self._row(rnd, pos, code, team) for rnd in self.done_rounds
                for pos, (code, team) in enumerate(FORM_DRIVERS, start=1)]

    def _row(self, rnd, pos, code, team):
        retired = code == "DEL" and rnd == 3
        return {"Round": rnd, "DriverId": code.lower(), "Driver": code, "FullName": code,
                "TeamId": team, "Team": team.title(), "Number": "0", "GridPosition": pos,
                "Position": None if retired else pos,
                "ClassifiedPosition": "R" if retired else str(pos),
                "Status": "Engine" if retired else "Finished",
                "Points": 0.0 if retired else RACE_POINTS[pos],
                "LapsCompleted": 50, "FastestLapSeconds": None}

    def get_driver_standings(self, year, rnd=None):
        totals = {code: 0.0 for code, _ in FORM_DRIVERS}
        for row in self.get_season_results(year):
            totals[row["Driver"]] += row["Points"]
        ranked = sorted(FORM_DRIVERS, key=lambda d: -totals[d[0]])
        return [{"Round": max(self.done_rounds), "DriverId": code.lower(), "Driver": code,
                 "FullName": code, "TeamId": team, "Team": team.title(), "Position": pos,
                 "Points": totals[code], "Wins": len(self.done_rounds) if code == "ALP" else 0}
                for pos, (code, team) in enumerate(ranked, start=1)]

    def get_constructor_standings(self, year, rnd=None):
        return []


def _form_season(**kwargs):
    return Season(2025, source=FakeFormSource(**kwargs)).load()


def test_dnf_rates_shrink_towards_field_average():
    model = build_field_model(_form_season())
    dnf = dict(zip(model.drivers, model.dnf_prob))
    field = 1 / 12  # one retirement in 12 starts
    assert dnf["ALP"] == pytest.approx((0 + 5 * field) / (3 + 5))
    assert dnf["DEL"] == pytest.approx((1 + 5 * field) / (3 + 5))


def test_title_odds_respect_the_maths():
    # After 3 of 4 rounds: ALP 75, BRA 54, CHA 45, DEL 24; 25 points left.
    odds = simulate_championship(_form_season()).set_index("Driver")
    assert odds["TitleChance"].sum() == pytest.approx(1.0)
    assert odds.loc["CHA", "Status"] == "Out" and odds.loc["CHA", "TitleChance"] == 0
    assert odds.loc["DEL", "TitleChance"] == 0
    # BRA only wins if ALP retires AND BRA wins the race: possible, but unlikely
    assert odds.loc["ALP", "TitleChance"] > odds.loc["BRA", "TitleChance"] > 0


def test_same_seed_same_answer():
    season = _form_season()
    first = simulate_championship(season, n_sims=2000, seed=1)
    second = simulate_championship(season, n_sims=2000, seed=1)
    pd.testing.assert_frame_equal(first, second)


def test_season_over_means_certainty():
    odds = simulate_championship(_form_season(done_rounds=(1, 2, 3, 4)))
    assert odds["Driver"].iloc[0] == "ALP" and odds["TitleChance"].iloc[0] == 1.0


def test_title_odds_plot_shows_only_live_drivers():
    season = _form_season()
    ax = title_odds_plot(season, simulate_championship(season, n_sims=5000))
    assert [t.get_text() for t in ax.get_yticklabels()] == ["ALP", "BRA"]
