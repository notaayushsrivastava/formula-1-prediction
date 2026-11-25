"""Phase 14: odds maths, staking, market sources, engine vs market, paper trading.
Offline: fake prices, fake Polymarket responses, fake races."""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

import fake_world as fw
from pitwall import cli
from pitwall.errors import DataNotAvailableError
from pitwall.market import (
    ManualCsvSource, MarketDataError, PolymarketSource, Roster, StakePolicy, TicketExistsError,
    american_from_decimal, benter_walk_forward, decimal_from_american,
    decimal_from_fractional, devig, expected_value, fit_benter, growth_rate,
    kelly_multi, kelly_single, load_prices, make_ticket, market_snapshot, overround,
    paper_stats, race_table, risk_of_ruin, save_ticket, score_vs_market, settle, shin_z,
    stakes, to_decimal, write_template,
)
from pitwall.market.paper import paper_stats as _stats  # noqa: F401  (same object)
from pitwall.predict.models.base import prediction_frame
from pitwall.predict.records import RecordMeta, make_record, record_frame

H = dt.timedelta(hours=1)


# ---------- odds ----------
def test_formats():
    assert decimal_from_fractional("5/2") == 3.5 and decimal_from_fractional("Evens") == 2.0
    assert decimal_from_american(250) == 3.5 and decimal_from_american(-200) == 1.5
    assert american_from_decimal(3.5) == pytest.approx(250)
    assert american_from_decimal(1.5) == pytest.approx(-200)
    assert to_decimal("0.25", "probability") == 4.0
    with pytest.raises(ValueError):
        to_decimal("1.0", "decimal")
    with pytest.raises(ValueError):
        decimal_from_american(50)


@pytest.mark.parametrize("method", ["multiplicative", "additive", "power", "shin"])
def test_devig_sums_to_one_and_keeps_order(method):
    odds = [2.0, 3.5, 5.0, 10.0, 26.0]
    fair = devig(odds, method)
    assert fair.sum() == pytest.approx(1.0)
    assert list(np.argsort(-fair)) == [0, 1, 2, 3, 4]
    assert (fair < 1 / np.array(odds)).all()  # every price had margin in it


def test_power_and_shin_shrink_longshots_more():
    odds = [1.6, 3.5, 7.0, 15.0, 34.0]  # a 15% margin
    plain = devig(odds, "multiplicative")
    for method in ("power", "shin"):
        fair = devig(odds, method)
        assert fair[-1] < plain[-1] and fair[0] > plain[0]


def test_no_margin_means_no_change():
    fair_odds = 1 / np.array([0.5, 0.3, 0.2])
    assert shin_z(1 / fair_odds) == 0.0
    assert devig(fair_odds, "shin") == pytest.approx([0.5, 0.3, 0.2])
    assert overround(fair_odds) == pytest.approx(0.0)
    assert overround([3.0, 3.0, 3.0, 3.0], places=1) == pytest.approx(1 / 3)


def test_top_n_market_sums_to_places():
    odds = [1.3, 1.6, 2.2, 4.0, 7.0, 12.0]
    assert devig(odds, "shin", places=3).sum() == pytest.approx(3.0)


def test_expected_value():
    assert expected_value(0.5, 2.5) == pytest.approx(0.25)


# ---------- staking ----------
def test_kelly_single_by_hand():
    assert kelly_single(0.5, 3.0) == pytest.approx(0.25)
    assert kelly_single(0.2, 3.0) == 0.0


@pytest.mark.parametrize("seed", range(5))
def test_kelly_multi_matches_a_brute_force_optimum(seed):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(5))
    odds = (1 / p) * rng.uniform(0.7, 1.4, 5) + 0.05
    odds = np.maximum(odds, 1.05)
    f = kelly_multi(p, odds)
    cons = ({"type": "ineq", "fun": lambda x: 0.999 - x.sum()},)
    best = minimize(lambda x: -growth_rate(x, p, odds), np.full(5, 0.01), bounds=[(0, 1)] * 5,
                    constraints=cons)
    assert growth_rate(f, p, odds) >= -best.fun - 1e-6
    assert (f >= 0).all() and f.sum() <= 1 + 1e-9


def test_kelly_growth_beats_half_and_double():
    p, odds = np.array([0.6]), np.array([2.0])
    f = kelly_multi(p, odds)
    g = growth_rate(f, p, odds)
    assert g > growth_rate(f / 2, p, odds) and g > growth_rate(np.minimum(2 * f, 0.99), p, odds)


def test_stake_policy_filters_and_caps():
    policy = StakePolicy()
    p_model = np.array([0.40, 0.30, 0.20, 0.10])
    p_market = np.array([0.30, 0.31, 0.25, 0.14])
    odds = 1 / p_market * 0.95
    out = stakes(p_model, p_market, odds, policy)
    assert out[1] == out[2] == out[3] == 0          # no edge: no stake
    assert 0 < out[0] <= policy.max_bet
    assert stakes(np.full(4, 0.9), np.full(4, 0.1), np.full(4, 9.0), policy,
                  exclusive=False).sum() == pytest.approx(policy.max_race)
    with pytest.raises(ValueError):
        StakePolicy(kelly_fraction=0)


# ---------- sources ----------
ROSTER = {"max-verstappen": "VER", "lando-norris": "NOR", "sergio-perez": "PER",
          "andrea-kimi-antonelli": "ANT"}


def test_roster_matches_ids_codes_names_and_surnames():
    roster = Roster(dict(ROSTER))
    assert roster.match("max-verstappen") == "max-verstappen"
    assert roster.match("NOR") == "lando-norris"
    assert roster.match("Sergio Pérez") == "sergio-perez"
    assert roster.match("Kimi Antonelli") == "andrea-kimi-antonelli"
    assert roster.match("Verstappen") == "max-verstappen"
    assert roster.match("Senna") is None


def _csv(tmp_path, body):
    folder = tmp_path / "markets"
    folder.mkdir(exist_ok=True)
    (folder / "2026-06.csv").write_text(
        "# comment\ntimestamp_utc,market,driver,price,price_kind,source,kind\n" + body)
    return ManualCsvSource(folder)


def test_manual_csv_parses_every_format(tmp_path):
    source = _csv(tmp_path, "2026-06-12T10:00Z,win,VER,2.5,decimal,BookA,bookmaker\n"
                            "2026-06-12T10:00Z,win,Norris,5/2,fractional,BookA,bookmaker\n"
                            "2026-06-12T10:00Z,win,PER,+900,american,BookA,bookmaker\n"
                            "2026-06-12T10:00Z,win,ANT,,decimal,BookA,bookmaker\n")
    prices = source.get_prices(2026, 6, Roster(dict(ROSTER)))
    assert prices["Odds"].tolist() == [2.5, 3.5, 10.0]
    assert str(prices["Timestamp"].iloc[0].tz) == "UTC"


def test_manual_csv_reports_every_problem(tmp_path):
    source = _csv(tmp_path, "2026-06-12T10:00Z,win,VER,0.9,decimal,BookA,bookmaker\n"
                            "2026-06-12T10:00Z,winner,NOR,3,decimal,BookA,bookmaker\n"
                            "2026-06-12T10:00Z,win,Senna,3,decimal,BookA,bookmaker\n")
    with pytest.raises(MarketDataError) as err:
        source.get_prices(2026, 6, Roster(dict(ROSTER)))
    text = str(err.value)
    assert "line 2" in text and "line 3" in text and "Senna" in text


def test_template_is_never_overwritten(tmp_path):
    path = write_template(2026, 6, Roster(dict(ROSTER)), tmp_path)
    assert "VER" in path.read_text()
    with pytest.raises(MarketDataError):
        write_template(2026, 6, Roster(dict(ROSTER)), tmp_path)
    assert ManualCsvSource(tmp_path).get_prices(2026, 6, Roster(dict(ROSTER))).empty


class FakePolyHttp:
    def __init__(self):
        self.calls = []

    def get(self, url, params, max_age=None):
        self.calls.append((url, params))
        if url.endswith("/events/slug/abu-dhabi-gp-winner"):
            return {"markets": [
                {"groupItemTitle": "Max Verstappen", "clobTokenIds": '["111", "112"]'},
                {"question": "Will Lando Norris win the Abu Dhabi Grand Prix?",
                 "clobTokenIds": '["221", "222"]'},
                {"groupItemTitle": "Other", "clobTokenIds": '["331", "332"]'}]}
        if url.endswith("/prices-history"):
            p = {"111": 0.5, "221": 0.25}[params["market"]]
            return {"history": [{"t": 1_780_000_000, "p": p}, {"t": 1_780_003_600, "p": p + 0.05}]}
        if url.endswith("/public-search"):
            return {"events": [{"slug": "abu-dhabi-gp-winner", "title": "Abu Dhabi GP Winner",
                                "closed": False}]}
        return None


def test_polymarket_reads_yes_prices_by_slug(tmp_path):
    index = tmp_path / "polymarket.csv"
    index.write_text("year,round,event_slug\n2026,24,abu-dhabi-gp-winner\n")
    source = PolymarketSource(http=FakePolyHttp(), index=index)
    prices = source.get_prices(2026, 24, Roster(dict(ROSTER)))
    assert set(prices["DriverId"]) == {"max-verstappen", "lando-norris"}
    assert prices["Odds"].iloc[0] == pytest.approx(2.0)
    assert (prices["Kind"] == "prediction_market").all()
    assert source.get_prices(2026, 23, Roster(dict(ROSTER))).empty  # no slug: nothing
    assert source.find_events("Abu Dhabi")[0]["slug"] == "abu-dhabi-gp-winner"


def test_load_prices_needs_some_source(tmp_path):
    with pytest.raises(DataNotAvailableError, match="template"):
        load_prices(2026, 6, Roster(dict(ROSTER)), [ManualCsvSource(tmp_path)])


# ---------- engine vs market ----------
def _prices(rows):
    df = pd.DataFrame(rows, columns=["Timestamp", "Market", "DriverId", "Odds", "Source", "Kind"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    return df


BOOK = [("2026-06-12T10:00Z", "win", d, o, "BookA", "bookmaker")
        for d, o in zip(ROSTER, [2.0, 3.0, 8.0, 12.0])]


def test_prices_after_the_prediction_are_never_used():
    later = [("2026-06-13T10:00Z", "win", "max-verstappen", 50.0, "BookA", "bookmaker")]
    prices = _prices(BOOK + later)
    snap = market_snapshot(prices, "win", "2026-06-12T12:00Z")
    assert snap.set_index("DriverId").loc["max-verstappen", "best_odds"] == 2.0
    assert market_snapshot(prices, "win", "2026-06-12T09:00Z").empty


def test_snapshot_averages_sources_and_skips_thin_books():
    other = [("2026-06-12T11:00Z", "win", d, o, "BookB", "bookmaker")
             for d, o in zip(ROSTER, [2.2, 2.8, 9.0, 11.0])]
    thin = [("2026-06-12T11:00Z", "win", "max-verstappen", 3.0, "Tiny", "bookmaker")]
    snap = market_snapshot(_prices(BOOK + other + thin), "win", "2026-06-12T12:00Z")
    assert snap["p_market"].sum() == pytest.approx(1.0)
    row = snap.set_index("DriverId").loc["max-verstappen"]
    assert row["sources"] == 2 and row["best_odds"] == 2.2 and row["best_source"] == "BookB"


def _engine_frame(p):
    race = pd.DataFrame({"RaceId": 1, "DriverId": list(ROSTER), "Driver": list(ROSTER.values())})
    n = len(p)
    matrix = np.tile(np.asarray(p)[:, None], (1, n))
    for _ in range(300):
        matrix /= matrix.sum(1, keepdims=True)
        matrix /= matrix.sum(0, keepdims=True)
    matrix[:, 0] = p
    return prediction_frame(race, matrix, np.full(n, 0.05))


def test_race_table_edges():
    snap = market_snapshot(_prices(BOOK), "win", "2026-06-12T12:00Z")
    table = race_table(_engine_frame([0.6, 0.2, 0.15, 0.05]), snap)
    top = table.iloc[0]
    assert top["Driver"] == "VER" and top["edge"] == pytest.approx(top["p_engine"] - top["p_market"])
    assert top["ev"] == pytest.approx(top["p_engine"] * 2.0 - 1)


def _synthetic_races(n_races, engine_informative, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        strength = rng.normal(0, 1, 8)
        truth = np.exp(strength) / np.exp(strength).sum()
        noisy = strength + rng.normal(0, 0.8, 8)
        market = np.exp(noisy) / np.exp(noisy).sum()
        engine = truth if engine_informative else rng.dirichlet(np.ones(8))
        winner = rng.choice(8, p=truth)
        for i in range(8):
            rows.append({"RaceId": r, "Date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=r),
                         "p_engine": engine[i], "p_market": market[i], "Label": int(i == winner)})
    return pd.DataFrame(rows)


def test_benter_finds_information_only_when_it_exists():
    good = fit_benter(_synthetic_races(150, True))
    noise = fit_benter(_synthetic_races(150, False, seed=1))
    assert good.engine_adds_information and good.engine_weight > 0.5
    assert not noise.engine_adds_information
    assert good.loglik >= good.loglik_market_only


def test_benter_walk_forward_uses_only_earlier_races():
    rows = _synthetic_races(14, True)
    out = benter_walk_forward(rows, min_races=10)
    first = out[out["RaceId"] == 0]
    assert first["p_combined"].to_numpy() == pytest.approx(
        (first["p_market"] / first["p_market"].sum()).to_numpy())
    assert out.groupby("RaceId")["p_combined"].sum().to_numpy() == pytest.approx(1.0)


def test_score_vs_market_verdicts():
    rows = _synthetic_races(40, True)
    score = score_vs_market(rows, n_boot=500)
    assert score.races == 40 and score.engine_logloss < score.market_logloss
    assert "engine" in score.verdict


# ---------- paper trading ----------
def _record(p=(0.6, 0.2, 0.15, 0.05), created="2026-06-12T12:00:00+00:00"):
    meta = RecordMeta(2026, 6, 202606, "Test GP", "2026-06-14", "quali", {}, {}, [], [])
    record = make_record(_engine_frame(list(p)), meta)
    record["created_utc"] = created
    record["_stem"] = "2026-06-quali"
    return record


def test_ticket_bets_only_on_value_and_is_never_overwritten(tmp_path):
    ticket = make_ticket(_record(), _prices(BOOK), StakePolicy(), 1000.0)
    bets = {b["Driver"]: b for b in ticket["bets"]}
    assert "VER" in bets and "ANT" not in bets and "NOR" not in bets  # value only
    assert 0 < bets["VER"]["stake"] <= 20.0
    assert "no real money" in ticket["note"].lower()
    save_ticket(ticket, tmp_path)
    with pytest.raises(TicketExistsError):
        save_ticket(ticket, tmp_path)
    with pytest.raises(Exception, match="before the prediction"):
        make_ticket(_record(created="2026-06-01T00:00:00+00:00"), _prices(BOOK),
                    StakePolicy(), 1000.0)


def test_settlement_maths_by_hand():
    ticket = {"stem": "s", "year": 2026, "round": 6, "event": "T", "moment": "quali",
              "market": "win", "bankroll_before": 1000.0,
              "bets": [{"DriverId": "max-verstappen", "Driver": "VER", "stake": 10.0,
                        "best_odds": 2.5},
                       {"DriverId": "lando-norris", "Driver": "NOR", "stake": 4.0,
                        "best_odds": 6.0},
                       {"DriverId": "ghost", "Driver": "GHO", "stake": 3.0, "best_odds": 9.0}]}
    results = pd.DataFrame({"DriverId": ["max-verstappen", "lando-norris"], "Win": [1, 0]})
    closing = pd.DataFrame({"DriverId": ["max-verstappen"], "best_odds": [2.0]})
    done = settle(ticket, results, closing)
    by = {b["Driver"]: b for b in done["bets"]}
    assert by["VER"]["pnl"] == 15.0 and by["NOR"]["pnl"] == -4.0 and by["GHO"]["result"] == "void"
    assert done["pnl"] == 11.0 and done["staked"] == 14.0
    assert by["VER"]["clv"] == pytest.approx(0.25)


def test_paper_stats_and_risk_of_ruin():
    settled = [{"moment": "quali", "pnl": pnl, "staked": 20.0, "bankroll_before": 1000.0,
                "bets": [{"result": "won" if pnl > 0 else "lost", "best_odds": 3.0,
                          "stake": 20.0, "clv": 0.02}]} for pnl in (40.0, -20.0, -20.0, 40.0)]
    stats = paper_stats(settled, "quali", n_boot=500)
    assert stats.races == 4 and stats.pnl == 40.0 and stats.roi == pytest.approx(0.5)
    assert stats.hit_rate == 0.5 and stats.mean_clv == pytest.approx(0.02)
    assert stats.max_drawdown == pytest.approx(40 / 1040)
    assert paper_stats(settled, "pre") is None
    assert risk_of_ruin(np.array([-0.2, -0.3]), horizon=10) == 1.0
    assert risk_of_ruin(np.array([0.01, 0.02])) == 0.0
    assert np.isnan(risk_of_ruin(np.array([0.1])))


# ---------- the whole flow through the CLI ----------
@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {"now": fw.at(2026, 6, "Qualifying", 3 * H), "results_upto": 5, "quali_upto": 6}
    monkeypatch.setattr(cli, "make_sources", lambda openf1=True, weather=True: fw.sources(
        state["results_upto"], state["quali_upto"], weather, openf1))
    monkeypatch.setattr(cli, "utcnow", lambda: state["now"])
    monkeypatch.setattr("pitwall.predict.service.HISTORY_PREDICTIONS", tmp_path / "x.parquet")
    import pitwall.predict.service as service
    real = service.build_live_rows
    monkeypatch.setattr(service, "build_live_rows",
                        lambda *a, **k: real(*a, **{**k, "first_year": 2024,
                                                    "history_from": 2024}))
    import pitwall.report.build as build
    real_build = build.build_live_rows
    monkeypatch.setattr(build, "build_live_rows",
                        lambda *a, **k: real_build(*a, **{**k, "first_year": 2024,
                                                          "history_from": 2024}))
    monkeypatch.setattr("pitwall.market.sources.PolymarketSource.get_prices",
                        lambda self, *a, **k: pd.DataFrame())
    return state


def test_cli_paper_trading_flow(world):
    assert cli.main(["paper", "trade", "2026", "6", "--moment", "quali"]) == cli.EXIT_UNAVAILABLE
    assert cli.main(["predict", "2026", "6", "--moment", "quali"]) == 0
    assert cli.main(["paper", "trade", "2026", "6", "--moment", "quali",
                     "--skip-if-unavailable"]) == 0  # no prices yet: nothing to do
    assert cli.main(["market", "template", "2026", "6"]) == 0
    record = json.loads(Path("predictions/2026-06-quali.json").read_text())
    seen = (pd.Timestamp(record["created_utc"]) - pd.Timedelta(minutes=30)).isoformat()
    lines = ["timestamp_utc,market,driver,price,price_kind,source,kind"]
    for d in record["drivers"]:
        fair = max(0.01, d["P_win"])
        odds = round(1 / fair * (0.6 if d["Driver"] == "ALP" else 1.4), 2)
        lines.append(f"{seen},win,{d['Driver']},{max(odds, 1.05)},decimal,BookA,bookmaker")
    Path("data/markets/2026-06.csv").write_text("\n".join(lines) + "\n")
    assert cli.main(["market", "check", "2026", "6", "--moment", "quali"]) == 0
    assert cli.main(["paper", "trade", "2026", "6", "--moment", "quali"]) == 0
    ticket = json.loads(Path("predictions/paper/2026-06-quali.json").read_text())
    assert ticket["bets"] and all(b["Driver"] != "ALP" for b in ticket["bets"])
    assert cli.main(["paper", "trade", "2026", "6", "--moment", "quali"]) == cli.EXIT_ERROR
    assert cli.main(["paper", "trade", "2026", "6", "--moment", "quali",
                     "--skip-if-exists"]) == 0
    assert cli.main(["report", "2026", "6", "--moment", "quali", "--tex-only",
                     "--out", "r.pdf"]) == 0
    assert "Engine vs market" in Path("r.tex").read_text()
    assert cli.main(["paper", "settle"]) == 0           # race not run yet
    assert not Path("predictions/paper/2026-06-quali.settled.json").exists()
    world["results_upto"] = 6
    world["now"] = fw.at(2026, 6, "Race", 40 * H)
    assert cli.main(["paper", "settle"]) == 0
    settled = json.loads(Path("predictions/paper/2026-06-quali.settled.json").read_text())
    assert settled["staked"] > 0
    assert cli.main(["paper", "report"]) == 0
    text = Path("reports/market.md").read_text()
    assert "no real money" in text.lower() and "quali" in text and "Paper trading" in text


def test_record_frame_round_trip_used_by_tickets():
    assert set(record_frame(_record())["Driver"]) == set(ROSTER.values())


def test_market_report_with_many_graded_races(tmp_path):
    from pitwall.market import graded_market_rows, write_market_report
    rows = _synthetic_races(12, True)
    tickets, results, settlements = [], [], []
    for race_id, race in rows.groupby("RaceId"):
        drivers = [f"d{i}" for i in range(len(race))]
        table = race.assign(DriverId=drivers, Driver=drivers, best_odds=1 / race["p_market"],
                            best_source="BookA")
        tickets.append({"year": 2025, "round": int(race_id) + 1, "moment": "quali",
                        "market": "win",
                        "market_table": table[["DriverId", "Driver", "p_engine", "p_market",
                                               "best_odds", "best_source"]].to_dict("records")})
        results.append(pd.DataFrame({"Year": 2025, "Round": int(race_id) + 1,
                                     "DriverId": drivers, "Win": race["Label"].to_numpy()}))
        settlements.append({"moment": "quali", "pnl": 5.0 if race_id % 2 else -3.0,
                            "staked": 10.0, "bankroll_before": 1000.0,
                            "bets": [{"result": "won" if race_id % 2 else "lost",
                                      "best_odds": 4.0, "stake": 10.0, "clv": None}]})
    graded = graded_market_rows(tickets, pd.concat(results, ignore_index=True))
    assert graded["RaceId"].nunique() == 12
    path = write_market_report(tickets, settlements, graded, tmp_path / "market.md")
    text = path.read_text()
    assert "Benter" in text and "engine weight" in text and "Risk of ruin" in text
    assert (tmp_path / "market_bankroll.png").exists()
    assert (tmp_path / "market_calibration.png").exists()
    empty = write_market_report([], [], pd.DataFrame(), tmp_path / "empty.md").read_text()
    assert "No finished race" in empty and "Nothing settled" in empty
