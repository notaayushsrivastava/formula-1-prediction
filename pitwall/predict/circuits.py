"""Circuit features: what kind of track, how much positions change, how often races are neutralised."""
from __future__ import annotations

import math

import pandas as pd

from .features import asof
from .history import History

FAST_KMH = 215.0      # past median winner speed at or above this: 'high_speed'
SLOW_KMH = 195.0      # at or below this: 'high_downforce' (assumptions; try other values)
SC_PRIOR_RACES = 2.0  # shrinkage strength for safety-car rates
CLUSTERS = ("street", "high_speed", "high_downforce", "mixed")


def circuit_cluster(circuit_type: str, past_kmh: float) -> str:
    """'street', 'high_speed', 'high_downforce' or 'mixed' ('mixed' for a new circuit too)."""
    if str(circuit_type).upper() == "STREET":
        return "street"
    if math.isnan(past_kmh):
        return "mixed"
    if past_kmh >= FAST_KMH:
        return "high_speed"
    return "high_downforce" if past_kmh <= SLOW_KMH else "mixed"


def circuit_stats(hist: History) -> pd.DataFrame:
    """One row per race with running circuit numbers INCLUDING that race (used via asof).

    KmhMedian: median winner speed at the circuit so far.
    MovedMean: mean |finish - grid| of finishers at the circuit so far (low = hard to pass).
    MovedGlobal: the same over every circuit, for circuits with no history yet.
    """
    res = hist.results
    winners = res.loc[res["Finish"] == 1, ["RaceId", "TimeMillis"]]
    finishers = res[res["Finish"].notna() & (res["PitLaneStart"] == 0)]
    moved = (finishers["Finish"] - finishers["Grid"]).abs().groupby(finishers["RaceId"]).mean()
    races = hist.races.merge(winners, on="RaceId", how="inner").sort_values("Date")
    hours = pd.to_numeric(races["TimeMillis"], errors="coerce") / 3_600_000
    races["WinnerKmh"] = pd.to_numeric(races["DistanceKm"], errors="coerce") / hours
    races["Moved"] = races["RaceId"].map(moved)
    by_circuit = races.groupby("CircuitId")
    races["KmhMedian"] = by_circuit["WinnerKmh"].transform(lambda s: s.expanding().median())
    races["MovedMean"] = by_circuit["Moved"].transform(lambda s: s.expanding().mean())
    races["MovedGlobal"] = races["Moved"].expanding().mean()
    return races[["RaceId", "Date", "CircuitId", "KmhMedian", "MovedMean", "MovedGlobal"]]


def circuit_features(target: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Cluster one-hots and places moved for each target row, from earlier races only."""
    kmh = asof(target, stats, "KmhMedian", by="CircuitId")
    moved = asof(target, stats, "MovedMean", by="CircuitId")
    moved = moved.fillna(asof(target, stats, "MovedGlobal"))
    clusters = [circuit_cluster(str(t), float(k)) for t, k in zip(target["CircuitType"], kmh)]
    out = pd.DataFrame(index=target.index)
    out["CircuitCluster"] = clusters
    for name in CLUSTERS[:3]:  # 'mixed' is the baseline: all three are 0
        out[f"circuit_{name}"] = [int(c == name) for c in clusters]
    out["circuit_places_moved"] = moved
    return out


def circuit_sc_rates(races: pd.DataFrame, prior: float = SC_PRIOR_RACES) -> pd.DataFrame:
    """Past share of races with an SC, VSC or red flag at each race's circuit.

    races needs RaceId, Date, CircuitId and Neutralised (1/0, NaN if unknown). With n
    past races there and k neutralised, the rate is (k + prior * field_rate) / (n + prior):
    one safety car in one visit does not make a '100% safety-car track'.
    """
    known = races.dropna(subset=["Neutralised"])
    rows = []
    for race in races.itertuples(index=False):
        past = known[known["Date"] < race.Date]
        if past.empty:
            rows.append((race.RaceId, 0.0, 0))
            continue
        here = past.loc[past["CircuitId"] == race.CircuitId, "Neutralised"]
        rate = (here.sum() + prior * past["Neutralised"].mean()) / (len(here) + prior)
        rows.append((race.RaceId, float(rate), 1))
    frame = pd.DataFrame(rows, columns=["RaceId", "circuit_sc_rate", "has_sc_history"])
    return frame.astype({"RaceId": "int64"})
