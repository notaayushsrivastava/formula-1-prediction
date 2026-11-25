"""B0: the grid-slot baseline. "How often does the car in slot s finish in position p?"

The slot is the order we believe in at each moment:
- 'quali': the starting grid (after penalties);
- 'practice': long-run practice pace for drivers who have it, then the others by form;
- 'pre': recent form (average finish over the last 5 races).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import moment_of, prediction_frame, sinkhorn

LAPLACE = 1.0     # pseudo-count added to every (slot, position) cell
DNF_PRIOR = 2.0   # pseudo-races pulling each slot's DNF rate toward the overall rate


def slots(rows: pd.DataFrame, moment: str) -> np.ndarray:
    """1-based slot of every row within its race (ties broken by DriverId)."""
    if moment == "quali":
        first, second = rows["grid"], pd.Series(0.0, index=rows.index)
    elif moment == "practice":
        has_pace = rows["has_practice_pace"] == 1
        first = (~has_pace).astype(int)  # drivers with a long run come first...
        second = pd.Series(np.where(has_pace, rows["practice_pace_gap_pct"],
                                    rows["driver_avg_finish_5"]), index=rows.index)
    else:
        first, second = rows["driver_avg_finish_5"], -rows["champ_points_before"]
    frame = pd.DataFrame({"RaceId": rows["RaceId"], "first": first, "second": second,
                          "tie": rows["DriverId"]}, index=rows.index)
    ordered = frame.sort_values(["RaceId", "first", "second", "tie"])
    rank = ordered.groupby("RaceId").cumcount() + 1
    return rank.reindex(rows.index).to_numpy(int)


class GridSlotBaseline:
    """B0: P(finish position | slot) learned from history with Laplace smoothing.

    For a new race, each driver's row of the table is looked up by slot, then rows and
    columns are balanced (Sinkhorn) so that every position has exactly one driver.
    P_dnf is the slot's historical DNF rate, shrunk toward the overall rate.
    """

    name = "B0 grid slot"

    def __init__(self, laplace: float = LAPLACE, dnf_prior: float = DNF_PRIOR) -> None:
        self.laplace = laplace
        self.dnf_prior = dnf_prior

    def params(self) -> dict[str, Any]:
        """Settings logged with each experiment."""
        return {"laplace": self.laplace, "dnf_prior": self.dnf_prior}

    def fit(self, train: pd.DataFrame) -> GridSlotBaseline:
        """Count (slot, finishing position) pairs and DNFs per slot."""
        self.moment_ = moment_of(train)
        slot = slots(train, self.moment_) - 1
        place = train["FinishOrder"].to_numpy(int) - 1
        size = int(max(slot.max(), place.max())) + 1
        counts = np.zeros((size, size))
        np.add.at(counts, (slot, place), 1)
        totals = counts.sum(axis=1, keepdims=True)
        self.slot_table_ = (counts + self.laplace) / (totals + self.laplace * size)
        dnf = train["DNF"].to_numpy(float)
        overall = float(dnf.mean())
        dnf_count = np.bincount(slot, weights=dnf, minlength=size)
        starts = np.bincount(slot, minlength=size)
        self.dnf_by_slot_ = (dnf_count + self.dnf_prior * overall) / (starts + self.dnf_prior)
        return self

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """Probabilities for one race of the moment the model was fitted on."""
        if moment_of(race) != self.moment_:
            raise ValueError(f"{self.name} was fitted on '{self.moment_}' rows.")
        n = len(race)
        slot = slots(race, self.moment_) - 1
        table = self._table(n)
        positions = sinkhorn(table[np.minimum(slot, len(table) - 1)][:, :n])
        p_dnf = self.dnf_by_slot_[np.minimum(slot, len(self.dnf_by_slot_) - 1)]
        return prediction_frame(race, positions, p_dnf)

    def _table(self, n: int) -> np.ndarray:
        """The slot table, padded with flat rows/columns if this field is bigger than any seen."""
        size = len(self.slot_table_)
        if n <= size:
            return self.slot_table_
        return np.pad(self.slot_table_, ((0, n - size), (0, n - size)),
                      constant_values=1.0 / n)
