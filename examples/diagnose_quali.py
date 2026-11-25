"""Before/after check of the Phase 14 qualifying fix on the REAL dataset.

    python examples/diagnose_quali.py --old old.parquet --new new.parquet \
        --out reports/diagnose_quali.md

old.parquet is built by the previous code (quali gaps over any session, no input
clipping); new.parquet by this code. Both are backtested walk-forward on a past season
(never 2026) with the same fast models, and the report shows:
  1. the qualifying features' ranges (NaN rate, extremes) before and after,
  2. win / podium / DNF log loss per model, before and after,
  3. the rows that hurt the old B1 DNF predictions the most (to see the cause).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pitwall.predict import run_backtest
from pitwall.predict.metrics import row_log_loss
from pitwall.predict.models import GridSlotBaseline, LightGBMRanker, LogisticModel, StrengthModel
from pitwall.predict.models import base

WATCH = ["grid", "pit_lane_start", "quali_gap_pct", "teammate_quali_gap_pct",
         "practice_pace_gap_pct", "track_temp"]


def feature_table(data: pd.DataFrame) -> pd.DataFrame:
    """NaN rate and spread of the watched features at the quali moment."""
    rows = data[data["Moment"] == "quali"]
    out = []
    for column in WATCH:
        values = pd.to_numeric(rows[column], errors="coerce")
        finite = values[np.isfinite(values)]
        out.append({"Feature": column, "NaN %": 100 * values.isna().mean(),
                    "Inf": int(np.isinf(values).sum()), "Min": finite.min(),
                    "P1": finite.quantile(0.01), "Median": finite.median(),
                    "P99": finite.quantile(0.99), "Max": finite.max()})
    return pd.DataFrame(out)


def models() -> list:
    return [GridSlotBaseline(), LogisticModel(), StrengthModel(), LightGBMRanker()]


def scores(data: pd.DataFrame, season: int, moments: list[str]) -> pd.DataFrame:
    result = run_backtest(models(), data, seasons=(season,), moments=moments, retrain_every=4)
    keep = ["Moment", "Model", "win_logloss", "podium_logloss", "dnf_logloss",
            "winner_hit_rate"]
    return result.summary[keep], result.predictions


def worst_rows(preds: pd.DataFrame, data: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    rows = preds[(preds["Model"] == "B1 logistic") & (preds["Moment"] == "quali")].copy()
    rows["dnf_loss"] = row_log_loss(rows["P_dnf"], rows["DNF"])
    wanted = ["EventName", "Year", *WATCH[:4]]
    extra = [c for c in wanted if c not in rows.columns]
    info = data[data["Moment"] == "quali"][["RaceId", "DriverId", *extra]]
    rows = rows.merge(info, on=["RaceId", "DriverId"], how="left")
    columns = ["Year", "EventName", "Driver", *WATCH[:4], "P_dnf", "DNF", "dnf_loss"]
    return rows.nlargest(n, "dnf_loss")[columns]


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/diagnose_quali.md"))
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    if args.season >= 2026:
        raise SystemExit("2026 is the locked final test: diagnose on 2025 or earlier.")
    old, new = pd.read_parquet(args.old), pd.read_parquet(args.new)
    moments = ["pre", "quali"]
    base.Z_CLIP = None                       # the old behaviour: no input clipping
    old_scores, old_preds = scores(old, args.season, moments)
    base.Z_CLIP = 5.0                        # the fix
    new_scores, _ = scores(new, args.season, moments)
    both = old_scores.merge(new_scores, on=["Moment", "Model"], suffixes=(" before", " after"))
    lines = ["# Diagnostic: the after-qualifying problem", "",
             f"Season {args.season}, walk-forward, retrained every 4 races. 'before' = "
             "Phase 13 code; 'after' = Phase 14 fix (per-session qualifying gaps, capped; "
             "inputs clipped to +-5 standard deviations; inf treated as missing).", "",
             "## Log loss before vs after (lower is better)", "", md(both), "",
             "## Qualifying features before", "", md(feature_table(old)), "",
             "## Qualifying features after", "", md(feature_table(new)), "",
             "## Rows that hurt the old B1 DNF predictions most", "",
             md(worst_rows(old_preds, old)), ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
