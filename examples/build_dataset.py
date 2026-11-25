"""Phase 7: build the prediction dataset from free sources (f1db + OpenF1 + Open-Meteo).

Run:  python examples/build_dataset.py 2018
The first run downloads f1db once and makes ~11 OpenF1 requests per 2023+ race at
OpenF1's free limit (30/min): expect 40-60 minutes. Later runs read the cache.
No F1 TV token is used.
"""
import logging
import sys

from pitwall.predict import build_dataset, save_dataset
from pitwall.sources import F1dbLoader, OpenF1Client, OpenMeteoClient


def main(first_year: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # URLs only
    data = build_dataset(F1dbLoader(), OpenF1Client(), OpenMeteoClient(),
                         first_year=first_year)
    path = save_dataset(data)
    rows = data[data["Moment"] == "quali"]
    print(f"\nSaved {len(data)} rows ({rows['RaceId'].nunique()} races x 3 moments) to {path}")
    summary = rows.groupby("Year").agg(Races=("RaceId", "nunique"),
                                       OpenF1=("has_openf1", "mean"),
                                       Forecast=("has_forecast", "mean"))
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2018)
