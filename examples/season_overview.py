"""Season overview: standings, points per round, races left, and a qualifying result.

Run:  python examples/season_overview.py 2026
      python examples/season_overview.py 2024 Monza
"""
import logging
import sys

import pitwall
from pitwall.plotting import format_laptime


def main(year: int, event: int | str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # see downloads
    season = pitwall.Season(year).load()

    print(f"\n{year} drivers' championship (top 10)")
    cols = ["Position", "Driver", "Team", "Points", "Wins"]
    print(season.driver_standings[cols].head(10).to_string(index=False))

    print(f"\n{year} constructors' championship")
    print(season.constructor_standings[["Position", "Team", "Points", "Wins"]]
          .to_string(index=False))

    print("\nPoints per round (race + sprint), top 5")
    print(season.points_by_round().head(5).to_string())

    left = season.remaining_events
    print(f"\nRounds done: {len(season.completed_rounds)}, left: {len(left)}")
    if not left.empty:
        print(left[["Round", "EventName", "Date", "HasSprint"]].to_string(index=False))

    if event is not None:
        quali = season.session(event, "Q").load()
        table = quali.results[["Position", "Driver", "Team", "BestSeconds",
                               "DeltaToFastestSeconds"]].copy()
        table["BestSeconds"] = table["BestSeconds"].map(format_laptime)
        table["DeltaToFastestSeconds"] = table["DeltaToFastestSeconds"].round(3)
        print(f"\n{quali.name}")
        print(table.to_string(index=False))


if __name__ == "__main__":
    args = sys.argv[1:]
    year = int(args[0]) if args else 2026
    event = args[1] if len(args) > 1 else None
    if event is not None and event.isdigit():
        event = int(event)
    main(year, event)
