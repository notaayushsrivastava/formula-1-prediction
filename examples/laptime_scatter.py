"""FastF1 gallery 'Driver Laptimes Scatterplot', pitwall version.

Run:  python examples/laptime_scatter.py 2024 Monza LEC
      (the event can be a round number or a name: 16, Monza, Italian, ...)
"""
import logging
import sys

import matplotlib.pyplot as plt

import pitwall
from pitwall.plotting import driver_laptimes_scatter, use_pitwall_style


def main(year: int, event: int | str, driver: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # see downloads
    use_pitwall_style()

    session = pitwall.get_session(year, event).load()
    print(f"\n{session.name}: top 10")
    print(session.results[["Position", "Driver", "Team", "Status"]].head(10).to_string(index=False))
    print(f"\n{driver} stints (inferred from pit stops)")
    print(session.laps.pick_drivers(driver).stints().to_string(index=False))

    ax = driver_laptimes_scatter(session, driver)
    out = f"laptimes_{year}_{driver}.png"
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    year = int(args[0]) if len(args) > 0 else 2024
    event = args[1] if len(args) > 1 else "Monza"
    event = int(event) if str(event).isdigit() else event
    driver = args[2] if len(args) > 2 else "LEC"
    main(year, event, driver)
