"""Phase 4: real tyre compounds and track status from F1 Live Timing.

Needs YOUR OWN F1 TV Access (or higher) token in the environment:
    export PITWALL_F1TV_TOKEN='...'        (Windows: set PITWALL_F1TV_TOKEN=...)
Run:  python examples/live_timing_demo.py 2024 Monza LEC
"""
import logging
import sys

import matplotlib.pyplot as plt

import pitwall
from pitwall.analysis import degradation_table
from pitwall.plotting import driver_laptimes_scatter, use_pitwall_style


def main(year: int, event: int | str, driver: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # URLs only, never the token
    use_pitwall_style()
    print(pitwall.token_status())

    session = pitwall.get_session(year, event).load()
    pitwall.load_live_timing(session)
    laps = session.laps.pick_drivers(driver)

    print(f"\n{driver} stints (from F1 Live Timing)")
    stints = laps.groupby(["Stint", "Compound"], as_index=False).agg(
        FirstLap=("LapNumber", "min"), LastLap=("LapNumber", "max"),
        TyreLifeAtEnd=("TyreLife", "max"), Fresh=("FreshTyre", "first"))
    print(stints.to_string(index=False))

    print("\nTrack status changes")
    print(session.track_status.to_string(index=False))
    print(f"\nGreen-flag laps: {len(laps.pick_green())} of {len(laps)}")

    print(f"\n{driver} degradation by stint (green laps, real tyre life)")
    print(degradation_table(laps).round(3).to_string(index=False))

    ax = driver_laptimes_scatter(session, driver)
    out = f"laptimes_compounds_{year}_{driver}.png"
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    year = int(args[0]) if args else 2024
    event = args[1] if len(args) > 1 else "Monza"
    event = int(event) if str(event).isdigit() else event
    driver = args[2] if len(args) > 2 else "LEC"
    main(year, event, driver)
