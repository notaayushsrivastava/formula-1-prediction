"""Phase 3 race analysis: tyre degradation and 'who was really faster?'.

Run:  python examples/race_analysis.py 2024 Monza LEC PIA
"""
import logging
import sys

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import pitwall
from pitwall.analysis import degradation_table, driver_pace_table, pace_difference
from pitwall.plotting import degradation_plot, pace_ranking_plot, use_pitwall_style


def save(ax: Axes, path: str) -> None:
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)
    print(f"Saved {path}")


def main(year: int, event: int | str, a: str, b: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # see downloads
    use_pitwall_style()
    session = pitwall.get_session(year, event).load()

    print(f"\nTyre degradation, {session.name} (s/lap, fuel-corrected, 95% range)")
    print(degradation_table(session.laps.pick_drivers(a, b)).round(3).to_string(index=False))

    diff = pace_difference(session.laps, a, b)
    verdict = "a clear difference" if diff.is_clear else "too close to call"
    print(f"\n{b} vs {a}: {diff.difference:+.3f} s/lap "
          f"(95% interval {diff.low:+.3f} to {diff.high:+.3f}) -> {verdict}")

    table = driver_pace_table(session)
    print("\nRace pace ranking (median fuel-corrected lap, seconds)")
    print(table.round(3).to_string(index=False))

    tag = f"{year}_r{session.round}"
    save(degradation_plot(session, a), f"degradation_{a}_{tag}.png")
    save(degradation_plot(session, b), f"degradation_{b}_{tag}.png")
    save(pace_ranking_plot(session, table=table), f"pace_ranking_{tag}.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    year = int(args[0]) if args else 2024
    event = args[1] if len(args) > 1 else "Monza"
    event = int(event) if str(event).isdigit() else event
    a = args[2] if len(args) > 2 else "LEC"
    b = args[3] if len(args) > 3 else "PIA"
    main(year, event, a, b)
