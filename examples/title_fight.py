"""Who can still win the title? (FastF1 gallery idea, pitwall version.)

Run:  python examples/title_fight.py            # 2026, as of now
      python examples/title_fight.py 2021 21    # just before the 2021 finale
"""
import logging
import sys

import matplotlib.pyplot as plt

import pitwall
from pitwall.plotting import title_fight_chart, use_pitwall_style


def main(year: int, after_round: int | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # see downloads
    use_pitwall_style()
    season = pitwall.Season(year, after_round=after_round).load()

    print("\nDrivers: " + pitwall.title_fight_summary(season))
    table = pitwall.who_can_still_win(season)
    print(table[table["Status"] != "Out"].to_string(index=False))  # contenders only
    print("\nConstructors: " + pitwall.title_fight_summary(season, constructors=True))

    ax = title_fight_chart(season)
    suffix = f"_after_r{after_round}" if after_round else ""
    out = f"title_fight_{year}{suffix}.png"
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    year = int(args[0]) if args else 2026
    after_round = int(args[1]) if len(args) > 1 else None
    main(year, after_round)
