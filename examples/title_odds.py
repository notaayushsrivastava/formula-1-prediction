"""Monte Carlo title odds.

Run:  python examples/title_odds.py            # 2026, as of now
      python examples/title_odds.py 2021 21    # before the 2021 finale
"""
import logging
import sys

import matplotlib.pyplot as plt

import pitwall
from pitwall.analysis import simulate_championship
from pitwall.plotting import title_odds_plot, use_pitwall_style


def main(year: int, after_round: int | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # see downloads
    use_pitwall_style()
    season = pitwall.Season(year, after_round=after_round).load()

    print("\n" + pitwall.title_fight_summary(season))
    odds = simulate_championship(season)
    shown = odds[odds["TitleChance"] > 0].copy()
    shown["TitleChance"] = shown["TitleChance"].map("{:.1%}".format)
    shown["ExpectedPoints"] = shown["ExpectedPoints"].round(1)
    print(shown.to_string(index=False))

    ax = title_odds_plot(season, odds)
    suffix = f"_after_r{after_round}" if after_round else ""
    out = f"title_odds_{year}{suffix}.png"
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    year = int(args[0]) if args else 2026
    after_round = int(args[1]) if len(args) > 1 else None
    main(year, after_round)
