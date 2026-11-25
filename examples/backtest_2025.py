"""Phase 8: walk-forward backtest of B0, B1 and B2 on the 2025 season, all 3 moments.

Run:  python examples/backtest_2025.py
Needs: pip install -e ".[ml]"  (macOS also needs `brew install libomp` for LightGBM).
Uses the saved dataset (examples/build_dataset.py); builds it first if it's missing
(that first build takes ~40-60 min because of OpenF1's free rate limit).
Runtime of the backtest itself on a laptop CPU: about 3-10 minutes (12 refits per
moment; B2 trains 5 small LightGBM models per refit; 20,000 simulated races per prediction).
Writes reports/backtest.md, appends to reports/experiments.csv and saves a calibration plot.
2026 is locked: it is only run once, at the end of the project, with final=True.
"""
import logging

import matplotlib.pyplot as plt

from pitwall.plotting import calibration_plot, use_pitwall_style
from pitwall.predict import (
    GridSlotBaseline, LightGBMRanker, LogisticModel, build_dataset, load_dataset,
    run_backtest, save_dataset, write_report,
)
from pitwall.predict.dataset import DEFAULT_PATH
from pitwall.sources import F1dbLoader, OpenF1Client, OpenMeteoClient


def dataset():
    """The saved dataset, or a freshly built one (free sources only, no token)."""
    if DEFAULT_PATH.exists():
        return load_dataset()
    data = build_dataset(F1dbLoader(), OpenF1Client(), OpenMeteoClient(), first_year=2018)
    save_dataset(data)
    return data


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    use_pitwall_style()
    data = dataset()
    models = [GridSlotBaseline(), LogisticModel(), LightGBMRanker()]
    result = run_backtest(models, data, seasons=(2025,))

    print("\nMetrics per moment and model (2025, walk-forward)")
    print(result.summary.round(4).to_string(index=False))
    print("\nModel minus baseline (negative = better), 95% paired-bootstrap interval")
    print(result.comparisons.round(4).to_string(index=False))
    print(f"\nSaved {write_report(result)}")

    quali = result.predictions[result.predictions["Moment"] == "quali"]
    for prob, label in (("P_win", "Win"), ("P_podium", "Podium")):
        ax = calibration_plot(quali, prob, label)
        out = f"reports/calibration_{label.lower()}_2025_quali.png"
        ax.figure.tight_layout()
        ax.figure.savefig(out, dpi=150)
        plt.close(ax.figure)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
