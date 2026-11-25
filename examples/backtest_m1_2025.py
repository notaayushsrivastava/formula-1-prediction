"""Phase 9: B0, B1, B2, M1 and M1-pure on 2025 (all 3 moments), the M1 gate, and plots.

Run:  python examples/backtest_m1_2025.py
Needs: pip install -e ".[ml]" and a dataset WITH the Phase 9 observation columns
(rebuild once with: python examples/build_dataset.py 2018 -- cached, a few minutes).
Runtime on a laptop CPU: about 6-15 minutes in total. M1 fits its hyperparameters twice
(~20-60 s each); B2's LightGBM refits are the slowest part.
2026 is never touched: everything here uses seasons up to 2025.
"""
import logging

import matplotlib.pyplot as plt

from pitwall.plotting import (
    car_strength_plot, driver_skill_plot, strength_vs_result, use_pitwall_style,
)
from pitwall.predict import (
    OBSERVATIONS, GridSlotBaseline, LightGBMRanker, LogisticModel, StrengthFilter,
    StrengthModel, gate, load_dataset, observation_table, run_backtest, write_report,
)
from pitwall.predict.strength import fitted_params, hyper_subset


def save(ax, path):
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)
    print(f"Saved {path}")


def plots(data):
    """Strength plots from 2018-2025 only (the smoother sees the whole period: plots only)."""
    rows = data[(data["Moment"] == "quali") & (data["Year"] <= 2025)]
    obs = observation_table(rows)
    params = fitted_params(hyper_subset(obs))
    history = StrengthFilter(params).run(obs, keep_history=True)
    save(car_strength_plot(history.smooth()), "reports/car_strength_2018_2025.png")
    save(driver_skill_plot(history.strengths()), "reports/driver_skill_2025.png")

    last = rows[rows["Date"] == rows["Date"].max()]          # last race of 2025
    before = StrengthFilter(params).run(obs[obs["Date"] < last["Date"].iloc[0]])
    table = before.forecast(last).merge(last[["DriverId", "Driver", "FinishOrder"]],
                                        on="DriverId")
    save(strength_vs_result(table), "reports/strength_vs_result_2025_last.png")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    use_pitwall_style()
    data = load_dataset()
    missing = [c for c in OBSERVATIONS if c not in data.columns]
    if missing:
        raise SystemExit(f"Dataset lacks {missing}: rebuild it with "
                         "python examples/build_dataset.py 2018")

    previous = [GridSlotBaseline(), LogisticModel(), LightGBMRanker()]
    models = [*previous, StrengthModel(), StrengthModel(pure=True)]
    result = run_backtest(models, data, seasons=(2025,))
    names = [m.name for m in previous]
    decisions = [gate(result, "M1 strength", names), gate(result, "M1-pure strength", names)]

    print("\nMetrics per moment and model (2025, walk-forward)")
    print(result.summary.round(4).to_string(index=False))
    for decision in decisions:
        print(f"\n{decision.comparisons.round(4).to_string(index=False)}")
        print(decision.verdict())
    print(f"\nSaved {write_report(result, gates=decisions)}")
    print("\nM1 ranking weights (last refit):")
    print(models[3].coefficients().round(3).to_string())
    plots(data)


if __name__ == "__main__":
    main()
