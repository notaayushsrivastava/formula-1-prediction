"""Phase 10: M2 neural ranker variants vs B0, B1, B2 and M1 on 2025 (all 3 moments).

Run:  python examples/backtest_m2_2025.py            # full: 10 seeds per M2 variant
      python examples/backtest_m2_2025.py --fast     # 2 seeds, 100 epochs: a quick look
      python examples/backtest_m2_2025.py --seeds 5
Needs (Linux: CPU wheel first, it is much smaller):
      pip install torch --index-url https://download.pytorch.org/whl/cpu
      pip install -e ".[ml,nn]"
Runtime on a laptop CPU: full about 30-60 min (5 M2 variants x 36 refits x 10 seeds x
2 trainings); --fast about 10-20 min. 2026 is never touched.
"""
import argparse
import logging

import matplotlib.pyplot as plt

from pitwall.plotting import (
    attribution_plot, calibration_plot, embedding_plot, use_pitwall_style,
)
from pitwall.predict import (
    OBSERVATIONS, GridSlotBaseline, LightGBMRanker, LogisticModel, NeuralRanker,
    StrengthModel, gate, load_dataset, run_backtest, write_report,
)
from pitwall.predict.backtest import strip_outcomes


def save(ax, path):
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)
    print(f"Saved {path}")


def m2_variants(seeds, epochs):
    """M2-full with and without strength inputs, plus three simpler ablations."""
    common = {"n_seeds": seeds, "max_epochs": epochs}
    return [NeuralRanker(**common),                                       # M2-full
            NeuralRanker(use_strength=False, **common),                   # M2-full (no strength)
            NeuralRanker(use_embeddings=False, use_mlp=False, **common),  # M2-linear
            NeuralRanker(use_mlp=False, **common),                        # M2-no-MLP
            NeuralRanker(use_embeddings=False, **common)]                 # M2-no-emb


def explain(data, seeds, epochs):
    """Integrated gradients for the last 2025 race and the embedding map (quali moment)."""
    quali = data[(data["Moment"] == "quali") & (data["Year"] <= 2025)]
    last_date = quali["Date"].max()
    model = NeuralRanker(n_seeds=seeds, max_epochs=epochs)
    model.fit(quali[quali["Date"] < last_date])
    last = quali[quali["Date"] == last_date]
    save(attribution_plot(model.integrated_gradients(strip_outcomes(last))),
         "reports/m2_integrated_gradients_2025_last.png")
    save(embedding_plot(model), "reports/m2_driver_embeddings.png")
    print("\nM2-full linear weights (standardised; monotone ones are <= 0):")
    print(model.coefficients().round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10, help="seeds per M2 variant")
    parser.add_argument("--fast", action="store_true", help="2 seeds, 100 epochs")
    args = parser.parse_args()
    seeds, epochs = (2, 100) if args.fast else (args.seeds, 300)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    use_pitwall_style()
    data = load_dataset()
    if any(c not in data.columns for c in OBSERVATIONS):
        raise SystemExit("Rebuild the dataset first: python examples/build_dataset.py 2018")

    previous = [GridSlotBaseline(), LogisticModel(), LightGBMRanker(), StrengthModel()]
    variants = m2_variants(seeds, epochs)
    result = run_backtest([*previous, *variants], data, seasons=(2025,))
    names = [m.name for m in previous]
    decisions = [gate(result, v.name, names) for v in variants]
    ablation = ("M2-full", [v.name for v in variants[1:]])

    print("\nMetrics per moment and model (2025, walk-forward)")
    print(result.summary[["Moment", "Model", "win_logloss", "podium_logloss"]]
          .round(4).to_string(index=False))
    for decision in decisions:
        print(decision.verdict())
    print(f"\nSaved {write_report(result, gates=decisions, ablation=ablation)}")

    shown = ["B0 grid slot", "M1 strength", "M2-full"]
    quali = result.predictions[(result.predictions["Moment"] == "quali")
                               & result.predictions["Model"].isin(shown)]
    save(calibration_plot(quali, "P_win", "Win"), "reports/calibration_win_2025_quali_m2.png")
    explain(data, seeds, epochs)


if __name__ == "__main__":
    main()
