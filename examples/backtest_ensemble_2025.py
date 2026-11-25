"""Phase 11: 2024+2025 backtests, M3, online stacking and calibration, final pipeline.

Run:  python examples/backtest_ensemble_2025.py            # full (M2: 10 seeds)
      python examples/backtest_ensemble_2025.py --fast     # M2: 2 seeds, 100 epochs
      python examples/backtest_ensemble_2025.py --reuse    # reuse cached backtest predictions
Runtime on a laptop CPU: full about 1-2 h (2 seasons x 3 moments x 7 models; M2 dominates);
--fast about 30-50 min; --reuse about 5-15 min (stacking + calibration only).
Raw predictions are cached in ~/.pitwall_cache/predict/ (never in git). 2026 is untouched.
"""
import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd

from pitwall.cache import DEFAULT_CACHE_DIR
from pitwall.plotting import (
    calibration_plot, start_weight_plot, use_pitwall_style, weights_plot,
)
from pitwall.predict import (
    METHODS, MOMENTS, GridSlotBaseline, LightGBMRanker, LogisticModel, MomentChoice,
    NeuralRanker, PipelineConfig, RaceSimulator, StrengthModel, calibrate_forward, gate,
    load_dataset, result_from_predictions, run_backtest, stack_forward, write_report,
)
from pitwall.predict.ensemble import (
    build_stacks, choice_section, choose_pipelines, ensemble_section,
)

CACHE = DEFAULT_CACHE_DIR / "predict" / "backtest_predictions_2024_2025.parquet"
SINGLES = ["B0 grid slot", "B1 logistic", "B2 LightGBM", "M1 strength", "M1-pure strength",
           "M2-full", "M3 simulator"]


def save(ax, path):
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)
    print(f"Saved {path}")


def backtest(data, seeds, epochs, reuse):
    """Walk-forward predictions for 2024 and 2025 (cached as Parquet, never in git)."""
    if reuse and CACHE.exists():
        print(f"Reusing {CACHE}")
        return pd.read_parquet(CACHE)
    models = [GridSlotBaseline(), LogisticModel(), LightGBMRanker(), StrengthModel(),
              StrengthModel(pure=True), NeuralRanker(n_seeds=seeds, max_epochs=epochs),
              RaceSimulator()]
    result = run_backtest(models, data, seasons=(2024, 2025))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(CACHE, index=False)
    return result.predictions


def ensembles(preds):
    """Every method, with and without calibration, per moment (warm-up 2024, eval 2025)."""
    tables, histories = [], []
    for moment in MOMENTS:
        for method in METHODS:
            ens, history = stack_forward(preds, moment, eval_seasons=[2025],
                                         warmup_seasons=[2024], method=method)
            calibrated, _ = calibrate_forward(ens)
            tables += [ens, calibrated]
            histories.append(history)
    return pd.concat(tables, ignore_index=True), pd.concat(histories, ignore_index=True)


def config_from(choices, preds):
    """Turn the chosen pipeline names into a saved FinalPipeline config."""
    moments = {}
    for moment, name in choices.items():
        if name.startswith("ENS-"):
            method = name.removeprefix("ENS-").removesuffix("+cal")
            models = build_stacks(preds, moment)[0].models
            moments[moment] = MomentChoice(models, method, name.endswith("+cal"))
        else:
            moments[moment] = MomentChoice([name])
    return PipelineConfig(moments, note="Chosen on 2025 by win log loss (Phase 11).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--fast", action="store_true", help="M2 with 2 seeds, 100 epochs")
    parser.add_argument("--reuse", action="store_true", help="reuse cached predictions")
    args = parser.parse_args()
    seeds, epochs = (2, 100) if args.fast else (args.seeds, 300)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    use_pitwall_style()

    data = load_dataset()
    preds = backtest(data, seeds, epochs, args.reuse)
    stacked, histories = ensembles(preds)
    singles = preds[preds["Year"] == 2025]
    combined = pd.concat([singles, stacked[stacked["Eval"]].drop(columns="Eval")],
                         ignore_index=True)
    result = result_from_predictions(combined, "B0 grid slot", [2025])

    m3 = gate(result, "M3 simulator", SINGLES[:-1])
    ens_names = sorted(stacked["Model"].unique())
    means = result.summary[result.summary["Model"].isin(ens_names)].groupby("Model")["win_logloss"]
    best_ens = means.mean().idxmin()
    best = gate(result, best_ens, SINGLES)
    choices = choose_pipelines(result.summary)
    for decision in (m3, best):
        print(decision.verdict())
    print("Chosen per moment:", choices)

    extra = ensemble_section(result.summary, ens_names) + choice_section(choices, result.summary)
    print(f"Saved {write_report(result, gates=[m3, best], extra=extra)}")
    print(f"Saved {config_from(choices, preds).save()}")

    quali = histories[(histories["Moment"] == "quali") & (histories["Method"] == "linear")
                      & (histories["Year"] == 2025)].dropna(axis=1, how="all")
    save(weights_plot(quali), "reports/ensemble_weights_2025_quali.png")
    pair = result.predictions[(result.predictions["Moment"] == "quali")
                              & result.predictions["Model"].isin(["ENS-linear", "ENS-linear+cal"])]
    save(calibration_plot(pair, "P_win", "Win"), "reports/calibration_before_after_2025.png")
    rows = data[(data["Moment"] == "quali") & (data["Year"] <= 2025)]
    simulator = RaceSimulator().fit(rows)
    save(start_weight_plot(simulator.start_curve()), "reports/m3_start_weight.png")


if __name__ == "__main__":
    main()
