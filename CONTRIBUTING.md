# Contributing to pitwall-f1

pitwall-f1 is a learning project: readable, typed, tested Python. These rules apply to
every phase from 15 to 30 (and to outside contributions).

## The rules for every phase (15-30)

1. **One algorithm family per phase.** Complete files, offline tests, docs. No
   half-finished modules and no "rest unchanged" patches.
2. **The gate.** A change goes into the engine ONLY if `backtest.gate_v2` passes:
   - the paired race-level bootstrap 95% CI of (new − old) validation log loss on
     **2025** lies entirely below 0 (win and podium log loss);
   - the **deflated p-value** is below 0.05 once the family has 10 or more logged
     trials (the number of trials comes from `reports/experiments.csv`);
   - the candidate's **median CPCV path** log loss is better than the incumbent's.
   The Diebold-Mariano test and the 3-race block bootstrap are shown as context, and
   Benjamini-Hochberg is applied across all gates run in the phase.
   If the gate fails, the code moves to `lab/` with a short write-up of why it failed.
   "No model passes the gate" is a valid, honest result.
3. **2026 is locked** until Phase 30. `LockedSeasonError` must stay; nothing (backtest,
   CPCV, gate, ensemble) may read 2026 results before then.
4. **Log every experiment** (name, params, score) in `reports/experiments.csv`, with its
   per-race losses in `reports/experiments/<experiment_id>.csv` (small text files,
   committed; never Parquet). Run `pitwall-f1 validate` or `run_backtest(...)` and the
   logging happens for you.
5. **Zero cost.** Free data, free open-source libraries, GitHub Actions on the public
   repo, CPU only. Paper trading only, never real money.
6. **Explain it simply.** Each algorithm gets a plain-English explanation in the docs and
   its maths in `docs/methods.md`; the LaTeX race report
   (`pitwall/report/templates/race_report.tex`) keeps a short methodology appendix and
   stays at 8 pages or fewer.

## Before you open a pull request

```bash
pip install -e ".[dev]"
ruff check .
pytest --cov=pitwall          # coverage must stay >= 90% (pyproject: fail_under = 90)
mypy pitwall                  # informational
```

- Tests are offline and use synthetic data (`tests/synthetic_f1.py`,
  `tests/fake_world.py`). Network smoke tests run only with `--run-network`.
- A statistic needs a test on simulated data where the true answer is known (two
  identical models must NOT pass; a planted improvement MUST pass with enough races).
- Every fix to existing code needs a test that would have failed before the fix.
- Never commit data, models or secrets (`.gitignore` covers them). Your F1 TV token
  stays in the `PITWALL_F1TV_TOKEN` environment variable.

## Gating a new model (the workflow)

```bash
pitwall-f1 validate --models "B0 grid slot" "M1 strength"      # walk-forward + CPCV, logged
pitwall-f1 gate "M1 strength" "B0 grid slot" --moment quali     # from the log, no retraining
```

Write the decision (pass or fail, with the reasons the gate prints) in the phase's
report. `reports/final_config.json` is changed only by a phase whose candidate passed.
