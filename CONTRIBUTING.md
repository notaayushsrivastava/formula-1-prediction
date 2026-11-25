# Contributing to `pitwall-f1`

Thanks for your interest in contributing! This project aims to stay small, readable, typed, and well-tested.

## Development Setup

Requires **Python 3.11+** (CI tests against 3.11, 3.12, and 3.13).

```bash
git clone https://github.com/notaayushsrivastava/formula-1-prediction.git
cd formula-1-prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Optional but recommended on Linux for a smaller PyTorch download:
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
```

## Running Checks

Before opening a pull request, run the same checks as [`.github/workflows/tests.yml`](.github/workflows/tests.yml):

```bash
# Linter (ruff: E, F, W; line-length 100)
ruff check .

# Full offline test suite with coverage (minimum 75%)
pytest --cov=pitwall --cov-report=term-missing

# Type checking (informational in CI)
mypy pitwall
```

### Notes on Tests
- **Offline by default**: Tests marked with `@pytest.mark.network` are skipped unless you pass `pytest --run-network`. Offline tests also clear any real F1 TV token from the environment automatically.
- **Headless plotting**: `tests/conftest.py` sets `matplotlib` to the `Agg` backend so plot tests run without a display.
- **LaTeX report tests**: `pytest tests/test_phase13.py -v` compiles a sample race-preview PDF and requires `lualatex` (e.g. TeX Live or MiKTeX).

## Pull Request Workflow

1. Create a branch from `main` for your change (`git checkout -b my-feature`).
2. Keep changes focused and add or update tests for new behavior or bug fixes.
3. Use clear, descriptive commit messages.
4. Ensure `ruff check .` and `pytest` pass locally.
5. Open a pull request against `main` describing what changed and why.

## Releases

Maintainers can follow [docs/RELEASING.md](docs/RELEASING.md) for cutting releases and publishing to PyPI / TestPyPI.
