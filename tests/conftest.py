"""Shared pytest setup: off-screen plotting, opt-in network tests, token safety."""
import matplotlib

matplotlib.use("Agg")  # every test draws off-screen

import pytest  # noqa: E402

from pitwall.auth import TOKEN_ENV_VAR  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--run-network", action="store_true",
                     help="also run tests that call the real Jolpica / F1 servers")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="calls real APIs; use --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def no_real_token(request, monkeypatch):
    """Offline tests never see your real F1 TV token."""
    if request.node.get_closest_marker("network") is None:
        monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    yield
