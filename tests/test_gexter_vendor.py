import copy

import pytest

from tradingagents import default_config
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import VendorNotConfiguredError
from tradingagents.dataflows.gexter import (
    GexterNotConfiguredError,
    gexter_configured,
    gexter_paths,
    resolve_gexter_symbol,
)


@pytest.fixture
def gexter_config(tmp_path):
    """Config pointing at a fake but existing GEXter repo and interpreter."""
    repo = tmp_path / "gexter"
    repo.mkdir()
    python = tmp_path / "python.exe"
    python.write_text("")
    cfg = copy.deepcopy(default_config.DEFAULT_CONFIG)
    cfg["gexter_repo"] = str(repo)
    cfg["gexter_python"] = str(python)
    cfg["gexter_timeout"] = 45
    set_config(cfg)
    return cfg


@pytest.fixture
def no_gexter_config():
    cfg = copy.deepcopy(default_config.DEFAULT_CONFIG)
    cfg["gexter_repo"] = None
    cfg["gexter_python"] = None
    set_config(cfg)
    return cfg


def test_gexter_paths_returns_configured_values(gexter_config):
    repo, python, timeout = gexter_paths()
    assert repo == gexter_config["gexter_repo"]
    assert python == gexter_config["gexter_python"]
    assert timeout == 45


def test_gexter_paths_raises_when_unset(no_gexter_config):
    with pytest.raises(GexterNotConfiguredError):
        gexter_paths()


def test_gexter_not_configured_is_a_vendor_not_configured_error():
    # The router treats VendorNotConfiguredError as "vendor unavailable".
    assert issubclass(GexterNotConfiguredError, VendorNotConfiguredError)


def test_gexter_paths_raises_when_paths_do_not_exist(tmp_path):
    cfg = copy.deepcopy(default_config.DEFAULT_CONFIG)
    cfg["gexter_repo"] = str(tmp_path / "missing-repo")
    cfg["gexter_python"] = str(tmp_path / "missing-python.exe")
    set_config(cfg)
    with pytest.raises(GexterNotConfiguredError):
        gexter_paths()


def test_gexter_configured_reflects_config(gexter_config):
    assert gexter_configured() is True


def test_gexter_configured_false_when_unset(no_gexter_config):
    assert gexter_configured() is False


@pytest.mark.parametrize(
    "ticker,expected",
    [
        ("^GSPC", "SPX"),
        ("SPX", "SPX"),
        ("SPX500", "SPX"),
        ("US500", "SPX"),
        ("SPY", "SPX"),
        ("XSP", "XSP"),
        ("ES=F", "ES"),
        ("ES", "ES"),
    ],
)
def test_sp_complex_tickers_map_and_are_flagged(ticker, expected):
    symbol, is_complex = resolve_gexter_symbol(ticker)
    assert symbol == expected
    assert is_complex is True


def test_unknown_ticker_falls_back_to_spx_and_is_not_complex():
    symbol, is_complex = resolve_gexter_symbol("NVDA")
    assert symbol == "SPX"
    assert is_complex is False


def test_symbol_resolution_is_case_and_whitespace_insensitive():
    assert resolve_gexter_symbol("  ^gspc  ") == ("SPX", True)
