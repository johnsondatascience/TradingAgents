import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from tradingagents import default_config
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import VendorNotConfiguredError
from tradingagents.dataflows.gexter import (
    GexterNotConfiguredError,
    GexterUnavailableError,
    fetch_document,
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


OK_DOCUMENT = {
    "schema_version": 1,
    "trading_day": "2026-08-28",
    "model_available": True,
    "symbols": {
        "SPX": {
            "status": "ok",
            "spot": 5412.3,
            "asof": "2026-08-28T14:30:00-04:00",
            "quality": {
                "modeled_gamma_frac": 0.83,
                "pre_1300": False,
                "used_fallback": False,
            },
            "regime_divergence": True,
            "stale": {
                "regime": "compression",
                "strength": "moderate",
                "confidence": 0.72,
                "trade_bias": "sell_premium",
                "risk_adjustment": 1.0,
                "interpretation": "Dealers are long gamma.",
                "recommended_structures": ["iron_condor"],
                "signals_aligned": True,
                "gex_signal": "positive",
                "event_context": None,
                "net_gex_bn": 1.23,
                "call_gex_bn": 2.0,
                "put_gex_bn": 0.77,
                "gex_change_rate": 0.04,
                "put_call_gex_ratio": 0.385,
                "near_spot_concentration": 21.4,
                "zero_dte_proportion": 44.2,
                "levels": {"flip": 5400.0, "call_wall": 5450.0, "put_wall": 5350.0},
                "top_strikes": [{"strike": 5450.0, "gex": 1200000000.0}],
            },
            "nowcast": {
                "regime": "transition",
                "strength": "weak",
                "confidence": 0.51,
                "trade_bias": "neutral",
                "risk_adjustment": 0.75,
                "interpretation": "Gamma is near flat.",
                "recommended_structures": [],
                "signals_aligned": False,
                "gex_signal": "neutral",
                "event_context": None,
                "net_gex_bn": 0.12,
                "call_gex_bn": 1.1,
                "put_gex_bn": 0.98,
                "gex_change_rate": -0.9,
                "put_call_gex_ratio": 0.89,
                "near_spot_concentration": 18.0,
                "zero_dte_proportion": 51.0,
                "levels": {"flip": 5410.0, "call_wall": 5450.0, "put_wall": 5300.0},
                "top_strikes": [{"strike": 5450.0, "gex": 900000000.0}],
            },
        }
    },
}


def _completed(stdout, returncode=0, stderr=""):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _patch_run(monkeypatch, result, recorder=None):
    """Replace subprocess.run so no process is ever spawned."""

    def fake_run(argv, **kwargs):
        if recorder is not None:
            recorder["argv"] = argv
            recorder["kwargs"] = kwargs
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("tradingagents.dataflows.gexter.subprocess.run", fake_run)


def test_fetch_document_parses_stdout(monkeypatch, gexter_config):
    _patch_run(monkeypatch, _completed(json.dumps(OK_DOCUMENT)))
    assert fetch_document("SPX") == OK_DOCUMENT


def test_fetch_document_invokes_the_cli_correctly(monkeypatch, gexter_config):
    recorder = {}
    _patch_run(monkeypatch, _completed(json.dumps(OK_DOCUMENT)), recorder)
    fetch_document("XSP")
    argv = recorder["argv"]
    assert argv[0] == gexter_config["gexter_python"]
    assert argv[1].endswith("nowcast_signals.py")
    assert "--json" in argv
    assert argv[argv.index("--symbols") + 1] == "XSP"
    assert "--top-strikes" not in argv        # omitted so GEXter's default applies
    assert recorder["kwargs"]["cwd"] == gexter_config["gexter_repo"]
    assert recorder["kwargs"]["timeout"] == 45


def test_fetch_document_passes_top_strikes_when_given(monkeypatch, gexter_config):
    recorder = {}
    _patch_run(monkeypatch, _completed(json.dumps(OK_DOCUMENT)), recorder)
    fetch_document("SPX", top_strikes=3)
    argv = recorder["argv"]
    assert argv[argv.index("--top-strikes") + 1] == "3"


def test_error_document_raises_with_its_message(monkeypatch, gexter_config):
    doc = {"schema_version": 1, "error": "OperationalError: connection refused"}
    _patch_run(monkeypatch, _completed(json.dumps(doc), returncode=1))
    with pytest.raises(GexterUnavailableError, match="connection refused"):
        fetch_document("SPX")


def test_nonzero_exit_with_unparseable_stdout_raises(monkeypatch, gexter_config):
    _patch_run(monkeypatch, _completed("Traceback (most recent call last):", 1, "boom"))
    with pytest.raises(GexterUnavailableError):
        fetch_document("SPX")


def test_unparseable_stdout_raises(monkeypatch, gexter_config):
    _patch_run(monkeypatch, _completed("not json at all"))
    with pytest.raises(GexterUnavailableError):
        fetch_document("SPX")


def test_timeout_raises(monkeypatch, gexter_config):
    _patch_run(monkeypatch, subprocess.TimeoutExpired(cmd="gexter", timeout=45))
    with pytest.raises(GexterUnavailableError, match="timed out"):
        fetch_document("SPX")


def test_wrong_schema_version_raises_naming_the_version(monkeypatch, gexter_config):
    doc = dict(OK_DOCUMENT, schema_version=2)
    _patch_run(monkeypatch, _completed(json.dumps(doc)))
    with pytest.raises(GexterUnavailableError, match="2"):
        fetch_document("SPX")


def test_missing_schema_version_raises(monkeypatch, gexter_config):
    doc = {k: v for k, v in OK_DOCUMENT.items() if k != "schema_version"}
    _patch_run(monkeypatch, _completed(json.dumps(doc)))
    with pytest.raises(GexterUnavailableError):
        fetch_document("SPX")


def test_unavailable_is_a_vendor_error_not_a_crash(monkeypatch, gexter_config):
    from tradingagents.dataflows.errors import VendorError

    assert issubclass(GexterUnavailableError, VendorError)
