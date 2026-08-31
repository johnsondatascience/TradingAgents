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
    get_market_structure,
    gexter_configured,
    gexter_paths,
    render_document,
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


def _render(document=None, ticker="^GSPC", symbol="SPX", is_complex=True):
    return render_document(document or OK_DOCUMENT, ticker, symbol, is_complex)


def test_render_includes_regime_and_levels():
    out = _render()
    assert "compression" in out
    assert "5400" in out            # flip
    assert "5450" in out            # call wall
    assert "sell_premium" in out


def test_sp_complex_header_says_it_applies_to_this_instrument():
    out = _render(ticker="^GSPC", is_complex=True)
    assert "apply to this instrument" in out
    assert "NOT a recommendation" not in out


def test_non_complex_header_disclaims_and_names_the_ticker():
    out = _render(ticker="NVDA", is_complex=False)
    assert "NOT a recommendation for NVDA" in out
    assert "apply to this instrument" not in out


def test_spy_states_measurement_is_on_spx_chains():
    out = _render(ticker="SPY", symbol="SPX", is_complex=True)
    assert "SPX option chains" in out
    assert "apply to this instrument" in out   # SPY is still S&P complex


def test_non_spy_complex_ticker_has_no_chain_caveat():
    assert "SPX option chains" not in _render(ticker="^GSPC", is_complex=True)


def test_divergence_line_renders_when_regimes_differ():
    out = _render()
    assert "diverge" in out.lower()


def test_no_divergence_line_when_regimes_match():
    doc = copy.deepcopy(OK_DOCUMENT)
    doc["symbols"]["SPX"]["regime_divergence"] = False
    assert "diverge" not in _render(doc).lower()


def test_missing_model_renders_stale_and_says_nowcast_unavailable():
    doc = copy.deepcopy(OK_DOCUMENT)
    doc["model_available"] = False
    doc["symbols"]["SPX"]["nowcast"] = None
    doc["symbols"]["SPX"]["regime_divergence"] = None
    out = _render(doc)
    assert "compression" in out          # the stale view survives
    assert "real-time" in out.lower()
    assert "transition" not in out       # no fabricated nowcast


def test_no_data_symbol_renders_reason_not_an_error():
    doc = copy.deepcopy(OK_DOCUMENT)
    doc["symbols"]["SPX"] = {"status": "no_data", "reason": "no valid spot (data gap)"}
    out = _render(doc)
    assert "no valid spot (data gap)" in out


def test_missing_symbol_entry_renders_unavailable():
    doc = copy.deepcopy(OK_DOCUMENT)
    doc["symbols"] = {}
    assert "unavailable" in _render(doc).lower()


def test_null_levels_are_omitted_not_printed_as_none():
    doc = copy.deepcopy(OK_DOCUMENT)
    doc["symbols"]["SPX"]["stale"]["levels"] = {
        "flip": None, "call_wall": None, "put_wall": None,
    }
    doc["symbols"]["SPX"]["nowcast"]["levels"] = {
        "flip": None, "call_wall": None, "put_wall": None,
    }
    out = _render(doc)
    assert "None" not in out
    assert "null" not in out


def test_percentage_fields_render_with_a_percent_sign():
    out = _render()
    assert "21.4%" in out
    assert "44.2%" in out


def test_strike_numbers_carry_no_thousands_separator():
    # An LLM may quote these back; "5,400" invites a parse error.
    out = _render()
    assert "5400" in out
    assert "5,400" not in out


def test_top_strikes_render_converted_to_billions():
    # GEXter reports top_strikes[].gex in raw dollars (1.2e9), while
    # net_gex_bn is already in billions; one line must not mix scales.
    out = _render()
    assert "1.2Bn" in out
    assert "1200000000" not in out


def test_absent_top_strikes_render_nothing():
    doc = copy.deepcopy(OK_DOCUMENT)
    doc["symbols"]["SPX"]["stale"]["top_strikes"] = []
    doc["symbols"]["SPX"]["nowcast"]["top_strikes"] = []
    assert "gamma concentrated" not in _render(doc)


def test_trading_day_and_symbol_appear_in_the_heading():
    out = _render()
    assert "2026-08-28" in out
    assert "SPX" in out


def test_entry_point_renders_for_a_complex_ticker(monkeypatch, gexter_config):
    _patch_run(monkeypatch, _completed(json.dumps(OK_DOCUMENT)))
    out = get_market_structure("^GSPC")
    assert "apply to this instrument" in out
    assert "compression" in out


def test_entry_point_disclaims_for_an_unrelated_ticker(monkeypatch, gexter_config):
    _patch_run(monkeypatch, _completed(json.dumps(OK_DOCUMENT)))
    out = get_market_structure("NVDA")
    assert "NOT a recommendation for NVDA" in out


def test_entry_point_queries_the_mapped_symbol(monkeypatch, gexter_config):
    recorder = {}
    _patch_run(monkeypatch, _completed(json.dumps(OK_DOCUMENT)), recorder)
    get_market_structure("^GSPC")
    argv = recorder["argv"]
    assert argv[argv.index("--symbols") + 1] == "SPX"


def test_explicit_symbols_argument_overrides_the_map(monkeypatch, gexter_config):
    recorder = {}
    doc = copy.deepcopy(OK_DOCUMENT)
    doc["symbols"]["XSP"] = doc["symbols"].pop("SPX")
    _patch_run(monkeypatch, _completed(json.dumps(doc)), recorder)
    out = get_market_structure("^GSPC", symbols="XSP")
    argv = recorder["argv"]
    assert argv[argv.index("--symbols") + 1] == "XSP"
    assert "compression" in out


def test_entry_point_propagates_unavailability(monkeypatch, gexter_config):
    _patch_run(monkeypatch, _completed(json.dumps({"schema_version": 1, "error": "db down"}), 1))
    with pytest.raises(GexterUnavailableError):
        get_market_structure("^GSPC")


def test_entry_point_raises_when_unconfigured(no_gexter_config):
    with pytest.raises(GexterNotConfiguredError):
        get_market_structure("^GSPC")
