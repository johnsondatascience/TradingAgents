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


def test_gexter_configured_false_for_a_malformed_timeout(gexter_config):
    """A bad timeout must read as "not configured", never abort the run.

    gexter_configured() is called in the market analyst node body, outside
    route_to_vendor, so a raise there is not converted into a sentinel: it
    aborts the whole graph run.
    """
    gexter_config["gexter_timeout"] = "abc"
    set_config(gexter_config)
    assert gexter_configured() is False


def test_gexter_paths_accepts_a_numeric_string_timeout(gexter_config):
    gexter_config["gexter_timeout"] = "45"
    set_config(gexter_config)
    assert gexter_paths()[2] == 45


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
    # text=True decodes with the platform encoding; errors="replace" keeps a
    # non-decodable byte in a traceback from escaping as UnicodeDecodeError.
    assert recorder["kwargs"]["errors"] == "replace"


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


def test_unparseable_stdout_error_carries_the_stderr_excerpt(monkeypatch, gexter_config):
    """The likeliest misconfiguration is diagnosable only from stderr.

    Pointing gexter_python at the wrong interpreter kills GEXter on
    `import psycopg2` before its own error handling: stdout is empty and the
    traceback goes to stderr. Without it the user sees only
    "GEXter stdout was not JSON (exit 1): ''".
    """
    _patch_run(
        monkeypatch,
        _completed("", 1, "ModuleNotFoundError: No module named 'psycopg2'"),
    )
    with pytest.raises(GexterUnavailableError, match="psycopg2"):
        fetch_document("SPX")


def test_empty_stderr_adds_no_noise_to_the_error(monkeypatch, gexter_config):
    _patch_run(monkeypatch, _completed("not json", 1, ""))
    with pytest.raises(GexterUnavailableError) as excinfo:
        fetch_document("SPX")
    assert "stderr" not in str(excinfo.value)


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
    # By value: confidence is a 0..1 fraction rendered as a percent, so a
    # dropped "* 100" would render "confidence 1%" and pass a mere-substring
    # check. 0.72 -> 72%, 0.51 -> 51%.
    assert "confidence 72%" in out
    assert "confidence 51%" in out


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
    # The header branch must remain unaffected by the symbol override: ticker ^GSPC
    # is S&P complex, so the rendered output must say "apply to this instrument"
    # and never say "NOT a recommendation".
    assert "apply to this instrument" in out
    assert "NOT a recommendation" not in out


def test_entry_point_propagates_unavailability(monkeypatch, gexter_config):
    _patch_run(monkeypatch, _completed(json.dumps({"schema_version": 1, "error": "db down"}), 1))
    with pytest.raises(GexterUnavailableError):
        get_market_structure("^GSPC")


def test_entry_point_raises_when_unconfigured(no_gexter_config):
    with pytest.raises(GexterNotConfiguredError):
        get_market_structure("^GSPC")


PRECEDENCE = "The real-time regime is the operative view"


def test_precedence_sentence_names_the_operative_view_when_a_nowcast_exists():
    """Two regime views can disagree; the header speaks of one bias.

    Without a stated precedence a model sizing a position could justify either
    risk multiplier (0.75 nowcast vs 1 prior-close in this document).
    """
    out = _render()
    assert PRECEDENCE in out
    assert "prefer the real-time bias" in out


def test_no_precedence_sentence_when_there_is_no_nowcast_to_prefer():
    doc = copy.deepcopy(OK_DOCUMENT)
    doc["model_available"] = False
    doc["symbols"]["SPX"]["nowcast"] = None
    out = _render(doc)
    assert PRECEDENCE not in out
    assert "Prior-close regime" not in out   # the sole view is just "Regime"


FRESHNESS = "most recently collected"


@pytest.mark.parametrize("ticker,is_complex", [("^GSPC", True), ("NVDA", False)])
def test_header_flags_that_the_date_may_not_be_the_analysis_date(ticker, is_complex):
    """Silent look-ahead / staleness guard.

    The tool takes no date and GEXter reports its latest collected trading day,
    so a backtest run on 2026-06-15 would otherwise read "SPX, 2026-08-28" as if
    it were the analysis date.
    """
    out = _render(ticker=ticker, is_complex=is_complex)
    assert FRESHNESS in out
    assert "2026-08-28" in out
    assert "background only" in out


def test_pre_1300_renders_the_low_confidence_note():
    """GEXter's own readout prints this caveat; stripping it presents the flip
    strike and put wall as bare numbers on a morning run."""
    doc = copy.deepcopy(OK_DOCUMENT)
    doc["symbols"]["SPX"]["quality"]["pre_1300"] = True
    out = _render(doc)
    assert "Before 13:00 ET" in out
    assert "low-confidence" in out


def test_pre_1300_note_is_absent_after_1300():
    # OK_DOCUMENT carries pre_1300: False.
    assert "13:00" not in _render()


def test_modeled_gamma_fraction_renders_as_a_confidence_stat():
    out = _render()
    assert "gamma weight modeled 83%" in out    # 0.83


def test_missing_quality_block_renders_without_error():
    doc = copy.deepcopy(OK_DOCUMENT)
    doc["symbols"]["SPX"].pop("quality")
    out = _render(doc)
    assert "gamma weight modeled" not in out
    assert "13:00" not in out
    assert "compression" in out


def test_top_strikes_are_not_capped_below_the_documented_default():
    """The tool documents a default of 10; a [:5] cap silently rendered five.

    GEXter already bounds this list with --top-strikes, so the renderer must not
    bound it again.
    """
    doc = copy.deepcopy(OK_DOCUMENT)
    strikes = [
        {"strike": 5400.0 + 10 * i, "gex": (i + 1) * 1e8} for i in range(8)
    ]
    doc["symbols"]["SPX"]["nowcast"]["top_strikes"] = strikes
    out = _render(doc)

    line = next(
        text
        for text in out.splitlines()
        if text.startswith("gamma concentrated at") and "5470" in text
    )
    # By count, not by membership: 5400/5450 also appear as levels, so a
    # surviving [:5] cap could still satisfy a "strike in out" check.
    assert line.count("Bn)") == len(strikes)


# --- Date, cutoff and candidate flags ----------------------------------------

_MINIMAL_DOC = json.dumps({
    "schema_version": 1, "trading_day": "2026-08-31", "model_available": True,
    "symbols": {"SPX": {"status": "ok", "spot": 6416.2,
                        "asof": "2026-08-31T13:45:00-04:00",
                        "quality": {}, "stale": {"regime": "compression"},
                        "nowcast": None, "regime_divergence": None,
                        "spot_context": None, "candidates": None,
                        "candidates_suppressed_reason": None}},
})


def _capture_argv(monkeypatch, stdout=_MINIMAL_DOC):
    """Run fetch_document against a stubbed subprocess and expose its argv."""
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


def test_fetch_document_passes_date_and_cutoff(gexter_config, monkeypatch):
    seen = _capture_argv(monkeypatch)
    fetch_document("SPX", trade_date="2026-08-31", cutoff="11:00")
    argv = seen["argv"]
    assert argv[argv.index("--date") + 1] == "2026-08-31"
    assert argv[argv.index("--cutoff") + 1] == "11:00"


def test_fetch_document_omits_absent_optional_flags(gexter_config, monkeypatch):
    seen = _capture_argv(monkeypatch)
    fetch_document("SPX")
    for flag in ("--date", "--cutoff", "--candidates", "--dte-max"):
        assert flag not in seen["argv"]


def test_fetch_document_requests_candidates_when_asked(gexter_config, monkeypatch):
    seen = _capture_argv(monkeypatch)
    fetch_document("SPX", candidates=True, dte_max=1)
    argv = seen["argv"]
    assert "--candidates" in argv
    assert argv[argv.index("--dte-max") + 1] == "1"


def test_fetch_document_omits_dte_max_without_candidates(gexter_config, monkeypatch):
    # --dte-max alone would be accepted and silently ignored by the CLI, which
    # reads as "the tenor was honoured" when nothing was built.
    seen = _capture_argv(monkeypatch)
    fetch_document("SPX", candidates=False, dte_max=1)
    assert "--dte-max" not in seen["argv"]


def test_fetch_document_still_rejects_an_unknown_schema_version(gexter_config, monkeypatch):
    _capture_argv(monkeypatch, json.dumps({"schema_version": 2, "symbols": {}}))
    with pytest.raises(GexterUnavailableError, match="schema_version"):
        fetch_document("SPX")


# --- Live freshness gate -----------------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402

from tradingagents.dataflows.gexter import apply_freshness_gate  # noqa: E402

_ET_OFFSET = timezone(timedelta(hours=-4))


def _entry(quoted_at):
    return {
        "status": "ok",
        "spot_context": {"session_em_points": 47.1},
        "candidates": [{"candidate_id": "SPX_20260831_iron_condor_6380_6450",
                        "quoted_asof": quoted_at.isoformat()}],
        "candidates_suppressed_reason": None,
    }


def test_freshness_gate_drops_stale_candidates_on_a_live_run(gexter_config):
    now = datetime(2026, 8, 31, 15, 30, tzinfo=_ET_OFFSET)
    gated = apply_freshness_gate(_entry(now - timedelta(seconds=3600)),
                                 trade_date="2026-08-31", now=now)
    assert gated["candidates"] == []
    assert "3600" in gated["candidates_suppressed_reason"]
    assert gated["spot_context"] is not None       # levels survive the gate


def test_freshness_gate_keeps_fresh_candidates_on_a_live_run(gexter_config):
    now = datetime(2026, 8, 31, 15, 30, tzinfo=_ET_OFFSET)
    gated = apply_freshness_gate(_entry(now - timedelta(seconds=300)),
                                 trade_date="2026-08-31", now=now)
    assert len(gated["candidates"]) == 1


def test_freshness_gate_does_not_fire_on_a_replay_run(gexter_config):
    # trade_date is in the past: the cutoff defines the as-of, and wall-clock
    # age would suppress every candidate ever produced for a historical date.
    now = datetime(2026, 8, 31, 15, 30, tzinfo=_ET_OFFSET)
    gated = apply_freshness_gate(
        _entry(datetime(2026, 6, 10, 13, 45, tzinfo=_ET_OFFSET)),
        trade_date="2026-06-10", now=now)
    assert len(gated["candidates"]) == 1


def test_freshness_gate_is_a_no_op_without_candidates(gexter_config):
    entry = {"status": "ok", "candidates": None, "spot_context": None,
             "candidates_suppressed_reason": None}
    assert apply_freshness_gate(entry, trade_date="2026-08-31") == entry


def test_freshness_gate_tolerates_an_unparseable_quote_time(gexter_config):
    now = datetime(2026, 8, 31, 15, 30, tzinfo=_ET_OFFSET)
    entry = _entry(now)
    entry["candidates"][0]["quoted_asof"] = "not a timestamp"
    gated = apply_freshness_gate(entry, trade_date="2026-08-31", now=now)
    assert len(gated["candidates"]) == 1     # unmeasurable age is not staleness


# --- Rendering the candidate blocks ------------------------------------------

_CANDIDATE = {
    "candidate_id": "SPX_20260831_iron_condor_6380_6450",
    "structure": "iron_condor", "expiry": "2026-08-31", "dte": 0,
    "legs": [
        {"option_type": "put", "strike": 6380, "qty": -1, "mid": 2.25, "delta": -0.18},
        {"option_type": "put", "strike": 6355, "qty": 1, "mid": 1.20, "delta": -0.10},
        {"option_type": "call", "strike": 6450, "qty": -1, "mid": 3.40, "delta": 0.21},
        {"option_type": "call", "strike": 6475, "qty": 1, "mid": 1.95, "delta": 0.12},
    ],
    "net_premium": 2.50, "premium_kind": "credit",
    "max_loss": 22.50, "max_profit": 2.50,
    "size_multiplier": 0.85, "conviction": "moderate",
    "quoted_asof": "2026-08-31T13:45:00-04:00",
    "anchors": [{"role": "short_call", "kind": "call_wall", "strike": 6450,
                 "offset_em": 0.72, "level_in_play": True}],
}

_CONTEXT = {
    "spot": 6416.2, "atm_strike": 6415, "session_em_points": 47.1,
    "gamma_source_mix": 0.31, "join_dropped": 4,
    "levels": [{"name": "call_wall", "strike": 6450, "offset_points": 33.8,
                "offset_em": 0.72, "in_play": True}],
    "resolution": None, "es_basis": None,
}


def _doc_with(candidates, reason=None, context=None):
    doc = json.loads(_MINIMAL_DOC)
    entry = doc["symbols"]["SPX"]
    entry["spot_context"] = _CONTEXT if context is None else context
    entry["candidates"] = candidates
    entry["candidates_suppressed_reason"] = reason
    return doc


def test_render_includes_candidate_id_strikes_and_premium():
    out = render_document(_doc_with([_CANDIDATE]), "^GSPC", "SPX", True)
    assert "SPX_20260831_iron_condor_6380_6450" in out
    assert "6380" in out and "6475" in out
    assert "2.5" in out and "credit" in out
    assert "22.5" in out


def test_render_states_the_quote_time_and_conviction():
    out = render_document(_doc_with([_CANDIDATE]), "^GSPC", "SPX", True)
    assert "13:45" in out
    assert "moderate" in out


def test_render_shows_level_offsets_in_expected_moves():
    out = render_document(_doc_with([_CANDIDATE]), "^GSPC", "SPX", True)
    assert "0.72" in out
    assert "47.1" in out


def test_render_states_the_suppression_reason_when_there_are_no_candidates():
    out = render_document(_doc_with([], reason="quotes are 3600s old"),
                          "^GSPC", "SPX", True)
    assert "3600s old" in out
    assert "47.1" in out                   # spot_context still rendered


def test_render_tells_the_model_not_to_recompute():
    out = render_document(_doc_with([_CANDIDATE]), "^GSPC", "SPX", True)
    assert "do not recompute" in out.lower()


def test_render_is_unchanged_when_the_blocks_are_absent():
    out = render_document(json.loads(_MINIMAL_DOC), "^GSPC", "SPX", True)
    assert "Trade candidates" not in out
    assert "Spot context" not in out
    assert "compression" in out            # the regime read still renders


def test_render_shows_a_stale_basis_note_instead_of_a_number():
    context = dict(_CONTEXT, es_basis={
        "basis": None, "median_basis": None, "latest_session": "2026-08-20",
        "levels_es": [],
        "suppressed_reason": "ES basis is 11 days stale (latest session 2026-08-20)"})
    out = render_document(_doc_with([_CANDIDATE], context=context),
                          "^GSPC", "SPX", True)
    assert "11 days stale" in out


def test_live_run_detection_uses_market_time_not_utc(gexter_config):
    """After 20:00 ET the UTC calendar has already rolled over.

    trade_date is a *market* date. Comparing it against a UTC date makes every
    evening run look like a replay, silently disabling the freshness gate at
    exactly the hours when quotes are most stale.
    """
    from tradingagents.dataflows.gexter import _is_live_run
    # Production passes a UTC-aware clock, which is where this bites: at 21:00
    # ET the UTC date is already the next day.
    utc_now = datetime(2026, 9, 1, 1, 0, tzinfo=timezone.utc)
    assert utc_now.date().isoformat() == "2026-09-01"
    assert _is_live_run("2026-08-31", utc_now) is True


def test_evening_run_still_drops_stale_quotes(gexter_config):
    utc_now = datetime(2026, 9, 1, 1, 0, tzinfo=timezone.utc)
    gated = apply_freshness_gate(_entry(utc_now - timedelta(seconds=7200)),
                                 trade_date="2026-08-31", now=utc_now)
    assert gated["candidates"] == []


def test_a_genuine_replay_is_still_not_gated(gexter_config):
    utc_now = datetime(2026, 9, 1, 1, 0, tzinfo=timezone.utc)
    gated = apply_freshness_gate(
        _entry(datetime(2026, 6, 10, 13, 45, tzinfo=_ET_OFFSET)),
        trade_date="2026-06-10", now=utc_now)
    assert len(gated["candidates"]) == 1
