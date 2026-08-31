# GEXter Market-Structure Vendor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the market analyst a `get_market_structure` tool that surfaces GEXter's index options-positioning regime, by running GEXter's CLI as a subprocess and parsing its versioned JSON contract.

**Architecture:** A new vendor module `tradingagents/dataflows/gexter.py` shells out to a separate repo's CLI, validates the returned document, and renders markdown for an LLM reader. It registers as the sole vendor of a new `market_structure` category, which is listed in `OPTIONAL_CATEGORIES` so any GEXter failure degrades to a sentinel string instead of aborting a run. The tool is bound into the market analyst only when GEXter is configured, keeping this fork's addition invisible to upstream users.

**Tech Stack:** Python 3.10+, stdlib `subprocess`/`json`/`os`, LangChain `@tool`, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-30-gexter-market-structure-vendor-design.md`

## Global Constraints

- **No imports from GEXter.** TradingAgents must never import GEXter code. The subprocess boundary is the point; GEXter pulls in `psycopg2`, `lightgbm`, `polars`, and `ml4t-*` betas.
- **No new dependencies.** Standard library only.
- **This repo is a fork** of `TauricResearch/TradingAgents`. Keep the footprint on upstream-shared files minimal, and keep the feature inert when unconfigured.
- **A GEXter failure must never abort a run.** `market_structure` goes in `OPTIONAL_CATEGORIES`; every vendor failure raises a `VendorError` subclass, which the router converts to a `DATA_UNAVAILABLE:` sentinel.
- **`schema_version` must equal `1`.** Anything else is a hard failure, never a best-effort parse.
- **Symbol namespaces differ.** TradingAgents uses Yahoo symbols (`^GSPC`); GEXter uses `SPX`/`XSP`/`ES`. The map lives in `gexter.py`, NOT in `symbol_utils.py`.
- **The header is two-branch.** An S&P-complex ticker gets "applies to this instrument"; anything else gets an explicit disclaimer naming the ticker.
- **`near_spot_concentration` and `zero_dte_proportion` are percentages, not fractions.** Render with a `%` suffix.
- **Never modify `symbol_utils.py`.**
- Tests run: `python -m pytest tests/ -q`. No test may spawn a subprocess, touch the network, or require a GEXter checkout.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `tradingagents/dataflows/gexter.py` (create) | The whole vendor: config predicate, symbol map, subprocess call, document validation, markdown rendering. One module because these five concerns are meaningless apart and always change together. |
| `tradingagents/agents/utils/market_structure_tools.py` (create) | The `@tool` wrapper. Mirrors `prediction_markets_tools.py`. |
| `tradingagents/dataflows/interface.py` (modify) | Register the category, vendor, method, and optional-category membership. |
| `tradingagents/default_config.py` (modify) | Three config keys plus the `data_vendors` entry. |
| `tradingagents/agents/utils/agent_utils.py` (modify) | Re-export the tool. |
| `tradingagents/agents/analysts/market_analyst.py` (modify) | Conditional binding plus a prompt paragraph. |
| `tests/test_gexter_vendor.py` (create) | Vendor unit tests against canned documents. |
| `tests/test_market_structure_routing.py` (create) | The routing seam and optional-category degradation. |
| `tests/test_market_analyst_gexter_binding.py` (create) | Conditional binding. |

---

### Task 1: Config keys, availability predicate, and the symbol map

The foundation. All pure — no subprocess, no I/O beyond `os.path.isdir`/`isfile`. Everything later builds on `gexter_configured()` and `resolve_gexter_symbol`.

**Files:**
- Create: `tradingagents/dataflows/gexter.py`
- Modify: `tradingagents/default_config.py`
- Create: `tests/test_gexter_vendor.py`

**Interfaces:**
- Consumes: `get_config` from `tradingagents.dataflows.config`; `VendorNotConfiguredError` from `tradingagents.dataflows.errors`.
- Produces:
  - `GexterNotConfiguredError(VendorNotConfiguredError)`
  - `GexterUnavailableError(VendorError)`
  - `gexter_paths() -> tuple[str, str, int]` — returns `(repo, python, timeout)`, raising `GexterNotConfiguredError` when unusable
  - `gexter_configured() -> bool`
  - `resolve_gexter_symbol(ticker: str) -> tuple[str, bool]` — returns `(gexter_symbol, is_sp_complex)`
  - `SP_COMPLEX_TICKERS: dict[str, str]`

- [ ] **Step 1: Add the config keys**

In `tradingagents/default_config.py`, inside the `DEFAULT_CONFIG` dict, add these three keys immediately after the `"memory_log_path"` line:

```python
    # GEXter integration (optional, fork-local). Both paths default to None, so
    # the market-structure tool is not bound and no subprocess is ever spawned
    # unless explicitly configured. gexter_python must be the interpreter from
    # GEXter's own venv — the system interpreter lacks its dependencies.
    "gexter_repo": os.getenv("TRADINGAGENTS_GEXTER_REPO"),
    "gexter_python": os.getenv("TRADINGAGENTS_GEXTER_PYTHON"),
    "gexter_timeout": int(os.getenv("TRADINGAGENTS_GEXTER_TIMEOUT", "120")),
```

Then, in the same file, add one line to the `"data_vendors"` dict, after the `"prediction_markets"` line:

```python
        "market_structure": "gexter",       # Options: gexter (needs TRADINGAGENTS_GEXTER_* paths)
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_gexter_vendor.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_gexter_vendor.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradingagents.dataflows.gexter'`

- [ ] **Step 4: Write minimal implementation**

Create `tradingagents/dataflows/gexter.py`:

```python
"""GEXter market-structure vendor.

Surfaces index options-positioning context — the gamma-exposure regime, flip
strike, call/put walls, directional bias and position-size multiplier — from
GEXter, a separate options-analytics repo.

GEXter is reached as a SUBPROCESS, never an import: it depends on psycopg2,
lightgbm, polars and ml4t betas, and hard-importing it would multiply this
project's install weight for one optional feature. The process boundary is the
design. We invoke its CLI with --json and parse the versioned document it
prints (schema_version 1), which is valid JSON on every path including total
failure.

The feature is inert unless TRADINGAGENTS_GEXTER_REPO and
TRADINGAGENTS_GEXTER_PYTHON are configured: the tool is not bound and no
subprocess is spawned.
"""
import logging
import os

from .config import get_config
from .errors import VendorError, VendorNotConfiguredError

logger = logging.getLogger(__name__)

# The contract version this vendor understands. GEXter's spec states that
# additive changes keep the version at 1 while a removal or retype bumps it, so
# an unrecognized version must fail loudly rather than be misparsed.
SUPPORTED_SCHEMA_VERSION = 1

# GEXter's CLI, relative to its repo root.
GEXTER_CLI = os.path.join("scripts", "oi_model", "nowcast_signals.py")

# TradingAgents/Yahoo ticker -> GEXter/Tradier symbol, for instruments in the
# S&P complex. TradingAgents resolves the index to ^GSPC (symbol_utils maps
# SPX -> ^GSPC); GEXter collects under SPX, XSP and ES.
#
# SPY is a deliberate approximation: GEXter collects SPX and XSP chains, not
# SPY's, and SPY has its own gamma profile. It maps to SPX and counts as S&P
# complex, but the rendered output says the measurement is on SPX chains.
#
# This map lives here rather than in symbol_utils.py because that module is
# upstream-shared and this is a fork-local concern.
SP_COMPLEX_TICKERS = {
    "^GSPC": "SPX",
    "SPX": "SPX",
    "SPX500": "SPX",
    "US500": "SPX",
    "SPY": "SPX",
    "XSP": "XSP",
    "ES=F": "ES",
    "ES": "ES",
}

# What a ticker outside the S&P complex gets: index positioning as market
# context, with the disclaiming header.
DEFAULT_GEXTER_SYMBOL = "SPX"


class GexterNotConfiguredError(VendorNotConfiguredError):
    """GEXter's repo/interpreter paths are unset or do not exist.

    A VendorNotConfiguredError (and thus still a ValueError), so the routing
    layer treats it as "vendor unavailable" and moves on.
    """


class GexterUnavailableError(VendorError):
    """GEXter is configured but could not produce a usable document.

    Covers a non-zero exit, a timeout, unparseable stdout, and an unsupported
    schema version. A VendorError, so the router's optional-category handling
    turns it into a sentinel instead of aborting the run.
    """


def gexter_paths() -> tuple[str, str, int]:
    """Return ``(repo, python, timeout)``, or raise if GEXter is unusable here."""
    config = get_config()
    repo = config.get("gexter_repo")
    python = config.get("gexter_python")
    if not repo or not python:
        raise GexterNotConfiguredError(
            "GEXter is not configured. Set TRADINGAGENTS_GEXTER_REPO and "
            "TRADINGAGENTS_GEXTER_PYTHON to GEXter's repo root and the "
            "interpreter from its virtualenv."
        )
    if not os.path.isdir(repo):
        raise GexterNotConfiguredError(f"GEXter repo not found at {repo!r}.")
    if not os.path.isfile(python):
        raise GexterNotConfiguredError(f"GEXter interpreter not found at {python!r}.")
    return repo, python, int(config.get("gexter_timeout", 120))


def gexter_configured() -> bool:
    """True when GEXter is usable here.

    The single source of truth for availability: the vendor raises on it and
    the market analyst branches on it, so the condition is stated once.
    """
    try:
        gexter_paths()
    except GexterNotConfiguredError:
        return False
    return True


def resolve_gexter_symbol(ticker) -> tuple[str, bool]:
    """Map a run's ticker to ``(gexter_symbol, is_sp_complex)``.

    An unrecognized ticker still gets SPX positioning — index regime is
    legitimate market context for any name — but is flagged as not S&P complex
    so the caller can disclaim the directional fields.
    """
    key = str(ticker or "").strip().upper()
    if key in SP_COMPLEX_TICKERS:
        return SP_COMPLEX_TICKERS[key], True
    return DEFAULT_GEXTER_SYMBOL, False
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_gexter_vendor.py -q`
Expected: PASS, 16 tests

- [ ] **Step 6: Confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: PASS — the new config keys are additive and nothing else reads them yet.

- [ ] **Step 7: Commit**

```bash
git add tradingagents/dataflows/gexter.py tradingagents/default_config.py tests/test_gexter_vendor.py
git commit -m "Add GEXter vendor config, availability predicate, and symbol map"
```

---

### Task 2: Run the CLI and return a validated document

Turns the configured paths into a trustworthy document dict. Every failure mode becomes a typed exception here, so the formatter in Task 3 can assume a well-formed document.

**Files:**
- Modify: `tradingagents/dataflows/gexter.py`
- Modify: `tests/test_gexter_vendor.py`

**Interfaces:**
- Consumes: `gexter_paths`, `GexterUnavailableError`, `SUPPORTED_SCHEMA_VERSION`, `GEXTER_CLI` (Task 1).
- Produces: `fetch_document(symbol: str, top_strikes: int | None = None) -> dict` — the parsed, version-checked GEXter document.

- [ ] **Step 1: Write the failing test**

Add these imports to the top import block of `tests/test_gexter_vendor.py`:

```python
import json
import subprocess
from types import SimpleNamespace
```

and extend the existing `from tradingagents.dataflows.gexter import (...)` block with:

```python
    GexterUnavailableError,
    fetch_document,
)
```

Then append to `tests/test_gexter_vendor.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gexter_vendor.py -k fetch_document -q`
Expected: FAIL — `ImportError: cannot import name 'fetch_document'`

- [ ] **Step 3: Write minimal implementation**

Add `import json` and `import subprocess` to the top of `tradingagents/dataflows/gexter.py`, keeping the stdlib imports alphabetical (`json`, `logging`, `os`, `subprocess`).

Then append:

```python
def _excerpt(text, limit=300):
    """A short, single-line sample of process output for an error message."""
    flat = " ".join((text or "").split())
    return flat[:limit] + ("..." if len(flat) > limit else "")


def fetch_document(symbol, top_strikes=None) -> dict:
    """Run GEXter's CLI and return its parsed, version-checked document.

    Raises GexterUnavailableError for every failure mode, so the router's
    optional-category handling degrades to a sentinel rather than aborting.
    """
    repo, python, timeout = gexter_paths()
    argv = [python, os.path.join(repo, GEXTER_CLI), "--json", "--symbols", symbol]
    if top_strikes is not None:
        argv += ["--top-strikes", str(top_strikes)]

    try:
        completed = subprocess.run(
            argv, cwd=repo, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise GexterUnavailableError(
            f"GEXter timed out after {timeout}s. Its Postgres may be unreachable."
        ) from exc
    except OSError as exc:
        raise GexterUnavailableError(f"Could not run GEXter: {exc}") from exc

    try:
        document = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise GexterUnavailableError(
            f"GEXter stdout was not JSON (exit {completed.returncode}): "
            f"{_excerpt(completed.stdout)!r}"
        ) from exc

    if not isinstance(document, dict):
        raise GexterUnavailableError("GEXter returned a JSON value that is not an object.")

    # GEXter emits a parseable error document with exit 1 on total failure.
    if "error" in document:
        raise GexterUnavailableError(f"GEXter reported: {document['error']}")

    version = document.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise GexterUnavailableError(
            f"GEXter returned schema_version {version!r}; this vendor understands "
            f"only {SUPPORTED_SCHEMA_VERSION}. Its contract has changed — update "
            f"this vendor rather than parsing the document as-is."
        )
    return document
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gexter_vendor.py -q`
Expected: PASS, 26 tests

- [ ] **Step 5: Commit**

```bash
git add tradingagents/dataflows/gexter.py tests/test_gexter_vendor.py
git commit -m "Run the GEXter CLI and validate its JSON document"
```

---

### Task 3: Render the document as markdown

Pure formatting: a document dict plus a ticker in, a string out. No config, no subprocess. This is where the two-branch header lives — the part that decides whether the directional bias is presented as applying to the analyzed instrument or explicitly disclaimed.

**Files:**
- Modify: `tradingagents/dataflows/gexter.py`
- Modify: `tests/test_gexter_vendor.py`

**Interfaces:**
- Consumes: `resolve_gexter_symbol` (Task 1).
- Produces: `render_document(document: dict, ticker: str, symbol: str, is_sp_complex: bool) -> str`

- [ ] **Step 1: Write the failing test**

Extend the `from tradingagents.dataflows.gexter import (...)` block in `tests/test_gexter_vendor.py` with `render_document,` and append:

```python
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
```

Also add `import copy` to the test file's imports if Task 1 did not already (it did — `import copy` is at the top).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gexter_vendor.py -k render -q`
Expected: FAIL — `ImportError: cannot import name 'render_document'`

- [ ] **Step 3: Write minimal implementation**

Append to `tradingagents/dataflows/gexter.py`:

```python
def _fmt_number(value, suffix="", places=2):
    """Format a number, or return None when it is absent.

    No thousands separator: a strike renders as '5400', not '5,400'. The reader
    is an LLM that may quote these back, and a comma invites a parse error.
    """
    if value is None:
        return None
    try:
        text = f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return None
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text + suffix


def _top_strikes_line(top_strikes):
    """'gamma concentrated at 5450 (1.2Bn), 5400 (-0.85Bn)', or '' when absent.

    GEXter reports top_strikes[].gex in raw dollars while the view's net_gex_bn
    is in billions; convert so one line does not mix scales.
    """
    parts = []
    for entry in (top_strikes or [])[:5]:
        strike = _fmt_number((entry or {}).get("strike"))
        raw = (entry or {}).get("gex")
        if strike is None or raw is None:
            continue
        try:
            billions = _fmt_number(float(raw) / 1e9)
        except (TypeError, ValueError):
            continue
        parts.append(f"{strike} ({billions}Bn)" if billions is not None else strike)
    return "gamma concentrated at " + ", ".join(parts) if parts else ""


def _levels_line(levels):
    """'flip 5400 - call wall 5450 - put wall 5350', omitting absent levels."""
    parts = []
    for key, label in (("flip", "flip"), ("call_wall", "call wall"), ("put_wall", "put wall")):
        rendered = _fmt_number((levels or {}).get(key))
        if rendered is not None:
            parts.append(f"{label} {rendered}")
    return "  ·  ".join(parts)


def _view_lines(view, label):
    """Render one regime view (stale or nowcast) as markdown lines."""
    if not view:
        return []
    regime = view.get("regime") or "unknown"
    strength = view.get("strength") or "unknown"
    confidence = _fmt_number(
        None if view.get("confidence") is None else float(view["confidence"]) * 100,
        "%", places=0,
    )
    head = f"**{label}:** {regime} ({strength}"
    head += f", confidence {confidence}" if confidence else ""
    head += ")"
    lines = [head]

    stats = []
    net = _fmt_number(view.get("net_gex_bn"))
    if net is not None:
        stats.append(f"net GEX {net}Bn")
    levels = _levels_line(view.get("levels"))
    if levels:
        stats.append(levels)
    if stats:
        lines.append("  ·  ".join(stats))

    bias = view.get("trade_bias")
    risk = _fmt_number(view.get("risk_adjustment"))
    if bias or risk:
        detail = f"bias {bias or 'n/a'}"
        if risk is not None:
            detail += f"  ·  risk multiplier {risk}"
        lines.append(detail)

    structures = view.get("recommended_structures") or []
    if structures:
        lines.append("structures: " + ", ".join(str(s) for s in structures))

    strikes = _top_strikes_line(view.get("top_strikes"))
    if strikes:
        lines.append(strikes)

    concentration = _fmt_number(view.get("near_spot_concentration"), "%", places=1)
    zero_dte = _fmt_number(view.get("zero_dte_proportion"), "%", places=1)
    # These two are percentages, not fractions — GEXter divides then multiplies
    # by 100 — so they carry a % suffix.
    extras = []
    if concentration is not None:
        extras.append(f"near-spot concentration {concentration}")
    if zero_dte is not None:
        extras.append(f"0DTE share {zero_dte}")
    if extras:
        lines.append("  ·  ".join(extras))

    interpretation = (view.get("interpretation") or "").strip()
    if interpretation:
        lines.append(interpretation)
    return lines


def _header(ticker, symbol, is_sp_complex, trading_day):
    title = f"## S&P 500 Index Options Positioning — {symbol}, {trading_day}"
    if is_sp_complex:
        caveat = (
            "This describes options positioning in the index you are analyzing. "
            "The directional bias and risk multiplier below apply to this instrument."
        )
        if str(ticker).strip().upper() == "SPY":
            caveat += (
                " Note that gamma exposure is measured on SPX option chains, not "
                "SPY's own; SPY tracks the same index but has a distinct chain."
            )
    else:
        caveat = (
            "INDEX-level market regime context. The directional bias and risk "
            f"multiplier below describe the S&P complex, NOT a recommendation "
            f"for {ticker}."
        )
    return [title, "", caveat, ""]


def render_document(document, ticker, symbol, is_sp_complex) -> str:
    """Render a GEXter document as markdown for an LLM reader."""
    trading_day = document.get("trading_day") or "unknown date"
    lines = _header(ticker, symbol, is_sp_complex, trading_day)

    entry = (document.get("symbols") or {}).get(symbol)
    if not entry:
        lines.append(
            f"Positioning for {symbol} is unavailable: GEXter returned no entry "
            "for it. Do not estimate or fabricate values."
        )
        return "\n".join(lines)

    if entry.get("status") != "ok":
        reason = entry.get("reason") or "no reason given"
        lines.append(f"Positioning for {symbol} is unavailable ({reason}).")
        return "\n".join(lines)

    spot = _fmt_number(entry.get("spot"))
    asof = entry.get("asof")
    context = []
    if spot:
        context.append(f"spot ~{spot}")
    if asof:
        context.append(f"as of {asof}")
    if context:
        lines.append("  ·  ".join(context))
        lines.append("")

    nowcast = entry.get("nowcast")
    if nowcast:
        lines += _view_lines(nowcast, "Real-time regime")
        lines.append("")
        lines += _view_lines(entry.get("stale"), "Prior-close regime")
    else:
        lines += _view_lines(entry.get("stale"), "Regime")
        lines.append("")
        lines.append(
            "GEXter's real-time nowcast is unavailable (its model artifact is "
            "missing), so this reflects prior-close open interest rather than "
            "live positioning."
        )

    if entry.get("regime_divergence"):
        lines.append("")
        lines.append(
            "*The real-time and prior-close regimes diverge: intraday open "
            "interest has shifted the regime.*"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gexter_vendor.py -q`
Expected: PASS, 42 tests

- [ ] **Step 5: Commit**

```bash
git add tradingagents/dataflows/gexter.py tests/test_gexter_vendor.py
git commit -m "Render GEXter documents as ticker-scoped markdown"
```

---

### Task 4: The public vendor entry point

Composes Tasks 1-3 into the single function the router calls. Thin by design — it resolves the symbol, fetches, and renders.

**Files:**
- Modify: `tradingagents/dataflows/gexter.py`
- Modify: `tests/test_gexter_vendor.py`

**Interfaces:**
- Consumes: `resolve_gexter_symbol`, `fetch_document`, `render_document`.
- Produces: `get_market_structure(ticker, symbols=None, top_strikes=None) -> str` — the callable registered in `VENDOR_METHODS`.

- [ ] **Step 1: Write the failing test**

Extend the `from tradingagents.dataflows.gexter import (...)` block with `get_market_structure,` and append to `tests/test_gexter_vendor.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gexter_vendor.py -k entry_point -q`
Expected: FAIL — `ImportError: cannot import name 'get_market_structure'`

- [ ] **Step 3: Write minimal implementation**

Append to `tradingagents/dataflows/gexter.py`:

```python
def get_market_structure(ticker, symbols=None, top_strikes=None) -> str:
    """Index options-positioning context from GEXter, formatted for an LLM.

    ``ticker`` is the instrument under analysis; it selects the GEXter symbol
    and decides whether the directional bias is presented as applying to that
    instrument or explicitly disclaimed. ``symbols`` overrides the mapping when
    a caller wants a specific GEXter symbol.
    """
    mapped, is_sp_complex = resolve_gexter_symbol(ticker)
    symbol = (symbols or mapped).strip().upper()
    document = fetch_document(symbol, top_strikes=top_strikes)
    return render_document(document, ticker, symbol, is_sp_complex)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gexter_vendor.py -q`
Expected: PASS, 48 tests

- [ ] **Step 5: Commit**

```bash
git add tradingagents/dataflows/gexter.py tests/test_gexter_vendor.py
git commit -m "Add the GEXter market-structure vendor entry point"
```

---

### Task 5: Register the category, vendor, and tool

Wires the vendor into the routing layer and exposes it as a LangChain tool. Registering `market_structure` in `OPTIONAL_CATEGORIES` is what makes a GEXter failure degrade to a sentinel instead of aborting a run.

**Files:**
- Modify: `tradingagents/dataflows/interface.py`
- Create: `tradingagents/agents/utils/market_structure_tools.py`
- Modify: `tradingagents/agents/utils/agent_utils.py`
- Create: `tests/test_market_structure_routing.py`

**Interfaces:**
- Consumes: `get_market_structure` from `tradingagents.dataflows.gexter` (Task 4).
- Produces: the `market_structure` category; `get_market_structure` as a LangChain tool re-exported from `agent_utils`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_market_structure_routing.py`:

```python
import copy

import pytest

from tradingagents import default_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.gexter import GexterUnavailableError


@pytest.fixture(autouse=True)
def _base_config():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


def test_market_structure_is_a_registered_category():
    assert "market_structure" in interface.TOOLS_CATEGORIES
    assert "get_market_structure" in interface.TOOLS_CATEGORIES["market_structure"]["tools"]


def test_gexter_is_a_registered_vendor():
    assert "gexter" in interface.VENDOR_LIST
    assert "gexter" in interface.VENDOR_METHODS["get_market_structure"]


def test_market_structure_is_optional():
    # This is what keeps a GEXter outage from aborting an analysis.
    assert "market_structure" in interface.OPTIONAL_CATEGORIES


def test_method_resolves_to_its_category():
    assert interface.get_category_for_method("get_market_structure") == "market_structure"


def test_vendor_failure_degrades_to_a_sentinel(monkeypatch):
    def boom(*args, **kwargs):
        raise GexterUnavailableError("GEXter reported: db down")

    monkeypatch.setitem(interface.VENDOR_METHODS["get_market_structure"], "gexter", boom)
    result = interface.route_to_vendor("get_market_structure", "^GSPC")
    assert result.startswith("DATA_UNAVAILABLE")
    assert "market_structure" in result


def test_tool_wrapper_routes_through_the_vendor(monkeypatch):
    from tradingagents.agents.utils.market_structure_tools import get_market_structure

    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_market_structure"],
        "gexter",
        lambda ticker, symbols=None, top_strikes=None: f"rendered:{ticker}:{symbols}",
    )
    assert get_market_structure.invoke({"ticker": "^GSPC"}) == "rendered:^GSPC:None"


def test_tool_is_reexported_from_agent_utils():
    from tradingagents.agents.utils import agent_utils

    assert "get_market_structure" in agent_utils.__all__
    assert hasattr(agent_utils, "get_market_structure")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_market_structure_routing.py -q`
Expected: FAIL — `KeyError: 'market_structure'` / `ModuleNotFoundError` for `market_structure_tools`

- [ ] **Step 3: Register in `interface.py`**

Add the import, keeping the `from .` block alphabetical — it goes after the `from .fred import ...` line:

```python
from .gexter import get_market_structure as get_gexter_market_structure
```

Add a new entry at the end of the `TOOLS_CATEGORIES` dict, after `"prediction_markets"`:

```python
    "market_structure": {
        "description": "Index options positioning and gamma-exposure regime",
        "tools": [
            "get_market_structure",
        ]
    }
```

Add `"gexter"` to `VENDOR_LIST`:

```python
VENDOR_LIST = [
    "yfinance",
    "fred",
    "polymarket",
    "alpha_vantage",
    "gexter",
]
```

Extend `OPTIONAL_CATEGORIES` and its comment:

```python
OPTIONAL_CATEGORIES = {"macro_data", "prediction_markets", "market_structure"}
```

Add a new entry at the end of `VENDOR_METHODS`, after `get_prediction_markets`:

```python
    # market_structure
    "get_market_structure": {
        "gexter": get_gexter_market_structure,
    },
```

- [ ] **Step 4: Create the tool wrapper**

Create `tradingagents/agents/utils/market_structure_tools.py`:

```python
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_market_structure(
    ticker: Annotated[
        str,
        "The ticker under analysis, e.g. '^GSPC' or 'NVDA'. Determines whether "
        "the directional bias applies to your instrument or is index context.",
    ],
    symbols: Annotated[
        str | None,
        "Override the GEXter symbol (SPX, XSP, ES); omit to derive it from the ticker.",
    ] = None,
    top_strikes: Annotated[
        int | None, "How many strikes by |GEX| to include; omit for a default of 10"
    ] = None,
) -> str:
    """
    Retrieve index options positioning and the gamma-exposure regime for the
    S&P complex: the regime label and confidence, net GEX, the gamma flip
    strike, call and put walls, a directional bias and a position-size
    multiplier. Describes how dealer hedging is likely to shape index price
    behavior today — mean reversion when dealers are long gamma, trend
    acceleration when short.

    This is INDEX-level data. When the ticker under analysis is the S&P complex
    (^GSPC, SPY, XSP, ES=F) the bias applies to it directly; for any other
    ticker it is market-regime context only. Uses the configured
    market_structure vendor.

    Args:
        ticker (str): The instrument under analysis
        symbols (str): Optional GEXter symbol override
        top_strikes (int): Max strikes to include; omit for a default of 10

    Returns:
        str: A formatted markdown report of index options positioning
    """
    return route_to_vendor("get_market_structure", ticker, symbols, top_strikes)
```

- [ ] **Step 5: Re-export from `agent_utils.py`**

Add the import after the `macro_data_tools` import line (keeping the block alphabetical):

```python
from tradingagents.agents.utils.market_structure_tools import get_market_structure
```

and add `"get_market_structure",` to `__all__`, immediately after `"get_macro_indicators",`.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_market_structure_routing.py -q`
Expected: PASS, 7 tests

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS — no existing test broken. Pay attention to any test asserting the exact contents of `TOOLS_CATEGORIES`, `VENDOR_LIST`, or `OPTIONAL_CATEGORIES`; if one exists, update it to include the new entries and say so in your report.

- [ ] **Step 8: Commit**

```bash
git add tradingagents/dataflows/interface.py tradingagents/agents/utils/market_structure_tools.py tradingagents/agents/utils/agent_utils.py tests/test_market_structure_routing.py
git commit -m "Register the market_structure category and GEXter vendor"
```

---

### Task 6: Bind the tool into the market analyst when configured

The tool is bound only when GEXter is configured, so an upstream user without GEXter sees exactly today's tool list and never burns a tool call on something that cannot succeed.

**Files:**
- Modify: `tradingagents/agents/analysts/market_analyst.py`
- Create: `tests/test_market_analyst_gexter_binding.py`

**Interfaces:**
- Consumes: `gexter_configured` from `tradingagents.dataflows.gexter` (Task 1); `get_market_structure` from `agent_utils` (Task 5).
- Produces: no new interface; changes the analyst's bound tool list.

- [ ] **Step 1: Write the failing test**

Create `tests/test_market_analyst_gexter_binding.py`:

```python
import copy

import pytest

from tradingagents import default_config
from tradingagents.agents.analysts import market_analyst
from tradingagents.dataflows.config import set_config


class _RecordingLLM:
    """Captures the tools bound to it; never calls a model."""

    def __init__(self):
        self.bound = None

    def bind_tools(self, tools):
        self.bound = tools
        return self


def _bound_tool_names(monkeypatch, configured):
    monkeypatch.setattr(market_analyst, "gexter_configured", lambda: configured)
    llm = _RecordingLLM()
    node = market_analyst.create_market_analyst(llm)
    state = {
        "trade_date": "2026-08-30",
        "company_of_interest": "^GSPC",
        "asset_type": "stock",
        "instrument_context": "Analyzing ^GSPC.",
        "messages": [],
    }
    with pytest.raises(Exception):
        # The recording LLM has no invoke(); we only need bind_tools to have run.
        node(state)
    return [tool.name for tool in llm.bound]


@pytest.fixture(autouse=True)
def _base_config():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


def test_tool_is_bound_when_gexter_is_configured(monkeypatch):
    assert "get_market_structure" in _bound_tool_names(monkeypatch, configured=True)


def test_tool_is_absent_when_gexter_is_not_configured(monkeypatch):
    names = _bound_tool_names(monkeypatch, configured=False)
    assert "get_market_structure" not in names
    assert names == ["get_stock_data", "get_indicators", "get_verified_market_snapshot"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_market_analyst_gexter_binding.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'gexter_configured'`

- [ ] **Step 3: Write minimal implementation**

In `tradingagents/agents/analysts/market_analyst.py`, extend the existing import from `agent_utils` with `get_market_structure` (keep the list alphabetical), and add one new import below it:

```python
from tradingagents.dataflows.gexter import gexter_configured
```

Replace the `tools = [...]` assignment with:

```python
        tools = [
            get_stock_data,
            get_indicators,
            get_verified_market_snapshot,
        ]
        # Fork-local: GEXter supplies index options positioning. Bound only when
        # configured, so an upstream user without GEXter sees an unchanged tool
        # list and never spends a tool call on something that cannot succeed.
        gexter_available = gexter_configured()
        if gexter_available:
            tools.append(get_market_structure)
```

Then, immediately before the line `+ """ Make sure to append a Markdown table at the end...`, insert the conditional paragraph:

```python
            + (
                """

You also have get_market_structure, which reports S&P index options positioning: the gamma-exposure regime, the gamma flip strike, and call/put walls. Dealers long gamma (positive net GEX, "compression") tends to dampen moves toward the flip strike; dealers short gamma ("expansion") tends to accelerate them. Call it once when index positioning would inform your read of the tape, and pass the ticker you are analyzing. When that ticker is not in the S&P complex, treat the result as market-regime background, not as a signal about your ticker. If it returns DATA_UNAVAILABLE, proceed without it and do not fabricate positioning."""
                if gexter_available
                else ""
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_market_analyst_gexter_binding.py -q`
Expected: PASS, 2 tests

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS. If a prompt-snapshot test asserts the market analyst's system message, it must still pass in the unconfigured case — the paragraph is empty then. Report it if one fails.

- [ ] **Step 6: Commit**

```bash
git add tradingagents/agents/analysts/market_analyst.py tests/test_market_analyst_gexter_binding.py
git commit -m "Bind get_market_structure into the market analyst when configured"
```

---

### Task 7: Manual verification against the real environment

The success criteria are about live behavior across two repos. Verify what can be verified and record honestly what cannot.

**Files:** none (verification only)

- [ ] **Step 1: Verify the unconfigured path**

With no `TRADINGAGENTS_GEXTER_*` variables set:

Run:
```bash
python -c "from tradingagents.dataflows.gexter import gexter_configured; print(gexter_configured())"
```
Expected: `False`.

Confirm the analyst binds three tools, exactly as before this branch:
```bash
python -c "
from tradingagents.agents.analysts import market_analyst
class L:
    def bind_tools(self, tools):
        print([t.name for t in tools]); raise SystemExit(0)
market_analyst.create_market_analyst(L())({'trade_date':'2026-08-30','company_of_interest':'NVDA','asset_type':'stock','instrument_context':'x','messages':[]})
"
```
Expected: `['get_stock_data', 'get_indicators', 'get_verified_market_snapshot']`.

- [ ] **Step 2: Verify the configured-but-GEXter-down path**

GEXter's Postgres is expected to be down on this machine. Point the config at the real GEXter checkout and confirm the run degrades rather than raising:

```bash
TRADINGAGENTS_GEXTER_REPO="C:/Users/johnsnmi/gexter" \
TRADINGAGENTS_GEXTER_PYTHON="C:/Users/johnsnmi/gexter/.venv/Scripts/python.exe" \
python -c "
from tradingagents.dataflows import interface
print(interface.route_to_vendor('get_market_structure', '^GSPC')[:200])
"
```
Expected: a string starting `DATA_UNAVAILABLE: optional market_structure could not be retrieved`, carrying GEXter's own error text. **Not** a traceback.

This exercises the whole path end to end — config, subprocess, GEXter's error document, exit code 1, the vendor's exception, and the router's optional-category degradation.

- [ ] **Step 3: Verify the happy path (requires GEXter's stack up)**

Start GEXter's Postgres (`docker-compose up -d postgres` in the gexter repo) and, if a collection exists for the latest trading day, re-run the Step 2 command. Expected: rendered markdown containing a regime label and a flip strike, with the "apply to this instrument" header for `^GSPC`.

If Postgres cannot be started, record Steps 1-2 as passing and Step 3 as **unverified** — do not mark this task complete without noting it.

- [ ] **Step 4: Commit any fixes**

If Steps 1-3 surface defects, fix them with a test first, then commit. If everything passes, no commit is needed for this task; say so rather than manufacturing one.

---

## Done criteria

1. `python -m pytest tests/ -q` passes with no existing test broken.
2. With GEXter unconfigured, the market analyst binds exactly `get_stock_data`, `get_indicators`, `get_verified_market_snapshot`, and no subprocess is spawned.
3. With GEXter configured but its Postgres down, `route_to_vendor("get_market_structure", ...)` returns a `DATA_UNAVAILABLE` sentinel and does not raise.
4. A run analyzing `^GSPC` renders the "apply to this instrument" header; a run analyzing `NVDA` renders a header naming NVDA in the disclaimer.
5. A `schema_version` other than `1` raises `GexterUnavailableError` naming the observed version.
6. No file outside these nine is modified: `tradingagents/dataflows/gexter.py`, `tradingagents/dataflows/interface.py`, `tradingagents/default_config.py`, `tradingagents/agents/utils/market_structure_tools.py`, `tradingagents/agents/utils/agent_utils.py`, `tradingagents/agents/analysts/market_analyst.py`, `tests/test_gexter_vendor.py`, `tests/test_market_structure_routing.py`, `tests/test_market_analyst_gexter_binding.py`.
7. `tradingagents/dataflows/symbol_utils.py` is untouched.
