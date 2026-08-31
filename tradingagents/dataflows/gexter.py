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
import json
import logging
import os
import subprocess

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
