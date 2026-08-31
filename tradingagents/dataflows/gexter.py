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
