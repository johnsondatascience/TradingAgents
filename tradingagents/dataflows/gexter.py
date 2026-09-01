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
import os
import subprocess
from datetime import datetime, timezone

from .config import get_config
from .errors import VendorError, VendorNotConfiguredError

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
    except (GexterNotConfiguredError, TypeError, ValueError):
        # Called from the market analyst node body, outside route_to_vendor,
        # so a raise here aborts the whole graph run instead of degrading to
        # a sentinel. A malformed gexter_timeout (int() on 'abc') must read
        # as 'not configured' rather than crash the run.
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


def fetch_document(symbol, top_strikes=None, trade_date=None, cutoff=None,
                   candidates=None, dte_max=None) -> dict:
    """Run GEXter's CLI and return its parsed, version-checked document.

    Raises GexterUnavailableError for every failure mode, so the router's
    optional-category handling degrades to a sentinel rather than aborting.
    """
    repo, python, timeout = gexter_paths()
    argv = [python, os.path.join(repo, GEXTER_CLI), "--json", "--symbols", symbol]
    if top_strikes is not None:
        argv += ["--top-strikes", str(top_strikes)]
    # --date and --cutoff have existed on the CLI all along; the gap was that
    # nothing here passed them. They are what makes a replay run
    # point-in-time correct. Flags are omitted rather than passed empty, so
    # GEXter's own defaults apply and the subprocess contract stays explicit.
    if trade_date:
        argv += ["--date", str(trade_date)]
    if cutoff:
        argv += ["--cutoff", str(cutoff)]
    if candidates:
        argv += ["--candidates"]
        # Only meaningful alongside --candidates: passing it alone is
        # accepted and silently ignored, which reads as a honoured tenor.
        if dte_max is not None:
            argv += ["--dte-max", str(dte_max)]

    try:
        completed = subprocess.run(
            # errors="replace": text=True decodes with the platform encoding,
            # so a traceback carrying a non-decodable byte would otherwise
            # raise UnicodeDecodeError out of subprocess.run itself — an
            # untyped escape from a function that promises to degrade.
            argv, cwd=repo, capture_output=True, text=True, errors="replace",
            timeout=timeout,
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
        # stderr is the diagnostic channel by design: GEXter routes its notices
        # there in --json mode, and the likeliest misconfiguration — gexter_python
        # pointing at an interpreter without GEXter's dependencies — dies on
        # 'import psycopg2' with an empty stdout and the traceback on stderr.
        detail = (
            f"GEXter stdout was not JSON (exit {completed.returncode}): "
            f"{_excerpt(completed.stdout)!r}"
        )
        stderr = _excerpt(completed.stderr)
        if stderr:
            detail += f"; stderr: {stderr!r}"
        raise GexterUnavailableError(detail) from exc

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


def _fmt_fraction_pct(value):
    """A 0..1 fraction as a whole-number percent ('72%'), or None when absent.

    GEXter reports ``confidence`` and ``quality.modeled_gamma_frac`` as
    fractions; both read as percentages to a human or an LLM.
    """
    if value is None:
        return None
    try:
        return _fmt_number(float(value) * 100, "%", places=0)
    except (TypeError, ValueError):
        return None


def _top_strikes_line(top_strikes):
    """'gamma concentrated at 5450 (1.2Bn), 5400 (-0.85Bn)', or '' when absent.

    GEXter reports top_strikes[].gex in raw dollars while the view's net_gex_bn
    is in billions; convert so one line does not mix scales.

    Uncapped: GEXter already bounds the list with --top-strikes (default 10),
    so a cap here would render fewer strikes than the tool documents.
    """
    parts = []
    for entry in top_strikes or []:
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
    confidence = _fmt_fraction_pct(view.get("confidence"))
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
    # The tool takes no date and GEXter reports its latest collected trading
    # day, which need not be the date under analysis: a backtest run would see
    # a future date, a live run on a missed collection day a stale one.
    # Passing the run date through is deferred, so the output must say so.
    freshness = (
        f"The date above ({trading_day}) is GEXter's most recently collected "
        "trading day, not necessarily the date you are analyzing. If the two "
        "differ, treat this positioning as background only and do not apply it "
        "to the date under analysis."
    )
    return [title, "", caveat, "", freshness, ""]


def _leg_text(leg):
    """'short 6380P' / 'long 6475C' -- the shape a trader reads."""
    side = "short" if leg.get("qty", 0) < 0 else "long"
    letter = "P" if str(leg.get("option_type", "")).lower() == "put" else "C"
    return f"{side} {_fmt_number(leg.get('strike'))}{letter}"


def _spot_context_lines(context):
    """Levels measured against spot. Rendered even when no candidate survives.

    This is what gives a reader something on days when nothing is tradeable,
    which is why it sits outside the candidate guard.
    """
    if not context:
        return []
    lines = ["", "### Spot context (0-2DTE)"]
    head = []
    for label, key in (("spot", "spot"), ("ATM", "atm_strike"),
                       ("expected move", "session_em_points")):
        value = _fmt_number(context.get(key))
        if value is not None:
            head.append(f"{label} {value}")
    if head:
        lines.append("  ·  ".join(head))
    for level in context.get("levels") or []:
        mark = "in play" if level.get("in_play") else "too far to anchor"
        lines.append(
            f"- **{level.get('name')}** {_fmt_number(level.get('strike'))} "
            f"({_fmt_number(level.get('offset_points'))} pts, "
            f"{_fmt_number(level.get('offset_em'))} EM — {mark})")
    resolution = context.get("resolution")
    if resolution:
        lines.append(
            f"Regime resolves at {_fmt_number(resolution.get('flip'))}: above, "
            f"{resolution.get('above')}; below, {resolution.get('below')}.")
    basis = context.get("es_basis")
    if basis and basis.get("basis") is not None:
        lines.append(f"ES basis {_fmt_number(basis['basis'])} (as of "
                     f"{basis.get('latest_session')}); strikes remain SPX contracts.")
    elif basis and basis.get("suppressed_reason"):
        lines.append(f"*{basis['suppressed_reason']}*")
    return lines


def _candidate_lines(candidates, reason):
    """Priced structures, or why there are none.

    Every number here is GEXter's; nothing is computed in this process. The
    instruction line exists because a model asked to 'check' a spread will
    otherwise re-derive it and disagree with the source by a few cents.
    """
    if candidates is None:
        return []
    if not candidates:
        note = reason or "no structure priced within the fill guards"
        return ["", "### Trade candidates", f"None available: {note}."]
    lines = ["", "### Trade candidates",
             "These are computed structures. Quote them exactly; do not "
             "recompute strikes, premiums, or max loss."]
    for candidate in candidates:
        legs = ", ".join(_leg_text(leg) for leg in candidate.get("legs") or [])
        lines.append("")
        lines.append(f"**`{candidate.get('candidate_id')}`** — "
                     f"{candidate.get('structure')} {candidate.get('dte')}DTE "
                     f"exp {candidate.get('expiry')}")
        lines.append(legs)
        detail = [f"net {_fmt_number(candidate.get('net_premium'))} pts "
                  f"{candidate.get('premium_kind')}"]
        for label, key in (("max loss", "max_loss"), ("max profit", "max_profit")):
            value = _fmt_number(candidate.get(key))
            if value is not None:
                detail.append(f"{label} {value} pts")
        detail.append(f"size x{_fmt_number(candidate.get('size_multiplier'))}")
        detail.append(f"conviction {candidate.get('conviction')}")
        lines.append("  ·  ".join(detail))
        for anchor in candidate.get("anchors") or []:
            lines.append(f"- {anchor.get('role')} anchored on "
                         f"{anchor.get('kind')} at "
                         f"{_fmt_number(anchor.get('strike'))} "
                         f"({_fmt_number(anchor.get('offset_em'))} EM)")
        quoted = candidate.get("quoted_asof")
        if quoted:
            lines.append(f"quoted {quoted}")
    return lines


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
    # GEXter's own readout prints these caveats; dropping them would present
    # provisional levels as bare, unqualified numbers.
    quality = entry.get("quality") or {}
    modeled = _fmt_fraction_pct(quality.get("modeled_gamma_frac"))
    if modeled:
        context.append(f"gamma weight modeled {modeled}")
    if context:
        lines.append("  ·  ".join(context))
        lines.append("")
    if quality.get("pre_1300"):
        lines.append(
            "*Before 13:00 ET GEXter flags the flip strike and put wall as "
            "low-confidence: intraday open interest is still filling in. Treat "
            "those two levels as provisional.*"
        )
        lines.append("")

    nowcast = entry.get("nowcast")
    if nowcast:
        lines += _view_lines(nowcast, "Real-time regime")
        lines.append("")
        lines += _view_lines(entry.get("stale"), "Prior-close regime")
        lines.append("")
        lines.append(
            "The real-time regime is the operative view: it reflects today's "
            "intraday open interest. The prior-close regime is shown for "
            "contrast only — where the two disagree, prefer the real-time bias "
            "and risk multiplier."
        )
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

    # Appended after the regime read. Both are absent unless GEXter was asked
    # for candidates, so a document from an older build renders as it did.
    lines += _spot_context_lines(entry.get("spot_context"))
    lines += _candidate_lines(entry.get("candidates"),
                              entry.get("candidates_suppressed_reason"))
    return "\n".join(lines)


def get_market_structure(ticker, symbols=None, top_strikes=None) -> str:
    """Index options-positioning context from GEXter, formatted for an LLM.

    ``ticker`` is the instrument under analysis; it selects the GEXter symbol
    and decides whether the directional bias is presented as applying to that
    instrument or explicitly disclaimed. ``symbols`` overrides the mapping with
    a SINGLE GEXter symbol; GEXter's CLI accepts a comma-separated list, but the
    renderer looks up exactly one key, so a list would fetch data and discard it.
    """
    mapped, is_sp_complex = resolve_gexter_symbol(ticker)
    symbol = (symbols or mapped).strip().upper()
    document = fetch_document(symbol, top_strikes=top_strikes)
    return render_document(document, ticker, symbol, is_sp_complex)


def _is_live_run(trade_date, now) -> bool:
    """True when the run's date is today in the reference clock's own zone."""
    if not trade_date:
        return True
    try:
        return str(trade_date)[:10] == now.date().isoformat()
    except (AttributeError, ValueError):
        return True


def apply_freshness_gate(entry, trade_date, now=None, max_age=None) -> dict:
    """Drop candidates whose quotes are too old to trade on.

    Only fires on a live run. On a replay the cutoff already defines the
    as-of, and measuring a historical quote against wall-clock now would
    suppress every candidate ever produced for a past date.

    spot_context is never touched: stale quotes make a structure
    untradeable, not the levels wrong.
    """
    candidates = entry.get("candidates")
    if not candidates:
        return entry
    now = now or datetime.now(timezone.utc)
    if not _is_live_run(trade_date, now):
        return entry
    if max_age is None:
        max_age = int(get_config().get("gexter_max_quote_age_seconds", 1800))

    oldest = None
    for candidate in candidates:
        quoted = candidate.get("quoted_asof")
        if not quoted:
            continue
        try:
            stamp = datetime.fromisoformat(quoted)
        except (TypeError, ValueError):
            # An unreadable stamp is not evidence of staleness. Suppressing
            # on it would silently drop tradeable structures over a format
            # change rather than an age problem.
            continue
        age = (now - stamp).total_seconds()
        oldest = age if oldest is None else max(oldest, age)

    if oldest is not None and oldest > max_age:
        gated = dict(entry)
        gated["candidates"] = []
        gated["candidates_suppressed_reason"] = (
            f"quotes are {int(oldest)}s old, past the {max_age}s live "
            f"budget; the levels below remain valid")
        return gated
    return entry
