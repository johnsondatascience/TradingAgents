# 0-2DTE Strike Candidates — Plan B (TradingAgents) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry GEXter's priced 0-2DTE candidates from `AgentState` through the Trader and Portfolio Manager into the final decision, without any LLM ever restating a strike, a premium, or a max loss.

**Architecture:** The market analyst node fetches the GEXter document itself — with `state["trade_date"]`, before the LLM runs — and stores it in `AgentState`. The Trader selects a candidate **by id only**; the renderer resolves that id against the stored document and prints the authoritative numbers. The Portfolio Manager approves, rejects, or resizes. Every new schema field defaults to `None`, so equity runs and the free-text fallback path are untouched.

**Tech Stack:** Python 3.12, LangGraph, LangChain, Pydantic v2, pytest. No new dependencies.

**Spec:** `../gexter/docs/superpowers/specs/2026-08-31-0dte-strike-candidates-design.md` (Components 4-5)

## Prerequisite

**Plan A must be merged in the gexter repo before Task 2.** Tasks here assert against documents Plan A produces; writing them first would mean testing against a contract that does not exist. Task 1 (config keys) has no such dependency.

Work on `feat/gexter-market-structure`.

## Global Constraints

- **The LLM never restates a number.** Strikes, premiums, and max loss are rendered from the stored document. The model emits an id and a verdict.
- **Additive and defaulted.** Every new schema field defaults to `None`; every new config key has a default that preserves current behaviour.
- **`SUPPORTED_SCHEMA_VERSION` stays 1.** Plan A's additions are additive; do not bump it.
- **Render order is load-bearing.** `**Rating**`, `**Executive Summary**`, and `**Investment Thesis**` keep their positions — the memory log, CLI display, and report writers parse them.
- **Units.** Everything from the document is in SPX index points. Never convert to dollars in a render.
- **Degrade, never fabricate.** A missing document, an unknown id, or a failed structured call yields today's equity-shaped output, not an invented structure.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `tradingagents/default_config.py` (modify) | Four new GEXter keys plus their `_ENV_OVERRIDES` rows. |
| `tradingagents/dataflows/gexter.py` (modify) | Passes date/cutoff/candidate flags to the CLI; live freshness gate; renders the new blocks. |
| `tradingagents/agents/utils/agent_states.py` (modify) | `market_structure` state key. |
| `tradingagents/agents/analysts/market_analyst.py` (modify) | Fetches the document in the node and stores it. |
| `tradingagents/agents/schemas.py` (modify) | `CandidateResponse`, Trader and PM fields, renderers. |
| `tradingagents/agents/trader/trader.py` (modify) | Resolves and validates `candidate_id` against state. |
| `tradingagents/agents/managers/portfolio_manager.py` (modify) | Carries the structure verdict into the final decision. |
| `tests/test_gexter_vendor.py` (modify) | Argv, freshness gate, rendering. |
| `tests/test_market_structure_state.py` (create) | Node fetch, state plumbing, id validation, render order. |

---

### Task 1: Config keys

Four keys, all routed through `_ENV_OVERRIDES` so coercion is driven by the default's type — the same discipline `gexter_timeout` already follows, which exists because a malformed value should name its env var rather than raise a bare `int()` error at import.

**Files:**
- Modify: `tradingagents/default_config.py`
- Modify: `tests/test_env_overrides.py`

**Interfaces:**
- Consumes: nothing.
- Produces: config keys `gexter_candidates` (bool, `True`), `gexter_dte_max` (int, `2`), `gexter_cutoff` (str or None, `None`), `gexter_max_quote_age_seconds` (int, `1800`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_env_overrides.py`:

```python
def test_gexter_candidate_keys_have_defaults():
    from tradingagents.default_config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["gexter_candidates"] is True
    assert DEFAULT_CONFIG["gexter_dte_max"] == 2
    assert DEFAULT_CONFIG["gexter_cutoff"] is None
    assert DEFAULT_CONFIG["gexter_max_quote_age_seconds"] == 1800


def test_gexter_candidate_keys_coerce_from_env(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_GEXTER_CANDIDATES", "false")
    monkeypatch.setenv("TRADINGAGENTS_GEXTER_DTE_MAX", "1")
    monkeypatch.setenv("TRADINGAGENTS_GEXTER_MAX_QUOTE_AGE", "600")
    monkeypatch.setenv("TRADINGAGENTS_GEXTER_CUTOFF", "11:00")
    import importlib
    from tradingagents import default_config
    importlib.reload(default_config)
    cfg = default_config.DEFAULT_CONFIG
    assert cfg["gexter_candidates"] is False
    assert cfg["gexter_dte_max"] == 1
    assert cfg["gexter_max_quote_age_seconds"] == 600
    assert cfg["gexter_cutoff"] == "11:00"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_env_overrides.py -q`
Expected: FAIL — `KeyError: 'gexter_candidates'`

- [ ] **Step 3: Write minimal implementation**

Add to `_ENV_OVERRIDES`, beside the existing GEXter row:

```python
    "TRADINGAGENTS_GEXTER_CANDIDATES":    "gexter_candidates",
    "TRADINGAGENTS_GEXTER_DTE_MAX":       "gexter_dte_max",
    "TRADINGAGENTS_GEXTER_CUTOFF":        "gexter_cutoff",
    "TRADINGAGENTS_GEXTER_MAX_QUOTE_AGE": "gexter_max_quote_age_seconds",
```

Add to `DEFAULT_CONFIG`, beside `gexter_timeout`:

```python
    # Ask GEXter for priced 0-2DTE structure candidates. Costs it a second DB
    # query; harmless when GEXter is unconfigured, since nothing is spawned.
    "gexter_candidates": True,
    "gexter_dte_max": 2,
    # ET "HH:MM" bounding the snapshot load, for point-in-time replay runs.
    # None means "everything collected so far", which is what a live run wants.
    "gexter_cutoff": None,
    # A priced 0DTE structure goes stale in minutes. 1800s is two 15-minute
    # collection cycles, so one missed cycle does not blind a live run.
    "gexter_max_quote_age_seconds": 1800,
```

Note: `gexter_cutoff` defaults to `None`, so `_coerce` has no reference type. Confirm `_coerce` leaves a `None` default as the raw string; if it does not, special-case it the way any other string-valued key is handled.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_env_overrides.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/default_config.py tests/test_env_overrides.py
git commit -m "Add GEXter candidate config keys with env overrides"
```

---

### Task 2: Pass date, cutoff, and candidate flags to the CLI

`fetch_document` currently builds `[python, cli, "--json", "--symbols", symbol]` and never passes a date. The CLI has accepted `--date` and `--cutoff` all along — the gap was entirely here.

**Files:**
- Modify: `tradingagents/dataflows/gexter.py`
- Modify: `tests/test_gexter_vendor.py`

**Interfaces:**
- Consumes: Task 1's config keys.
- Produces: `fetch_document(symbol, top_strikes=None, trade_date=None, cutoff=None, candidates=None, dte_max=None) -> dict`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gexter_vendor.py`:

```python
def _capture_argv(monkeypatch, stdout):
    """Run fetch_document against a stubbed subprocess and return its argv."""
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


_MINIMAL_DOC = json.dumps({
    "schema_version": 1, "trading_day": "2026-08-31", "model_available": True,
    "symbols": {"SPX": {"status": "ok", "spot": 6416.2,
                        "asof": "2026-08-31T13:45:00-04:00",
                        "quality": {}, "stale": {"regime": "compression"},
                        "nowcast": None, "regime_divergence": None,
                        "spot_context": None, "candidates": None,
                        "candidates_suppressed_reason": None}},
})


def test_fetch_document_passes_date_and_cutoff(gexter_config, monkeypatch):
    seen = _capture_argv(monkeypatch, _MINIMAL_DOC)
    fetch_document("SPX", trade_date="2026-08-31", cutoff="11:00")
    argv = seen["argv"]
    assert "--date" in argv and argv[argv.index("--date") + 1] == "2026-08-31"
    assert "--cutoff" in argv and argv[argv.index("--cutoff") + 1] == "11:00"


def test_fetch_document_omits_absent_optional_flags(gexter_config, monkeypatch):
    seen = _capture_argv(monkeypatch, _MINIMAL_DOC)
    fetch_document("SPX")
    argv = seen["argv"]
    for flag in ("--date", "--cutoff", "--candidates", "--dte-max"):
        assert flag not in argv


def test_fetch_document_requests_candidates_when_asked(gexter_config, monkeypatch):
    seen = _capture_argv(monkeypatch, _MINIMAL_DOC)
    fetch_document("SPX", candidates=True, dte_max=1)
    argv = seen["argv"]
    assert "--candidates" in argv
    assert argv[argv.index("--dte-max") + 1] == "1"


def test_fetch_document_still_rejects_an_unknown_schema_version(gexter_config, monkeypatch):
    _capture_argv(monkeypatch, json.dumps({"schema_version": 2, "symbols": {}}))
    with pytest.raises(GexterUnavailableError, match="schema_version"):
        fetch_document("SPX")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gexter_vendor.py -q`
Expected: FAIL — `TypeError: fetch_document() got an unexpected keyword argument 'trade_date'`

- [ ] **Step 3: Write minimal implementation**

In `tradingagents/dataflows/gexter.py`:

```python
def fetch_document(symbol, top_strikes=None, trade_date=None, cutoff=None,
                   candidates=None, dte_max=None) -> dict:
    """Run GEXter's CLI and return its parsed, version-checked document.

    ``trade_date`` and ``cutoff`` are what make a replay run point-in-time
    correct; both flags have existed on the CLI all along and were simply never
    passed. Omitting a flag rather than passing an empty value keeps the
    subprocess contract explicit -- GEXter's own defaults apply.

    Raises GexterUnavailableError for every failure mode, so the router's
    optional-category handling degrades to a sentinel rather than aborting.
    """
    repo, python, timeout = gexter_paths()
    argv = [python, os.path.join(repo, GEXTER_CLI), "--json", "--symbols", symbol]
    if top_strikes is not None:
        argv += ["--top-strikes", str(top_strikes)]
    if trade_date:
        argv += ["--date", str(trade_date)]
    if cutoff:
        argv += ["--cutoff", str(cutoff)]
    if candidates:
        argv += ["--candidates"]
        if dte_max is not None:
            argv += ["--dte-max", str(dte_max)]
    # ... rest of the existing body unchanged ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gexter_vendor.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/dataflows/gexter.py tests/test_gexter_vendor.py
git commit -m "Pass date, cutoff, and candidate flags through to GEXter"
```

---

### Task 3: The live freshness gate

A priced 0DTE structure has a shelf life of minutes. On a live run a stale quote is worse than no quote. On a replay run the cutoff *is* the as-of, so comparing it to wall-clock now would suppress every historical candidate — the gate must not fire there.

`spot_context` survives the gate: stale quotes make a structure untradeable, they do not make the levels wrong.

**Files:**
- Modify: `tradingagents/dataflows/gexter.py`
- Modify: `tests/test_gexter_vendor.py`

**Interfaces:**
- Consumes: `gexter_max_quote_age_seconds` (Task 1).
- Produces: `apply_freshness_gate(entry, trade_date, now=None, max_age=None) -> dict` — returns the entry with `candidates` emptied and `candidates_suppressed_reason` set when stale; otherwise unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gexter_vendor.py`:

```python
from datetime import datetime, timedelta, timezone

from tradingagents.dataflows.gexter import apply_freshness_gate

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
    entry = _entry(now - timedelta(seconds=3600))
    gated = apply_freshness_gate(entry, trade_date="2026-08-31", now=now)
    assert gated["candidates"] == []
    assert "3600" in gated["candidates_suppressed_reason"] or "stale" in gated["candidates_suppressed_reason"]
    assert gated["spot_context"] is not None       # levels survive


def test_freshness_gate_keeps_fresh_candidates_on_a_live_run(gexter_config):
    now = datetime(2026, 8, 31, 15, 30, tzinfo=_ET_OFFSET)
    entry = _entry(now - timedelta(seconds=300))
    gated = apply_freshness_gate(entry, trade_date="2026-08-31", now=now)
    assert len(gated["candidates"]) == 1


def test_freshness_gate_does_not_fire_on_a_replay_run(gexter_config):
    # trade_date is in the past: the cutoff defines the as-of, and wall-clock
    # age would suppress every historical candidate.
    now = datetime(2026, 8, 31, 15, 30, tzinfo=_ET_OFFSET)
    entry = _entry(datetime(2026, 6, 10, 13, 45, tzinfo=_ET_OFFSET))
    gated = apply_freshness_gate(entry, trade_date="2026-06-10", now=now)
    assert len(gated["candidates"]) == 1


def test_freshness_gate_is_a_no_op_without_candidates(gexter_config):
    entry = {"status": "ok", "candidates": None, "spot_context": None,
             "candidates_suppressed_reason": None}
    assert apply_freshness_gate(entry, trade_date="2026-08-31") == entry
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gexter_vendor.py -q`
Expected: FAIL — `ImportError: cannot import name 'apply_freshness_gate'`

- [ ] **Step 3: Write minimal implementation**

Append to `tradingagents/dataflows/gexter.py`:

```python
from datetime import datetime, timezone


def _is_live_run(trade_date, now) -> bool:
    """True when the run's date is today, in the reference clock's own zone."""
    if not trade_date:
        return True
    try:
        return str(trade_date)[:10] == now.date().isoformat()
    except (AttributeError, ValueError):
        return True


def apply_freshness_gate(entry, trade_date, now=None, max_age=None) -> dict:
    """Drop candidates whose quotes are too old to trade on, in place of a run.

    Only fires on a live run. On a replay the cutoff already defines the as-of,
    and measuring a historical quote against wall-clock now would suppress every
    candidate ever produced for a past date.

    spot_context is never touched: stale quotes make a structure untradeable,
    not the levels wrong.
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
            continue
        age = (now - stamp).total_seconds()
        oldest = age if oldest is None else max(oldest, age)

    if oldest is not None and oldest > max_age:
        gated = dict(entry)
        gated["candidates"] = []
        gated["candidates_suppressed_reason"] = (
            f"quotes are {int(oldest)}s old, past the {max_age}s live budget; "
            f"levels below remain valid"
        )
        return gated
    return entry
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gexter_vendor.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/dataflows/gexter.py tests/test_gexter_vendor.py
git commit -m "Gate stale candidate quotes on live runs only"
```

---

### Task 4: Render the new blocks

Renders `spot_context` and `candidates` into the markdown the market analyst reads. Every number comes from the document.

**Files:**
- Modify: `tradingagents/dataflows/gexter.py`
- Modify: `tests/test_gexter_vendor.py`

**Interfaces:**
- Consumes: the document shape from Plan A.
- Produces: `_spot_context_lines(context) -> list[str]`; `_candidate_lines(candidates, reason) -> list[str]`; both called from `render_document`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gexter_vendor.py`:

```python
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


def _doc_with(candidates, reason=None, context=_CONTEXT):
    doc = json.loads(_MINIMAL_DOC)
    entry = doc["symbols"]["SPX"]
    entry["spot_context"] = context
    entry["candidates"] = candidates
    entry["candidates_suppressed_reason"] = reason
    return doc


def test_render_includes_candidate_id_strikes_and_premium():
    out = render_document(_doc_with([_CANDIDATE]), "^GSPC", "SPX", True)
    assert "SPX_20260831_iron_condor_6380_6450" in out
    assert "6380" in out and "6450" in out
    assert "2.5" in out                    # net premium, points
    assert "credit" in out
    assert "22.5" in out                   # max loss


def test_render_states_the_quote_time_and_conviction():
    out = render_document(_doc_with([_CANDIDATE]), "^GSPC", "SPX", True)
    assert "13:45" in out
    assert "moderate" in out


def test_render_shows_level_offsets_in_expected_moves():
    out = render_document(_doc_with([_CANDIDATE]), "^GSPC", "SPX", True)
    assert "0.72" in out
    assert "47.1" in out


def test_render_states_the_suppression_reason_when_there_are_no_candidates():
    out = render_document(_doc_with([], reason="quotes are 3600s old"), "^GSPC", "SPX", True)
    assert "3600s old" in out
    assert "47.1" in out                   # spot_context still rendered


def test_render_is_unchanged_when_the_blocks_are_absent():
    doc = json.loads(_MINIMAL_DOC)
    out = render_document(doc, "^GSPC", "SPX", True)
    assert "Candidate" not in out
    assert "compression" in out            # the regime read still renders
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gexter_vendor.py -q`
Expected: FAIL on the candidate-rendering assertions.

- [ ] **Step 3: Write minimal implementation**

Append to `tradingagents/dataflows/gexter.py` and call both from `render_document` after the regime views:

```python
def _leg_text(leg):
    """'short 6380P' / 'long 6475C' — the shape a trader reads."""
    side = "short" if leg.get("qty", 0) < 0 else "long"
    letter = "P" if str(leg.get("option_type", "")).lower() == "put" else "C"
    return f"{side} {_fmt_number(leg.get('strike'))}{letter}"


def _spot_context_lines(context):
    """Levels measured against spot. Rendered even when no candidate survives."""
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
        em = _fmt_number(level.get("offset_em"))
        pts = _fmt_number(level.get("offset_points"))
        mark = "in play" if level.get("in_play") else "too far to anchor"
        lines.append(
            f"- **{level.get('name')}** {_fmt_number(level.get('strike'))} "
            f"({pts} pts, {em} EM — {mark})")
    resolution = context.get("resolution")
    if resolution:
        lines.append(f"Regime resolves at {_fmt_number(resolution.get('flip'))}: "
                     f"above, {resolution.get('above')}; below, {resolution.get('below')}.")
    basis = context.get("es_basis")
    if basis and basis.get("basis") is not None:
        lines.append(f"ES basis {_fmt_number(basis['basis'])} "
                     f"(as of {basis.get('latest_session')}); "
                     "strikes remain SPX contracts.")
    elif basis and basis.get("suppressed_reason"):
        lines.append(f"*{basis['suppressed_reason']}*")
    return lines


def _candidate_lines(candidates, reason):
    """Priced structures, or why there are none.

    Every number here is GEXter's. Nothing is computed in this process, and an
    LLM reading this must quote these values rather than derive its own.
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
        lines.append(f"{legs}")
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
                         f"{anchor.get('kind')} at {_fmt_number(anchor.get('strike'))} "
                         f"({_fmt_number(anchor.get('offset_em'))} EM)")
        quoted = candidate.get("quoted_asof")
        if quoted:
            lines.append(f"quoted {quoted}")
    return lines
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gexter_vendor.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/dataflows/gexter.py tests/test_gexter_vendor.py
git commit -m "Render spot context and priced candidates for the analyst"
```

---

### Task 5: `AgentState.market_structure` and fetch-in-node

The LLM must not choose the date, must not be able to skip the fetch, and the Trader must later read the same object rather than a paraphrase. So the node fetches before the LLM runs.

Gated on `gexter_configured()` **and** `is_sp_complex`, so a non-S&P ticker spawns no subprocess and the equity path is entirely unchanged.

**Files:**
- Modify: `tradingagents/agents/utils/agent_states.py`
- Modify: `tradingagents/agents/analysts/market_analyst.py`
- Create: `tests/test_market_structure_state.py`

**Interfaces:**
- Consumes: `fetch_document`, `apply_freshness_gate`, `render_document` (Tasks 2-4).
- Produces: `AgentState["market_structure"]`; `fetch_market_structure(state) -> dict | None` in `market_analyst.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_market_structure_state.py`:

```python
"""AgentState plumbing for the GEXter market-structure document."""
import copy

import pytest

from tradingagents import default_config
from tradingagents.agents.analysts.market_analyst import fetch_market_structure
from tradingagents.agents.utils.agent_states import AgentState


def test_agent_state_declares_market_structure():
    assert "market_structure" in AgentState.__annotations__


def _state(ticker="^GSPC", trade_date="2026-08-31"):
    return {"company_of_interest": ticker, "trade_date": trade_date, "messages": []}


def test_no_fetch_when_gexter_is_unconfigured(monkeypatch):
    monkeypatch.setattr("tradingagents.agents.analysts.market_analyst.gexter_configured",
                        lambda: False)
    assert fetch_market_structure(_state()) is None


def test_no_fetch_for_a_ticker_outside_the_sp_complex(monkeypatch):
    called = []
    monkeypatch.setattr("tradingagents.agents.analysts.market_analyst.gexter_configured",
                        lambda: True)
    monkeypatch.setattr("tradingagents.agents.analysts.market_analyst.fetch_document",
                        lambda **kw: called.append(kw))
    assert fetch_market_structure(_state(ticker="NVDA")) is None
    assert called == []


def test_fetch_passes_the_states_trade_date(monkeypatch):
    seen = {}
    monkeypatch.setattr("tradingagents.agents.analysts.market_analyst.gexter_configured",
                        lambda: True)
    monkeypatch.setattr(
        "tradingagents.agents.analysts.market_analyst.fetch_document",
        lambda **kw: seen.update(kw) or {"symbols": {"SPX": {"status": "ok"}}})
    fetch_market_structure(_state(trade_date="2026-08-27"))
    assert seen["trade_date"] == "2026-08-27"
    assert seen["symbol"] == "SPX"
    assert seen["candidates"] is True


def test_a_vendor_failure_yields_none_rather_than_aborting(monkeypatch):
    from tradingagents.dataflows.gexter import GexterUnavailableError
    monkeypatch.setattr("tradingagents.agents.analysts.market_analyst.gexter_configured",
                        lambda: True)

    def boom(**kw):
        raise GexterUnavailableError("postgres down")

    monkeypatch.setattr("tradingagents.agents.analysts.market_analyst.fetch_document", boom)
    assert fetch_market_structure(_state()) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_market_structure_state.py -q`
Expected: FAIL — `ImportError: cannot import name 'fetch_market_structure'`

- [ ] **Step 3: Write minimal implementation**

In `agent_states.py`, add to `AgentState`:

```python
    market_structure: Annotated[
        dict | None,
        "GEXter market-structure document (schema v1) fetched at run start; "
        "None when GEXter is unconfigured or the ticker is outside the S&P complex",
    ]
```

In `market_analyst.py`, add above `create_market_analyst`:

```python
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.gexter import (
    apply_freshness_gate, fetch_document, gexter_configured,
    resolve_gexter_symbol,
)


def fetch_market_structure(state):
    """Fetch GEXter's document for this run, or None when it does not apply.

    Called from the node body before the LLM runs, so the model cannot choose
    the date and cannot skip the fetch, and so the Trader later reads the same
    object rather than a paraphrase of it.

    Never raises: market structure is optional context, and a GEXter outage must
    degrade the run rather than abort it.
    """
    if not gexter_configured():
        return None
    symbol, is_sp_complex = resolve_gexter_symbol(state.get("company_of_interest"))
    if not is_sp_complex:
        # Index positioning is legitimate background for any name, but it is not
        # worth a subprocess the equity path never had.
        return None
    config = get_config()
    try:
        document = fetch_document(
            symbol=symbol,
            trade_date=state.get("trade_date"),
            cutoff=config.get("gexter_cutoff"),
            candidates=bool(config.get("gexter_candidates", True)),
            dte_max=config.get("gexter_dte_max", 2),
        )
    except (VendorError, ValueError, OSError):
        return None
    entry = (document.get("symbols") or {}).get(symbol)
    if isinstance(entry, dict):
        document["symbols"][symbol] = apply_freshness_gate(
            entry, state.get("trade_date"))
    return document
```

In `market_analyst_node`, before building the prompt:

```python
        market_structure = fetch_market_structure(state)
```

and add `"market_structure": market_structure` to the node's return dict.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_market_structure_state.py tests/test_market_analyst_gexter_binding.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/agent_states.py tradingagents/agents/analysts/market_analyst.py tests/test_market_structure_state.py
git commit -m "Fetch the market-structure document in the node, into AgentState"
```

---

### Task 6: Trader selects by id, renderer resolves it

The Trader's numeric output surface is empty by construction. It emits an id and a verdict; the renderer resolves the id against the stored document and prints the authoritative legs and premiums.

An unknown id coerces to `decline` — validated in the node, not in a `field_validator`, because the validator has no access to the document.

**Files:**
- Modify: `tradingagents/agents/schemas.py`
- Modify: `tradingagents/agents/trader/trader.py`
- Modify: `tests/test_market_structure_state.py`

**Interfaces:**
- Consumes: `AgentState["market_structure"]` (Task 5).
- Produces: `CandidateResponse` enum; `TraderProposal.candidate_response / candidate_id / candidate_reasoning / contracts`; `find_candidate(document, candidate_id) -> dict | None`; `render_candidate_block(candidate, contracts) -> list[str]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_market_structure_state.py`:

```python
from tradingagents.agents.schemas import (
    CandidateResponse, TraderProposal, find_candidate, render_trader_proposal,
)

_DOC = {"schema_version": 1, "symbols": {"SPX": {"candidates": [{
    "candidate_id": "SPX_20260831_iron_condor_6380_6450",
    "structure": "iron_condor", "expiry": "2026-08-31",
    "legs": [{"option_type": "put", "strike": 6380, "qty": -1},
             {"option_type": "put", "strike": 6355, "qty": 1},
             {"option_type": "call", "strike": 6450, "qty": -1},
             {"option_type": "call", "strike": 6475, "qty": 1}],
    "net_premium": 2.50, "premium_kind": "credit", "max_loss": 22.50,
    "quoted_asof": "2026-08-31T13:45:00-04:00",
}]}}}


def test_find_candidate_resolves_a_known_id():
    got = find_candidate(_DOC, "SPX_20260831_iron_condor_6380_6450")
    assert got["net_premium"] == 2.50


def test_find_candidate_returns_none_for_an_unknown_id():
    assert find_candidate(_DOC, "SPX_20260831_iron_condor_1_2") is None
    assert find_candidate(None, "anything") is None


def test_trader_proposal_defaults_leave_the_equity_shape_untouched():
    proposal = TraderProposal(action="Buy", reasoning="because")
    assert proposal.candidate_response is None
    assert proposal.candidate_id is None
    rendered = render_trader_proposal(proposal)
    assert "Options Structure" not in rendered
    assert rendered.endswith("FINAL TRANSACTION PROPOSAL: **BUY**")


def test_render_uses_the_documents_numbers_not_the_models():
    proposal = TraderProposal(
        action="Buy", reasoning="walls are in play",
        candidate_response=CandidateResponse.ACCEPT,
        candidate_id="SPX_20260831_iron_condor_6380_6450",
        candidate_reasoning="short call sits on the wall", contracts=2)
    rendered = render_trader_proposal(proposal, document=_DOC)
    assert "short 6380P" in rendered and "long 6475C" in rendered
    assert "2.5" in rendered and "22.5" in rendered
    assert "Contracts**: 2" in rendered
    assert rendered.endswith("FINAL TRANSACTION PROPOSAL: **BUY**")


def test_render_declares_a_decline_without_inventing_a_structure():
    proposal = TraderProposal(
        action="Hold", reasoning="conviction too low",
        candidate_response=CandidateResponse.DECLINE,
        candidate_id="SPX_20260831_iron_condor_6380_6450",
        candidate_reasoning="transition regime, 0.25 size")
    rendered = render_trader_proposal(proposal, document=_DOC)
    assert "Declined" in rendered
    assert "short 6380P" not in rendered


def test_an_unknown_id_renders_as_a_decline():
    proposal = TraderProposal(
        action="Buy", reasoning="x",
        candidate_response=CandidateResponse.ACCEPT,
        candidate_id="SPX_hallucinated_9999")
    rendered = render_trader_proposal(proposal, document=_DOC)
    assert "Declined" in rendered or "unavailable" in rendered
    assert "6380" not in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_market_structure_state.py -q`
Expected: FAIL — `ImportError: cannot import name 'CandidateResponse'`

- [ ] **Step 3: Write minimal implementation**

In `schemas.py`:

```python
class CandidateResponse(str, Enum):
    """The Trader's verdict on a GEXter candidate.

    There is deliberately no MODIFY. Any field letting the model adjust a strike
    or a premium would reintroduce the LLM arithmetic that building candidates
    deterministically exists to eliminate; "a different width" is a decline with
    a reason.
    """
    ACCEPT = "accept"
    DECLINE = "decline"


def find_candidate(document, candidate_id):
    """The candidate with this id anywhere in the document, or None."""
    if not document or not candidate_id:
        return None
    for entry in (document.get("symbols") or {}).values():
        for candidate in (entry or {}).get("candidates") or []:
            if candidate.get("candidate_id") == candidate_id:
                return candidate
    return None
```

Add to `TraderProposal`:

```python
    candidate_response: CandidateResponse | None = Field(
        default=None,
        description=(
            "Whether you accept or decline the offered options structure. "
            "Declining is a first-class outcome: a low-conviction, "
            "small-size candidate on a no-trade day should be declined."
        ),
    )
    candidate_id: str | None = Field(
        default=None,
        description=(
            "The candidate_id of the structure you are responding to, copied "
            "exactly. Never invent one, and never restate its strikes or "
            "premiums — they are rendered from the source data."
        ),
    )
    candidate_reasoning: str | None = Field(
        default=None,
        description="Why you accepted or declined it. One to three sentences.",
    )
    contracts: int | None = Field(
        default=None,
        description="How many contracts, if you accepted.",
    )
```

Rewrite `render_trader_proposal` to take an optional document:

```python
def render_trader_proposal(proposal: TraderProposal, document=None) -> str:
    """Render a TraderProposal to markdown.

    The options block is resolved from ``document`` by id — the model's own
    output contributes no numbers. Without a document, an unknown id, or a
    decline, no structure is printed: degrading to the equity shape is always
    correct, and printing a structure the model described would be fabrication.

    The trailing ``FINAL TRANSACTION PROPOSAL:`` line stays last, because the
    analyst stop-signal text and external greps depend on it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend(_trader_candidate_lines(proposal, document))
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


def _trader_candidate_lines(proposal, document):
    if proposal.candidate_response is None:
        return []
    candidate = find_candidate(document, proposal.candidate_id)
    if proposal.candidate_response is CandidateResponse.DECLINE or candidate is None:
        note = proposal.candidate_reasoning or "no reason given"
        if candidate is None and proposal.candidate_response is CandidateResponse.ACCEPT:
            note = f"referenced structure unavailable; {note}"
        return ["", f"**Options Structure**: Declined — {note}"]
    legs = ", ".join(
        f"{'short' if leg.get('qty', 0) < 0 else 'long'} "
        f"{leg.get('strike')}{'P' if leg.get('option_type') == 'put' else 'C'}"
        for leg in candidate.get("legs") or [])
    lines = ["", f"**Options Structure**: {candidate.get('structure')} "
                 f"exp {candidate.get('expiry')} — {legs}",
             "", f"**Net Premium**: {candidate.get('net_premium')} pts "
                 f"{candidate.get('premium_kind')}  ·  "
                 f"**Max Loss**: {candidate.get('max_loss')} pts"]
    if proposal.contracts is not None:
        lines[-1] += f"  ·  **Contracts**: {proposal.contracts}"
    lines.extend(["", f"**Quoted As Of**: {candidate.get('quoted_asof')}"])
    if proposal.candidate_reasoning:
        lines.extend(["", f"**Structure Rationale**: {proposal.candidate_reasoning}"])
    return lines
```

In `trader.py`, pass the document into the renderer:

```python
        market_structure = state.get("market_structure")
        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            lambda proposal: render_trader_proposal(proposal, document=market_structure),
            "Trader",
        )
```

and add the candidates to the Trader's user message when `market_structure` is present, stating that declining is acceptable and that strikes must not be restated.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_market_structure_state.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/schemas.py tradingagents/agents/trader/trader.py tests/test_market_structure_state.py
git commit -m "Let the Trader select a candidate by id, never by restating it"
```

---

### Task 7: Portfolio Manager verdict, appended not interleaved

`render_pm_decision`'s docstring warns that the memory log, CLI display, and report writers all parse `**Rating**`, `**Executive Summary**`, and `**Investment Thesis**`. The options block is appended after them, never interleaved, and omitted entirely when there is no verdict.

**Files:**
- Modify: `tradingagents/agents/schemas.py`
- Modify: `tradingagents/agents/managers/portfolio_manager.py`
- Modify: `tests/test_market_structure_state.py`

**Interfaces:**
- Consumes: `find_candidate` (Task 6).
- Produces: `PortfolioDecision.structure_verdict / contracts / structure_note`; `render_pm_decision(decision, document=None, candidate_id=None)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_market_structure_state.py`:

```python
from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision


def _decision(**kw):
    base = dict(rating="Hold", executive_summary="summary text",
                investment_thesis="thesis text")
    base.update(kw)
    return PortfolioDecision(**base)


def test_pm_render_without_a_verdict_is_byte_identical_to_today():
    rendered = render_pm_decision(_decision())
    assert rendered == ("**Rating**: Hold\n\n**Executive Summary**: summary text"
                        "\n\n**Investment Thesis**: thesis text")


def test_pm_render_appends_the_structure_after_the_parsed_headers():
    rendered = render_pm_decision(
        _decision(structure_verdict="approve", contracts=2),
        document=_DOC, candidate_id="SPX_20260831_iron_condor_6380_6450")
    assert rendered.index("**Rating**") < rendered.index("**Executive Summary**")
    assert rendered.index("**Investment Thesis**") < rendered.index("**Options Structure**")
    assert "Contracts**: 2" in rendered


def test_pm_reject_prints_no_structure():
    rendered = render_pm_decision(
        _decision(structure_verdict="reject", structure_note="too wide"),
        document=_DOC, candidate_id="SPX_20260831_iron_condor_6380_6450")
    assert "Rejected" in rendered
    assert "6380" not in rendered


def test_pm_resize_changes_only_the_contract_count():
    rendered = render_pm_decision(
        _decision(structure_verdict="resize", contracts=1),
        document=_DOC, candidate_id="SPX_20260831_iron_condor_6380_6450")
    assert "Contracts**: 1" in rendered
    assert "2.5" in rendered            # premium is still the document's
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_market_structure_state.py -q`
Expected: FAIL — `ValidationError: unexpected keyword 'structure_verdict'`

- [ ] **Step 3: Write minimal implementation**

Add to `PortfolioDecision`:

```python
    structure_verdict: Literal["approve", "reject", "resize"] | None = Field(
        default=None,
        description=(
            "Your verdict on the Trader's options structure. 'resize' changes "
            "only the contract count — the one number risk management owns and "
            "the only one that cannot corrupt a strike."
        ),
    )
    contracts: int | None = Field(
        default=None,
        description="Final contract count, if approving or resizing.",
    )
    structure_note: str | None = Field(
        default=None,
        description="One sentence on the structure verdict.",
    )
```

Extend the renderer, appending only:

```python
def render_pm_decision(decision: PortfolioDecision, document=None,
                       candidate_id=None) -> str:
    """Render a PortfolioDecision to the markdown the rest of the system parses.

    The options block is appended after the three headers the memory log, CLI
    display, and report writers read — never interleaved — and omitted entirely
    when there is no verdict, so an equity run is byte-identical to before.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    parts.extend(_pm_structure_lines(decision, document, candidate_id))
    return "\n".join(parts)


def _pm_structure_lines(decision, document, candidate_id):
    if decision.structure_verdict is None:
        return []
    if decision.structure_verdict == "reject":
        return ["", f"**Options Structure**: Rejected — "
                    f"{decision.structure_note or 'no reason given'}"]
    candidate = find_candidate(document, candidate_id)
    if candidate is None:
        return ["", "**Options Structure**: referenced structure unavailable; "
                    "no structure approved."]
    legs = ", ".join(
        f"{'short' if leg.get('qty', 0) < 0 else 'long'} "
        f"{leg.get('strike')}{'P' if leg.get('option_type') == 'put' else 'C'}"
        for leg in candidate.get("legs") or [])
    line = (f"**Net Premium**: {candidate.get('net_premium')} pts "
            f"{candidate.get('premium_kind')}  ·  "
            f"**Max Loss**: {candidate.get('max_loss')} pts")
    if decision.contracts is not None:
        line += f"  ·  **Contracts**: {decision.contracts}"
    out = ["", f"**Options Structure**: {candidate.get('structure')} "
               f"exp {candidate.get('expiry')} — {legs}", "", line,
           "", f"**Quoted As Of**: {candidate.get('quoted_asof')}"]
    if decision.structure_note:
        out.extend(["", f"**Structure Note**: {decision.structure_note}"])
    return out
```

In `portfolio_manager.py`, pass `state.get("market_structure")` and the Trader's `candidate_id` into the renderer, and include the rendered trader proposal in the prompt as it already does.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ -q -k "market_structure or gexter or schema"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/schemas.py tradingagents/agents/managers/portfolio_manager.py tests/test_market_structure_state.py
git commit -m "Carry the structure verdict into the final decision, appended only"
```

---

### Task 8: The degradation guarantees

Three ways this can fail, all of which must produce today's equity output rather than an invented structure. The spec calls the free-text fallback out explicitly; the other two follow the same rule.

**Files:**
- Modify: `tests/test_market_structure_state.py`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_market_structure_state.py`:

```python
def test_free_text_fallback_renders_no_options_block(monkeypatch):
    """When structured output fails there is no candidate_id, so no structure.

    Degrading to the equity shape is correct; printing a structure the model
    described in prose would be fabrication.
    """
    from tradingagents.agents.utils.structured import invoke_structured_or_freetext

    class FailingStructured:
        def invoke(self, messages):
            raise RuntimeError("provider returned malformed JSON")

    class FreeText:
        def invoke(self, messages):
            return type("M", (), {"content": "I like the 6380/6450 condor."})()

    out = invoke_structured_or_freetext(
        FailingStructured(), FreeText(), [],
        lambda p: render_trader_proposal(p, document=_DOC), "Trader")
    assert "Options Structure" not in out


def test_no_document_renders_no_options_block():
    proposal = TraderProposal(
        action="Buy", reasoning="x",
        candidate_response=CandidateResponse.ACCEPT,
        candidate_id="SPX_20260831_iron_condor_6380_6450")
    rendered = render_trader_proposal(proposal, document=None)
    assert "6380P" not in rendered
    assert "Declined" in rendered or "unavailable" in rendered


def test_a_suppressed_candidate_list_never_yields_a_structure():
    doc = {"schema_version": 1, "symbols": {"SPX": {
        "candidates": [],
        "candidates_suppressed_reason": "quotes are 3600s old"}}}
    proposal = TraderProposal(
        action="Buy", reasoning="x",
        candidate_response=CandidateResponse.ACCEPT,
        candidate_id="SPX_20260831_iron_condor_6380_6450")
    assert "6380P" not in render_trader_proposal(proposal, document=doc)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_market_structure_state.py -q`
Expected: PASS if Tasks 6-7 were implemented correctly; FAIL identifies a real gap. Do not weaken the test to make it pass — fix the renderer.

- [ ] **Step 3: Write minimal implementation**

None expected. If a test fails, the cause is a renderer path that prints a structure without resolving it from the document; fix that path.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_market_structure_state.py
git commit -m "Pin the degradation guarantees for the options block"
```

---

### Task 9: End-to-end verification

**Files:** none — produces a verification note.

- [ ] **Step 1: Configure and run a live-shaped run**

```bash
TRADINGAGENTS_GEXTER_REPO=/c/Users/johnsnmi/gexter \
TRADINGAGENTS_GEXTER_PYTHON=/c/Users/johnsnmi/gexter/.venv/Scripts/python.exe \
python -m cli.main --ticker SPX --date 2026-08-31
```

- [ ] **Step 2: Confirm the chain of custody**

Check that the final decision's `**Options Structure**` strikes and premium match, digit for digit, the candidate in the GEXter document produced by:

```bash
cd /c/Users/johnsnmi/gexter && .venv/Scripts/python.exe scripts/oi_model/nowcast_signals.py \
  --json --symbols SPX --candidates --date 2026-08-31 | python -m json.tool
```

Any divergence means a render path is printing model output rather than resolving by id. That is the defect this whole design exists to prevent — treat it as a stop-the-line bug.

- [ ] **Step 3: Confirm the equity path is untouched**

```bash
python -m cli.main --ticker NVDA --date 2026-08-31
```

Expected: no subprocess spawned, no options block, output shape identical to before this plan.

- [ ] **Step 4: Record what could not be verified**

Replay verification across many sessions remains gated on fuller database replication, not on code. State that explicitly.

- [ ] **Step 5: Commit the note**

```bash
git add docs/
git commit -m "Record end-to-end verification of the candidate chain of custody"
```

---

## Plan B self-review

**Spec coverage.** Component 4 (vendor: flags, freshness gate, fetch-in-node, rendering) → Tasks 2-5. Component 5 (AgentState, TraderProposal, PortfolioDecision, rendering, free-text fallback) → Tasks 5-8. Config → Task 1. The ES basis half of Component 4 is Plan A Task 10, since the block is built GEXter-side.

**Type consistency check.** `find_candidate(document, candidate_id)` is defined in Task 6 and reused unchanged in Task 7. `render_trader_proposal(proposal, document=None)` and `render_pm_decision(decision, document=None, candidate_id=None)` both keep their single-argument call signatures working, so any caller not yet updated still renders the equity shape. `CandidateResponse` has exactly two members; there is no `MODIFY` anywhere.

**Deliberate omission.** No prompt change for the three risk debators. Verified during design: `aggressive_debator`, `conservative_debator`, and `neutral_debator` all read `state["trader_investment_plan"]`, which holds the rendered proposal, so the structure already reaches them.
