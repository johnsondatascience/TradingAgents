"""AgentState plumbing for the GEXter market-structure document.

The node fetches before the LLM runs, so the model cannot choose the date and
cannot skip the fetch, and the Trader later reads the same object rather than a
paraphrase of it.
"""
import pytest

from tradingagents.agents.analysts.market_analyst import fetch_market_structure
from tradingagents.agents.utils.agent_states import AgentState

_MODULE = "tradingagents.agents.analysts.market_analyst"


def test_agent_state_declares_market_structure():
    assert "market_structure" in AgentState.__annotations__


def _state(ticker="^GSPC", trade_date="2026-08-31"):
    return {"company_of_interest": ticker, "trade_date": trade_date, "messages": []}


def test_no_fetch_when_gexter_is_unconfigured(monkeypatch):
    monkeypatch.setattr(f"{_MODULE}.gexter_configured", lambda: False)
    assert fetch_market_structure(_state()) is None


def test_no_fetch_for_a_ticker_outside_the_sp_complex(monkeypatch):
    called = []
    monkeypatch.setattr(f"{_MODULE}.gexter_configured", lambda: True)
    monkeypatch.setattr(f"{_MODULE}.fetch_document", lambda **kw: called.append(kw))
    assert fetch_market_structure(_state(ticker="NVDA")) is None
    assert called == []          # no subprocess for an equity run


def test_fetch_passes_the_states_trade_date(monkeypatch):
    seen = {}
    monkeypatch.setattr(f"{_MODULE}.gexter_configured", lambda: True)
    monkeypatch.setattr(
        f"{_MODULE}.fetch_document",
        lambda **kw: seen.update(kw) or {"symbols": {"SPX": {"status": "ok"}}})
    fetch_market_structure(_state(trade_date="2026-08-27"))
    assert seen["trade_date"] == "2026-08-27"
    assert seen["symbol"] == "SPX"
    assert seen["candidates"] is True
    assert seen["dte_max"] == 2


def test_a_vendor_failure_yields_none_rather_than_aborting(monkeypatch):
    from tradingagents.dataflows.gexter import GexterUnavailableError

    def boom(**kw):
        raise GexterUnavailableError("postgres down")

    monkeypatch.setattr(f"{_MODULE}.gexter_configured", lambda: True)
    monkeypatch.setattr(f"{_MODULE}.fetch_document", boom)
    # Market structure is optional context. A GEXter outage degrades the run;
    # it must never abort the graph.
    assert fetch_market_structure(_state()) is None


def test_the_freshness_gate_is_applied_to_the_fetched_entry(monkeypatch):
    stale = {
        "schema_version": 1,
        "symbols": {"SPX": {
            "status": "ok", "spot_context": {"session_em_points": 47.1},
            "candidates": [{"candidate_id": "x",
                            "quoted_asof": "2020-01-01T13:45:00-05:00"}],
            "candidates_suppressed_reason": None}},
    }
    monkeypatch.setattr(f"{_MODULE}.gexter_configured", lambda: True)
    monkeypatch.setattr(f"{_MODULE}.fetch_document", lambda **kw: stale)
    import datetime as _dt
    doc = fetch_market_structure(
        {"company_of_interest": "^GSPC",
         "trade_date": _dt.date.today().isoformat(), "messages": []})
    entry = doc["symbols"]["SPX"]
    assert entry["candidates"] == []
    assert entry["spot_context"] is not None      # levels survive the gate
